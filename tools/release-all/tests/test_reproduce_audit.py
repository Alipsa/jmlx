"""The dated reproduction must reject mismatched inputs and comparison evidence."""

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

RECIPE = (Path(__file__).resolve().parents[3]
          / "req/plans/evidence/pure-java-audit-2026-11-16/reproduce.py")
SPEC = importlib.util.spec_from_file_location("reproduce_audit", RECIPE)
R = importlib.util.module_from_spec(SPEC)
PREVIOUS_BYTECODE_SETTING = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPEC.loader.exec_module(R)
finally:
    sys.dont_write_bytecode = PREVIOUS_BYTECODE_SETTING


class ReproduceAuditTest(unittest.TestCase):
    def test_environment_removes_injection_and_controls_locale(self):
        caller = {"ORG_GRADLE_PROJECT_sonatypePassword": "credential",
                  "ORG_GRADLE_PROJECT_releaseVerificationRepository": "unexpected-repository",
                  "ORG_GRADLE_other": "unexpected-option", "FUTURE_INJECTION": "unknown",
                  "JAVA_TOOL_OPTIONS": "unexpected-option", "JDK_JAVA_OPTIONS": "unexpected-option",
                  "GRADLE_OPTS": "unexpected-option", "HTTPS_PROXY": "unapproved-proxy"}
        caller.update(LC_MESSAGES="de_DE", LC_ALL="fr_FR", LANG="sv_SE", LANGUAGE="sv",
                      TZ="Europe/Stockholm", JAVA_HOME="other-jdk", PATH="caller-path", HOME="home",
                      TMPDIR="temporary-directory", USER="user")
        actual = R.build_environment(caller, Path("/comparison-jdk"))
        self.assertEqual(actual.keys(), set(R.ALLOWED_ENVIRONMENT) | R.FIXED_ENVIRONMENT.keys()
                         | {"JAVA_HOME"})
        self.assertEqual({key: actual[key] for key in R.FIXED_ENVIRONMENT}, R.FIXED_ENVIRONMENT)
        self.assertEqual(actual["JAVA_HOME"], "/comparison-jdk")
        self.assertTrue(actual["PATH"].startswith("/comparison-jdk/bin"))
        self.assertEqual(actual["HOME"], "home")
        self.assertEqual(caller["LANG"], "sv_SE")
        command = json.loads((RECIPE.parent / "build-command.json").read_text())
        recorded = command["reproductionEnvironment"]
        self.assertEqual(recorded["fixed"], R.FIXED_ENVIRONMENT)
        self.assertEqual(recorded["inherit"], list(R.ALLOWED_ENVIRONMENT))
        self.assertEqual(set(command["environment"]), {"JAVA_HOME", "PATH_prefix"})
        report = json.loads((RECIPE.parents[3] / "phase8-4-pure-java-audit.json").read_text())
        self.assertEqual(report["reproductionEnvironment"]["inherit"], list(R.ALLOWED_ENVIRONMENT))
        self.assertEqual(report["reproductionEnvironment"]["fixed"], R.FIXED_ENVIRONMENT)
        self.assertEqual(set(report["build"]["environment"]), {"JAVA_HOME", "PATH_prefix"})

    def test_jdk_archive_is_verified_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "jdk.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("jdk/bin/java")
                info.size = 4
                info.mode = 0o755
                bundle.addfile(info, io.BytesIO(b"java"))
            wanted = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(R.extract_jdk_archive(archive, root / "verified", wanted),
                             (root / "verified/jdk").resolve())
            with self.assertRaisesRegex(ValueError, "JDK archive SHA-256"):
                R.extract_jdk_archive(archive, root / "rejected", "0" * 64)
            self.assertFalse((root / "rejected").exists())

    def test_archive_requires_one_jdk_and_rejects_path_traversal(self):
        for names, message in ((["other-file"], "exactly one bin/java"),
                               (["one/bin/java", "two/bin/java"], "exactly one bin/java"),
                               (["../outside-file"], "outside")):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = root / "jdk.tar.gz"
                with tarfile.open(archive, "w:gz") as bundle:
                    for name in names:
                        bundle.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
                wanted = hashlib.sha256(archive.read_bytes()).hexdigest()
                with self.assertRaisesRegex((ValueError, tarfile.FilterError), message):
                    R.extract_jdk_archive(archive, root / "extracted", wanted)
                self.assertFalse((root / "outside-file").exists())

    def test_gradle_version_is_checked(self):
        R.verify_gradle("banner\nGradle 9.9.0\nmore output\n", "9.9.0")
        for output in ("Gradle 9.7.0\n", "", "Gradle 9.9.0\nGradle 9.9.0\n"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "expected Gradle"):
                R.verify_gradle(output, "9.9.0")

    def test_every_daemon_and_compiler_javadoc_identity_is_checked(self):
        jdk = Path("/comparison-jdk").resolve()
        daemon = {"home": str(jdk), "vendor": "BellSoft", "runtimeVersion": "21.0.12.1+1-LTS",
                  "gradleVersion": "9.9.0"}
        tool = {"vendor": "BellSoft Liberica", "languageVersion": "21",
                "runtimeVersion": "21.0.12.1+1-LTS", "jvmVersion": "21.0.12.1+1-LTS"}
        expected = {"fixture": {role: dict(tool) for role in ("compiler", "javadoc")}}
        report = {"pins": {"gradle": "9.9.0"},
                  "validation": {"review": {"freshReproduction": {"daemon": daemon}}},
                  "comparisonJdk": {"compilerAndJavadoc": expected}}
        tools = {"fixture": {role: dict(tool, home=str(jdk)) for role in expected["fixture"]}}

        def log(daemons, selected=tools):
            return "\n".join(["AUDIT_DAEMON=" + json.dumps(item) for item in daemons]
                             + ["AUDIT_TOOLCHAINS=" + json.dumps(selected)])

        R.verify_toolchains(log([daemon, daemon]), report, jdk)
        for key, value in (("vendor", "other vendor"), ("runtimeVersion", "25"),
                           ("home", "/other-jdk"), ("gradleVersion", "9.7.0")):
            with self.subTest(daemon_field=key), self.assertRaisesRegex(ValueError, "daemon identity"):
                R.verify_toolchains(log([daemon, dict(daemon, **{key: value})]), report, jdk)
        for role in ("compiler", "javadoc"):
            for key in ("vendor", "runtimeVersion", "jvmVersion", "languageVersion", "home"):
                changed = copy.deepcopy(tools)
                changed["fixture"][role][key] = "other"
                with self.subTest(role=role, field=key), self.assertRaisesRegex(ValueError, "identity"):
                    R.verify_toolchains(log([daemon], changed), report, jdk)
        for incomplete in ("", "AUDIT_DAEMON=" + json.dumps(daemon)):
            with self.subTest(log=incomplete), self.assertRaisesRegex(ValueError, "missing AUDIT_"):
                R.verify_toolchains(incomplete, report, jdk)

    def test_central_download_hash_is_checked(self):
        data = b"recorded artifact"
        expected = {"artifacts": {"fixture.jar": {
            "rawSha256": {"central": hashlib.sha256(data).hexdigest()}}}}
        R.verify_central("fixture.jar", data, expected)
        with self.assertRaisesRegex(ValueError, "Central SHA-256.*fixture.jar"):
            R.verify_central("fixture.jar", b"changed artifact", expected)

    def test_result_requires_exact_artifacts_and_verdict(self):
        expected = {"allEqual": False, "artifacts": {"fixture.jar": {"equal": True}}}
        R.verify_result(copy.deepcopy(expected), expected)
        for changed in ({"allEqual": True, "artifacts": expected["artifacts"]},
                        {"allEqual": False, "artifacts": {}},
                        {"allEqual": False, "artifacts": {"fixture.jar": {"equal": False}}}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "differs from committed"):
                R.verify_result(changed, expected)


if __name__ == "__main__":
    unittest.main()
