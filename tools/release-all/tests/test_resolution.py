"""Run real Gradle repository routing against distinguishable disposable Maven artifacts."""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRADLE = os.environ.get("JMLX_RELEASE_TEST_GRADLE")


@unittest.skipUnless(GRADLE, "set JMLX_RELEASE_TEST_GRADLE to run Gradle integration fixtures")
class ResolutionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jmlx-repositories-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.consumer = self.root / "tools/release-smoke"
        self.consumer.mkdir(parents=True)
        (self.root / "tools/release-all").mkdir()
        shutil.copy(ROOT / "tools/release-all/release_inputs.py", self.root / "tools/release-all")
        for name in ("released-8.4.json", "gradle.properties"):
            shutil.copy(ROOT / "tools/release-smoke" / name, self.consumer)
        manifest_file = self.consumer / "released-8.4.json"
        doc = json.loads(manifest_file.read_text())
        for entry in doc["modules"]:
            if entry["module"] == "jmlx-jinja":
                entry.update(version="0.6.1", source="central")
        manifest_file.write_text(json.dumps(doc))
        self.central = self.root / "central"
        self.local = self.root / "local"
        self.maven_local = self.root / "m2"
        for directory in (self.central, self.local, self.maven_local):
            directory.mkdir()
        text = (ROOT / "tools/release-smoke/repositories.gradle").read_text()
        text = text.replace("https://repo.maven.apache.org/maven2/", self.central.as_uri() + "/")
        (self.consumer / "repositories.gradle").write_text(text)
        (self.consumer / "settings.gradle").write_text("rootProject.name = 'routing-fixture'\n")
        (self.consumer / "build.gradle").write_text("""
apply from: 'repositories.gradle'
configurations { probe { canBeConsumed = false } }
dependencies { probe providers.gradleProperty('coordinate').get() }
tasks.register('resolve') {
    def artifacts = configurations.probe
    doLast {
        def value = artifacts.singleFile.text
        if (value != providers.gradleProperty('expected').get()) throw new GradleException('wrong artifact bytes: ' + value)
    }
}
""")

    def artifact(self, repository, name, version, value):
        directory = repository / "se/alipsa" / name / version
        directory.mkdir(parents=True, exist_ok=True)
        base = name + "-" + version
        (directory / (base + ".jar")).write_text(value)
        (directory / (base + ".pom")).write_text(
            f'<project><modelVersion>4.0.0</modelVersion><groupId>se.alipsa</groupId>'
            f'<artifactId>{name}</artifactId><version>{version}</version></project>')
        (directory / (base + ".module")).write_text(json.dumps({
            "formatVersion": "1.1", "component": {"group": "se.alipsa", "module": name, "version": version},
            "variants": [{"name": "runtime", "files": [{"name": base + ".jar", "url": base + ".jar"}]}]}))

    def resolve(self, name, version, expected, pom=False, success=True):
        args = [GRADLE, "-p", str(self.consumer), "--no-daemon", "--no-build-cache", "resolve",
                "-PjmlxMode=release", "-PsmokeSourceMode=candidate", "-PsmokeRelease=8.4",
                "-PsmokeProfile=pure-java", f"-PjmlxRepoUrl={self.local.as_uri()}",
                f"-D maven.repo.local={self.maven_local}".replace("-D ", "-D"),
                f"-Pcoordinate=se.alipsa:{name}:{version}", f"-Pexpected={expected}"]
        if pom:
            args.append("-PpomOnlyResolution=true")
        result = subprocess.run(args, env=dict(os.environ, GRADLE_USER_HOME=str(self.root / "gradle-home")),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode == 0, success, result.stdout[-7000:])

    def test_source_assignment_and_missing_source_no_fallback(self):
        self.artifact(self.central, "jmlx-jinja", "0.6.1", "central-bytes")
        self.artifact(self.local, "jmlx-jinja", "0.6.1", "wrong-local-bytes")
        self.artifact(self.maven_local, "jmlx-jinja", "0.6.1", "wrong-maven-local-bytes")
        self.artifact(self.local, "jmlx-tokenizer", "0.1.2", "local-bytes")
        self.artifact(self.central, "jmlx-tokenizer", "0.1.2", "wrong-central-bytes")
        self.artifact(self.maven_local, "jmlx-tokenizer", "0.1.2", "wrong-maven-local-bytes")
        for pom in (False, True):
            self.resolve("jmlx-jinja", "0.6.1", "central-bytes", pom)
            self.resolve("jmlx-tokenizer", "0.1.2", "local-bytes", pom)
        shutil.rmtree(self.central / "se/alipsa/jmlx-jinja")
        shutil.rmtree(self.local / "se/alipsa/jmlx-tokenizer")
        for pom in (False, True):
            self.resolve("jmlx-jinja", "0.6.1", "", pom, success=False)
            self.resolve("jmlx-tokenizer", "0.1.2", "", pom, success=False)
            self.resolve("jmlx-unknown", "0.1.0", "", pom, success=False)


if __name__ == "__main__":
    unittest.main()
