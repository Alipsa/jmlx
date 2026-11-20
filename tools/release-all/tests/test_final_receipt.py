"""Final upload authorization rejects preliminary records and changed verification evidence."""

import copy
import io
import json
import runpy
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from xml.etree import ElementTree
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_receipt as F
from release_inputs import sha256
from release_inputs import MODULE_KEYS
from verify_native_tests import required_tasks, verify_events


class FinalReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        build = self.root / "build/releaseAll"
        build.mkdir(parents=True)
        self.bundle = build / "jmlx-mega-bundle.zip"
        self.bundle.write_bytes(b"verified fixture bundle")
        bundle_hash = sha256(self.bundle)
        def write(name, doc):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc))
            return path
        self.write = write
        plugin = write("build/releaseAll/bundle-receipt.json", {"bundleHash": "bundle"})
        plan = write("build/releaseAll/manifest.json", {})
        events = write("events.json", {"schema": "jmlx-task-evidence-1", "passed": True,
                                      "invocationId": "run", "tasks": [
                                          {"path": task, "outcome": "EXECUTED"} for task in sorted(required_tasks())]})
        junit = self.root / "result.xml"
        junit.write_text('<testsuite tests="1"/>')
        self.record = {"tag": {"sha": "c"*40}, "native": {}, "bundleSha256": bundle_hash,
                       "pluginReceiptSha256": sha256(plugin), "pluginReceipt": {"manifestHash": sha256(plan)},
                       "invocationId": "run", "taskEvidence": {"path": str(events), "sha256": sha256(events)},
                       "junit": {"result.xml": sha256(junit)}}
        self.record["native"] = {"supportedMinimum": [27, 2, 0]}
        argv = write("gradle-argv.json", ["gradlew", "bundleAll", "--no-build-cache"])
        self.record["gradleArgv"] = json.loads(argv.read_text())
        self.record["gradleArgvFile"] = {"path": str(argv), "sha256": sha256(argv)}
        for key, feature in (("releaseJdk", "25"), ("pureJavaJdk", "21")):
            self.record[key] = {"java.feature": feature, "java.version": feature + ".0.1",
                                "java.runtime.version": feature + ".0.1+7", "java.vm.name": "Fixture VM",
                                "java.vendor": "Fixture", "java.home": "/fixture/jdk-" + feature}
        preliminary = write("preliminary.json", self.record)
        self.receipt = {"schema": "jmlx-verified-bundle-1", "bundle": str(self.bundle),
                        "preliminary": {"path": str(preliminary), "sha256": sha256(preliminary)}, "reports": {}}
        for kind in F.REQUIRED:
            env = {"system": "Linux" if kind == "linux-pure" else "Darwin",
                   "architecture": "aarch64" if kind == "linux-pure" else "arm64",
                   "osVersion": "27.2.1", "virtualized": False, "chip": "Apple fixture",
                   "swVers": "macOS 27.2.1", "containerImage": "fixture-image", "osRelease": "Ubuntu fixture"}
            log = self.root / (kind + ".log")
            log.write_text("smoke passed")
            runtime = {"java.feature": "21" if kind.endswith("pure") else "25",
                       "java.version": "21.0.12" if kind.endswith("pure") else "25.0.1",
                       "java.runtime.version": "fixture-runtime+1", "java.vendor": "Fixture JVM",
                       "java.home": "/fixture/jdk", "java.vm.name": "Fixture VM"}
            tree = ElementTree.Element("properties")
            for key, value in runtime.items():
                ElementTree.SubElement(tree, "entry", key=key).text = value
            runtime_file = self.root / (kind + ".runtime.xml")
            ElementTree.ElementTree(tree).write(runtime_file)
            report = write(kind + ".json", {"schema": "jmlx-bundle-smoke-1", "kind": kind, "passed": True,
                "bundleSha256": bundle_hash, "preliminarySha256": sha256(preliminary), "candidateSha": "c"*40,
                "runId": "42", "environment": env,
                **({"workflow": {"ref": "refs/heads/main", "event": "workflow_dispatch",
                     "workflowRef": "Alipsa/jmlx/.github/workflows/release-candidate.yml@refs/heads/main",
                     "sha": "d" * 40}} if kind == "minimum-host" else {}),
                "outputs": {log.name: sha256(log), runtime_file.name: sha256(runtime_file)},
                "consumerRuntime": runtime, "consumerRuntimeFile": runtime_file.name})
            self.receipt["reports"][kind] = {"path": str(report), "sha256": sha256(report)}
        self.path = write("final.json", self.receipt)
        for target, value in (("check_bundle", self.record), ("native_report", self.record["native"])):
            patcher = patch.object(F, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
            if target == "check_bundle":
                self.bundle_checker_patcher = patcher
        patcher = patch.object(F.subprocess, "check_output", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_plugin_digests_are_reviewed_record_anchors_after_full_verification(self):
        self.assertEqual(F.plugin_digests(self.root, self.path),
                         (self.record["bundleSha256"], self.record["pluginReceipt"]["manifestHash"]))
        self.bundle.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            F.plugin_digests(self.root, self.path)

    def test_plugin_digests_cli_stdout_contract(self):
        stdout = io.StringIO()
        with patch("release_inputs.check_bundle", return_value=self.record), \
                patch("release_inputs.native_report", return_value=self.record["native"]), \
                patch.object(sys, "argv", [str(Path(F.__file__)), "--root", str(self.root),
                                          "--verify", str(self.path), "--plugin-digests"]), \
                redirect_stdout(stdout):
            runpy.run_path(F.__file__, run_name="__main__")
        self.assertRegex(stdout.getvalue(), r"\A[0-9a-f]{64}\t[0-9a-f]{64}\n\Z")
        self.assertEqual(stdout.getvalue(),
                         self.record["bundleSha256"] + "\t" +
                         self.record["pluginReceipt"]["manifestHash"] + "\n")

    def test_final_valid_and_preliminary_rejected(self):
        self.assertEqual(F.verify(self.root, self.path), (self.receipt, self.record))
        self.write("final.json", {"schema": "jmlx-preliminary-bundle-1"})
        with self.assertRaisesRegex(ValueError, "only a final"):
            F.verify(self.root, self.path)

    def test_minimum_host_requires_supported_minor_and_accepts_patch_releases(self):
        path = self.root / "minimum-host.json"
        report = json.loads(path.read_text())
        for version in ("27.2", "27.2.0", "27.2.1", "27.2.9", "26.2.1", "27.1.9", "27.3.0"):
            with self.subTest(version=version):
                report["environment"]["osVersion"] = version
                self.write("minimum-host.json", report)
                self.receipt["reports"]["minimum-host"]["sha256"] = sha256(path)
                self.write("final.json", self.receipt)
                if version.split(".")[:2] == ["27", "2"]:
                    F.verify(self.root, self.path)
                else:
                    with self.assertRaisesRegex(ValueError, "minimum-host evidence"):
                        F.verify(self.root, self.path)

    def test_minimum_host_requires_main_workflow_identity(self):
        path = self.root / "minimum-host.json"
        original = json.loads(path.read_text())
        valid = original["workflow"]
        for workflow in (None, {}, [], dict(valid, ref="refs/heads/feature"),
                         dict(valid, event="pull_request"), dict(valid, sha="bad"),
                         dict(valid, workflowRef=valid["workflowRef"].replace("main", "feature"))):
            with self.subTest(workflow=workflow):
                report = dict(original, workflow=workflow)
                self.write(path.name, report)
                self.receipt["reports"]["minimum-host"]["sha256"] = sha256(path)
                self.write("final.json", self.receipt)
                with self.assertRaisesRegex(ValueError, "minimum-host workflow"):
                    F.verify(self.root, self.path)

    def test_all_macos_acceptance_reports_require_supported_os(self):
        for kind in ("macos-full", "macos-pure"):
            path = self.root / (kind + ".json")
            report = json.loads(path.read_text())
            for version in ("26.2.1", "27.1.9", "27.2", "27.2.1", "27.3.0", "28.0", "27", "bad", None):
                with self.subTest(kind=kind, version=version):
                    report["environment"]["osVersion"] = version
                    self.write(path.name, report)
                    self.receipt["reports"][kind]["sha256"] = sha256(path)
                    self.write("final.json", self.receipt)
                    if version in ("27.2", "27.2.1", "27.3.0", "28.0"):
                        F.verify(self.root, self.path)
                    else:
                        with self.assertRaisesRegex(ValueError, "supported macOS"):
                            F.verify(self.root, self.path)
            report["environment"]["osVersion"] = "27.2.1"
            self.write(path.name, report)
            self.receipt["reports"][kind]["sha256"] = sha256(path)
            self.write("final.json", self.receipt)

    def test_missing_reports_changed_plan_and_logs_block_upload(self):
        missing = copy.deepcopy(self.receipt)
        del missing["reports"]["minimum-host"]
        self.write("final.json", missing)
        with self.assertRaisesRegex(ValueError, "report set"):
            F.verify(self.root, self.path)
        self.write("final.json", self.receipt)
        (self.root / "macos-full.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "output changed"):
            F.verify(self.root, self.path)
        (self.root / "build/releaseAll/manifest.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "plan changed"):
            F.verify(self.root, self.path)

    def test_native_payload_verification_must_execute(self):
        events = json.loads((self.root / "events.json").read_text())
        payload = ":jmlx-native-macos-arm64:verifyNativeReleasePayload"
        self.assertIn(payload, {task["path"] for task in events["tasks"]})
        verify_events(events, "run")
        for outcome in ("SKIPPED", "UP-TO-DATE", "FROM-CACHE", "FAILED", "MISSING"):
            with self.subTest(outcome=outcome):
                changed = copy.deepcopy(events)
                if outcome == "MISSING":
                    changed["tasks"] = [task for task in changed["tasks"] if task["path"] != payload]
                else:
                    next(task for task in changed["tasks"] if task["path"] == payload)["outcome"] = outcome
                with self.assertRaisesRegex(ValueError, "task outcomes"):
                    verify_events(changed, "run")

    def test_cached_task_cannot_claim_passed(self):
        path = self.root / "events.json"
        events = json.loads(path.read_text())
        events["tasks"][0]["outcome"] = "FROM-CACHE"
        self.write("events.json", events)
        self.record["taskEvidence"]["sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "task outcomes"):
            F.verify(self.root, self.path)

    def test_actual_plugin_bundle_replacement_is_rejected_even_if_receipt_copy_is_verified(self):
        F.verify(self.root, self.path)
        self.bundle.write_bytes(b"replaced upload bytes")
        with self.assertRaisesRegex(ValueError, "unchanged ZIP configured for plugin upload"):
            F.verify(self.root, self.path)

    def test_receipt_cannot_select_a_different_upload_path(self):
        copy = self.root / "verified-copy.zip"
        copy.write_bytes(self.bundle.read_bytes())
        self.receipt["bundle"] = str(copy)
        self.write("final.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "configured for plugin upload"):
            F.verify(self.root, self.path)

    def test_report_cannot_invent_runtime_properties(self):
        path = self.root / "macos-pure.json"
        report = json.loads(path.read_text())
        report["consumerRuntime"]["java.vendor"] = "Invented vendor"
        self.write("macos-pure.json", report)
        self.receipt["reports"]["macos-pure"]["sha256"] = sha256(path)
        self.write("final.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "runtime measurement"):
            F.verify(self.root, self.path)

    def test_real_bundle_check_does_not_authorize_a_different_plugin_upload_file(self):
        manifest = {"schema": 1, "baselineSha": "a"*40, "audit": "req/audit.md", "modules": [
            {"module": name, "version": "0.1.0", "source": "local"} for name in MODULE_KEYS]}
        manifest_path = self.write("tools/release-smoke/released-8.4.json", manifest)
        with zipfile.ZipFile(self.bundle, "w") as archive:
            for name in MODULE_KEYS:
                base = f"se/alipsa/{name}/0.1.0/{name}-0.1.0"
                for suffix in (".pom", ".module", ".jar", "-sources.jar", "-javadoc.jar"):
                    archive.writestr(base + suffix, "fixture")
                    archive.writestr(base + suffix + ".asc", "signature")
        self.record.update(schema="jmlx-preliminary-bundle-1", bundleSha256=sha256(self.bundle),
                           releaseManifestSha256=sha256(manifest_path))
        self.record["tag"].update(name="jmlx-phase8.4-rc.1", object="b"*40)
        preliminary = self.write("preliminary.json", self.record)
        self.receipt["preliminary"]["sha256"] = sha256(preliminary)
        for kind, info in self.receipt["reports"].items():
            report = json.loads(Path(info["path"]).read_text())
            report.update(bundleSha256=sha256(self.bundle), preliminarySha256=sha256(preliminary))
            self.write(kind + ".json", report)
            info["sha256"] = sha256(info["path"])
        self.write("final.json", self.receipt)
        def git(command, **kwargs):
            if command[-1] == "--porcelain":
                return ""
            if command[3] == "cat-file":
                return "tag\n"
            return ("b"*40 if command[-1] == "refs/tags/jmlx-phase8.4-rc.1" else "c"*40) + "\n"
        self.bundle_checker_patcher.stop()
        with patch.object(F.subprocess, "check_output", side_effect=git):
            F.verify(self.root, self.path)
            copy = self.root / "verified-copy.zip"
            copy.write_bytes(self.bundle.read_bytes())
            self.bundle.write_bytes(b"unverified replacement at actual upload path")
            self.receipt["bundle"] = str(copy)
            self.write("final.json", self.receipt)
            with self.assertRaisesRegex(ValueError, "configured for plugin upload"):
                F.verify(self.root, self.path)


if __name__ == "__main__":
    unittest.main()
