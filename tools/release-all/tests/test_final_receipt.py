"""Final upload authorization rejects preliminary records and changed verification evidence."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_receipt as F
from release_inputs import sha256
from verify_native_tests import required_tasks


class FinalReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        build = self.root / "build/releaseAll"
        build.mkdir(parents=True)
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
        self.record = {"tag": {"sha": "c"*40}, "native": {}, "bundleSha256": "bundle",
                       "pluginReceiptSha256": sha256(plugin), "pluginReceipt": {"manifestHash": sha256(plan)},
                       "invocationId": "run", "taskEvidence": {"path": str(events), "sha256": sha256(events)},
                       "junit": {"result.xml": sha256(junit)}}
        self.record["native"] = {"minimum": [26, 2, 0]}
        preliminary = write("preliminary.json", self.record)
        self.receipt = {"schema": "jmlx-verified-bundle-1", "bundle": str(self.root / "bundle.zip"),
                        "preliminary": {"path": str(preliminary), "sha256": sha256(preliminary)}, "reports": {}}
        for kind in F.REQUIRED:
            env = {"system": "Linux" if kind == "linux-pure" else "Darwin",
                   "architecture": "aarch64" if kind == "linux-pure" else "arm64",
                   "osVersion": "26.2", "virtualized": False, "chip": "Apple fixture",
                   "swVers": "macOS 26.2", "containerImage": "fixture-image", "osRelease": "Ubuntu fixture"}
            log = self.root / (kind + ".log")
            log.write_text("smoke passed")
            report = write(kind + ".json", {"schema": "jmlx-bundle-smoke-1", "kind": kind, "passed": True,
                "bundleSha256": "bundle", "preliminarySha256": sha256(preliminary), "candidateSha": "c"*40,
                "runId": "42", "environment": env, "outputs": {log.name: sha256(log)},
                "consumerJavaFeature": 21 if kind.endswith("pure") else 25})
            self.receipt["reports"][kind] = {"path": str(report), "sha256": sha256(report)}
        self.path = write("final.json", self.receipt)
        for target, value in (("check_bundle", self.record), ("native_report", self.record["native"])):
            patcher = patch.object(F, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(F.subprocess, "check_output", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_final_valid_and_preliminary_rejected(self):
        F.verify(self.root, self.path)
        self.write("final.json", {"schema": "jmlx-preliminary-bundle-1"})
        with self.assertRaisesRegex(ValueError, "only a final"):
            F.verify(self.root, self.path)

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

    def test_cached_task_cannot_claim_passed(self):
        path = self.root / "events.json"
        events = json.loads(path.read_text())
        events["tasks"][0]["outcome"] = "FROM-CACHE"
        self.write("events.json", events)
        self.record["taskEvidence"]["sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "task outcomes"):
            F.verify(self.root, self.path)


if __name__ == "__main__":
    unittest.main()
