"""The retained RC3 comparator requires a complete, unencrypted native publication set."""

import contextlib
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "req/plans/evidence/rc3-final-review-2026-11-23/compare-native.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CompareRc3NativeTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("rc3_comparator", SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(self.module)
        finally:
            sys.dont_write_bytecode = previous
        self.temp = tempfile.TemporaryDirectory(prefix="rc3-native-compare-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, encrypted=False):
        approval = {"localArtifactsReviewed": [], "approvedMetadata": [],
                    "approvedLocalLegalTexts": {"NOTICE": {"bytes": 6, "sha256": sha(b"notice")}}}
        bundle = self.root / "bundle.zip"
        prefix = "se/alipsa/jmlx-native-macos-arm64/0.1.0/"
        with zipfile.ZipFile(bundle, "w") as archive:
            for suffix in (".jar", "-sources.jar", "-javadoc.jar"):
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as jar:
                    jar.writestr("META-INF/NOTICE", b"notice")
                data = stream.getvalue()
                if encrypted and suffix == ".jar":
                    data = bytearray(data)
                    data[6] |= 1
                    data[data.index(b"PK\x01\x02") + 8] |= 1
                    data = bytes(data)
                name = "jmlx-native-macos-arm64-0.1.0" + suffix
                archive.writestr(prefix + name, data)
                approval["localArtifactsReviewed"].append(
                    {"path": name, "bytes": len(data), "sha256": sha(data)})
            for name, suffix in (("native-module.json", ".module"), ("native-pom.xml", ".pom")):
                data = name.encode()
                archive.writestr(prefix + "jmlx-native-macos-arm64-0.1.0" + suffix, data)
                approval["approvedMetadata"].append(
                    {"path": "some/path/" + name, "bytes": len(data), "sha256": sha(data)})
            approval["approvedMetadata"].insert(1, {"path": "literal-cc-pom-comment.txt"})
        return bundle, approval

    def run_comparison(self, bundle, approval, error=None):
        source = self.root / "approval.json"
        source.write_text(json.dumps(approval))
        output = self.root / "comparison.json"
        argv = [str(SOURCE), "--bundle", str(bundle), "--approval", str(source),
                "--output", str(output)]
        stderr = io.StringIO()
        with patch.object(self.module, "APPROVAL_SHA256", sha(source.read_bytes())), \
                patch.object(self.module, "BUNDLE_SHA256", sha(bundle.read_bytes())), \
                patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
            if error:
                with self.assertRaises(SystemExit):
                    self.module.main()
                self.assertIn(error, stderr.getvalue())
                self.assertFalse(output.exists())
                return
            self.module.main()
        report = json.loads(output.read_text())
        self.assertEqual(len(report["nativeArtifacts"]), 3)
        self.assertEqual(len(report["nativeMetadata"]), 2)
        self.assertTrue(report["allFiveApprovedNativeArtifactsMatch"])
        self.assertTrue(report["javadocProvenance"]["archiveMatchesApprovedBytes"])

    def test_metadata_order_is_irrelevant(self):
        self.run_comparison(*self.fixture())

    def test_missing_archive_is_rejected(self):
        bundle, approval = self.fixture()
        approval["localArtifactsReviewed"].pop()
        self.run_comparison(bundle, approval, "exactly three native archives")

    def test_missing_metadata_is_rejected(self):
        bundle, approval = self.fixture()
        approval["approvedMetadata"] = [approval["approvedMetadata"][0]]
        self.run_comparison(bundle, approval, "native POM and module metadata pin")

    def test_encrypted_entries_are_rejected(self):
        self.run_comparison(*self.fixture(encrypted=True), error="encrypted native archive entries")

    def test_revised_admin_checks_survive_optimization(self):
        helper = SOURCE.parent / "window/admin-window.py"
        self.assertFalse(any(isinstance(node, ast.Assert)
                             for node in ast.walk(ast.parse(helper.read_text()))))
        result = subprocess.run(
            [sys.executable, "-O", "-c",
             "import os,runpy,sys; os.geteuid=lambda:1000; "
             "runpy.run_path(sys.argv[1],run_name='__main__')", str(helper)],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Administrator authentication required", result.stderr)


if __name__ == "__main__":
    unittest.main()
