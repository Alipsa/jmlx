"""The retained RC3 comparator requires a complete, unencrypted native publication set."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
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

    def run_comparison(self, bundle, approval, error=None, bad_pin=None):
        source = self.root / "approval.json"
        source.write_text(json.dumps(approval))
        output = self.root / "comparison.json"
        argv = [str(SOURCE), "--bundle", str(bundle), "--approval", str(source),
                "--output", str(output)]
        stderr = io.StringIO()
        approval_pin = "0" * 64 if bad_pin == "approval" else sha(source.read_bytes())
        bundle_pin = "0" * 64 if bad_pin == "bundle" else sha(bundle.read_bytes())
        with patch.object(self.module, "APPROVAL_SHA256", approval_pin), \
                patch.object(self.module, "BUNDLE_SHA256", bundle_pin), \
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

    def test_archive_digest_and_size_mismatches_are_rejected(self):
        for field, value in (("sha256", "0" * 64), ("bytes", 0)):
            bundle, approval = self.fixture()
            approval["localArtifactsReviewed"][0][field] = value
            with self.subTest(field=field):
                self.run_comparison(bundle, approval, "native archive differs")

    def test_legal_text_digest_and_size_mismatches_are_rejected(self):
        for field, value in (("sha256", "0" * 64), ("bytes", 0)):
            bundle, approval = self.fixture()
            approval["approvedLocalLegalTexts"]["NOTICE"][field] = value
            with self.subTest(field=field):
                self.run_comparison(bundle, approval, "legal text differs")

    def test_empty_or_missing_legal_text_pins_are_rejected(self):
        for value in ({}, None, []):
            bundle, approval = self.fixture()
            approval["approvedLocalLegalTexts"] = value
            with self.subTest(value=value):
                self.run_comparison(bundle, approval, "non-empty native legal text pins")
        del approval["approvedLocalLegalTexts"]
        self.run_comparison(bundle, approval, "non-empty native legal text pins")

    def test_duplicate_metadata_pin_is_rejected(self):
        bundle, approval = self.fixture()
        approval["approvedMetadata"].append(approval["approvedMetadata"][0])
        self.run_comparison(bundle, approval, "duplicate native metadata pin")

    def test_three_archives_with_wrong_names_are_rejected(self):
        bundle, approval = self.fixture()
        approval["localArtifactsReviewed"][0]["path"] = "other.jar"
        self.run_comparison(bundle, approval, "runtime, sources and Javadoc archives")

    def test_reviewed_input_digest_mismatches_are_rejected(self):
        for source in ("approval", "bundle"):
            with self.subTest(source=source):
                self.run_comparison(*self.fixture(),
                                    error="differs from the reviewed RC3 inputs", bad_pin=source)

    def test_metadata_digest_and_size_mismatches_are_rejected(self):
        for field, value in (("sha256", "0" * 64), ("bytes", 0)):
            bundle, approval = self.fixture()
            approval["approvedMetadata"][0][field] = value
            with self.subTest(field=field):
                self.run_comparison(bundle, approval, "native metadata differs")


if __name__ == "__main__":
    unittest.main()
