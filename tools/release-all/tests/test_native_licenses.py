"""Native license evidence rejects stale sources and incomplete companion artifacts."""

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import audit_native_licenses as A
import release_inputs as R


class NativeLicensesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        (self.root / "req").mkdir()
        self.legal = self.root / "jmlx-native-macos-arm64"
        self.legal.mkdir()
        self.source = b"Copyright example\nPermission and disclaimer\n"
        self.digest = hashlib.sha256(self.source).hexdigest()
        (self.cache / self.digest).write_bytes(self.source)
        self.wheel = self.root / "wheel.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("license.txt", self.source)
        self.inventory = {"wheelSha256": R.sha256(self.wheel), "components": [
            {"component": "example", "source": "https://example.invalid/license",
             "sha256": self.digest}], "packagedTexts": {}}
        for name in ("LICENSE", "NOTICE", "LICENSE-APACHE-2.0", "THIRD-PARTY-NOTICES"):
            (self.legal / name).write_bytes(self.source)
            self.inventory["packagedTexts"][name] = self.digest
        self.audit = {"sourceEvidence": [], "unresolvedFindings": [{"id": "open-license"}]}
        self.save()

    def save(self):
        (self.root / "req/native-license-inventory.json").write_text(json.dumps(self.inventory))
        (self.root / "req/phase8-4-native-license-audit.json").write_text(json.dumps(self.audit))

    def verify(self, jars=()):
        return A.verify(self.root, self.wheel, self.cache, jars)

    def test_valid_sources_and_texts_do_not_clear_unresolved_findings(self):
        result = self.verify()
        self.assertFalse(result["releaseReady"])
        self.assertEqual(result["unresolvedFindings"], ["open-license"])
        self.audit["unresolvedFindings"] = []
        self.save()
        self.assertTrue(self.verify()["releaseReady"])

    def test_corrupt_cache_and_changed_wheel_fail(self):
        (self.cache / self.digest).write_bytes(b"unreviewed")
        with self.assertRaisesRegex(ValueError, "source digest differs"):
            self.verify()
        (self.cache / self.digest).write_bytes(self.source)
        self.wheel.write_bytes(b"unreviewed wheel")
        with self.assertRaisesRegex(ValueError, "wheel digest differs"):
            self.verify()

    def test_notice_hash_update_cannot_hide_missing_upstream_attribution(self):
        (self.legal / "THIRD-PARTY-NOTICES").write_bytes(b"different attribution")
        self.inventory["packagedTexts"]["THIRD-PARTY-NOTICES"] = R.sha256(
            self.legal / "THIRD-PARTY-NOTICES")
        self.save()
        with self.assertRaisesRegex(ValueError, "upstream attribution missing"):
            self.verify()

    def test_wheel_member_has_its_own_digest_and_packaged_text_check(self):
        self.inventory["components"][0].update(archiveMember="license.txt", packagedAs="LICENSE")
        self.save()
        self.verify()
        self.inventory["components"][0]["sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "source digest differs"):
            self.verify()

    def test_all_three_artifacts_require_every_reviewed_text(self):
        jars = [self.root / f"native{suffix}.jar" for suffix in ("", "-sources", "-javadoc")]
        def write_jar(jar, omitted=None, changed=None):
            with zipfile.ZipFile(jar, "w") as archive:
                for name in self.inventory["packagedTexts"]:
                    if name != omitted:
                        archive.writestr("META-INF/" + name,
                                         b"different" if name == changed else self.source)
        for jar in jars:
            write_jar(jar)
        self.assertEqual(len(self.verify(jars)["checkedJars"]), 3)
        for jar in jars:
            for name in self.inventory["packagedTexts"]:
                with self.subTest(jar=jar.name, text=name):
                    write_jar(jar, omitted=name)
                    with self.assertRaisesRegex(ValueError, "missing native legal text"):
                        self.verify(jars)
                    write_jar(jar, changed=name)
                    with self.assertRaisesRegex(ValueError, "legal text differs"):
                        self.verify(jars)
                    write_jar(jar)

    def test_require_clear_cli_writes_findings_and_exits_nonzero(self):
        output = self.root / "result.json"
        args = ["audit_native_licenses.py", "--root", str(self.root), "--wheel", str(self.wheel),
                "--cache", str(self.cache), "--output", str(output), "--require-clear"]
        with patch.object(sys, "argv", args), self.assertRaisesRegex(SystemExit, "open-license"):
            A.main()
        self.assertFalse(json.loads(output.read_text())["releaseReady"])


if __name__ == "__main__":
    unittest.main()
