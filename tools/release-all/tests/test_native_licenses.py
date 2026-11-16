"""Native license evidence rejects stale sources and incomplete companion artifacts."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from contextlib import redirect_stdout
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
        self.source = "Copyright © Björn\nPermission and disclaimer\n".encode("utf-8")
        self.digest = hashlib.sha256(self.source).hexdigest()
        self.wheel = self.root / "wheel.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("license.txt", self.source)
        self.inventory = {"wheelSha256": R.sha256(self.wheel), "components": [
            {"component": "example", "source": "https://example.invalid/license",
             "sha256": self.digest}], "packagedTexts": {}}
        self.cached = A.cache_path(self.inventory["components"][0], self.cache)
        self.cached.write_bytes(self.source)
        for name in ("LICENSE", "NOTICE", "LICENSE-APACHE-2.0", "THIRD-PARTY-NOTICES"):
            data = (b"example\n" + self.source if name == "THIRD-PARTY-NOTICES" else self.source)
            (self.legal / name).write_bytes(data)
            self.inventory["packagedTexts"][name] = hashlib.sha256(data).hexdigest()
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
        self.cached.write_bytes(b"unreviewed")
        with self.assertRaisesRegex(ValueError, "source digest differs"):
            self.verify()
        self.cached.write_bytes(self.source)
        self.wheel.write_bytes(b"unreviewed wheel")
        with self.assertRaisesRegex(ValueError, "wheel digest differs"):
            self.verify()

    def test_notice_hash_update_cannot_hide_missing_upstream_attribution(self):
        (self.legal / "THIRD-PARTY-NOTICES").write_bytes(b"example\ndifferent attribution")
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
                                         b"different" if name == changed else (self.legal / name).read_bytes())
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

    def test_non_utf8_locale_reads_notices_as_utf8(self):
        output = self.root / "locale-result.json"
        result = subprocess.run(
            [sys.executable, str(Path(A.__file__)), "--root", str(self.root),
             "--wheel", str(self.wheel), "--cache", str(self.cache), "--output", str(output)],
            env=dict(os.environ, LC_ALL="en_US.ISO8859-1", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0"),
            capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertFalse(json.loads(output.read_bytes())["releaseReady"])

    def test_distinct_urls_with_shared_digest_are_fetched_independently(self):
        first = dict(self.inventory["components"][0], source="https://example.invalid/first")
        second = dict(first, source="https://example.invalid/second")
        with patch.object(A.urllib.request, "urlopen", side_effect=[io.BytesIO(self.source),
                                                                 io.BytesIO(self.source)]) as fetch:
            A.source_bytes(first, self.cache, self.wheel)
            A.source_bytes(second, self.cache, self.wheel)
        self.assertEqual([call.args[0] for call in fetch.call_args_list], [first["source"], second["source"]])
        self.assertNotEqual(A.cache_path(first, self.cache), A.cache_path(second, self.cache))
        third = dict(first, source="https://example.invalid/wrong-digest")
        with patch.object(A.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "source digest differs"):
                A.source_bytes(third, self.cache, self.wheel)
        self.assertFalse(A.cache_path(third, self.cache).exists())

    def test_github_rewrite_and_atomic_cache_write_then_read_only_hit(self):
        entry = dict(self.inventory["components"][0],
                     source="https://github.com/owner/repo/blob/revision/path/blob/LICENSE")
        with patch.object(A.urllib.request, "urlopen", return_value=io.BytesIO(self.source)) as fetch:
            with patch.object(A.os, "replace", wraps=os.replace) as replace:
                self.assertEqual(A.source_bytes(entry, self.cache, self.wheel), self.source)
        fetch.assert_called_once_with(
            "https://raw.githubusercontent.com/owner/repo/revision/path/blob/LICENSE", timeout=45)
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(replace.call_args.args[1], A.cache_path(entry, self.cache))
        self.assertEqual(list(self.cache.glob("*.tmp")), [])
        with patch.object(A.urllib.request, "urlopen", side_effect=AssertionError("unexpected fetch")):
            with patch.object(A.os, "replace", side_effect=AssertionError("unexpected write")):
                with patch.object(Path, "write_bytes", side_effect=AssertionError("unexpected rewrite")):
                    self.assertEqual(A.source_bytes(entry, self.cache, self.wheel), self.source)

    def test_interrupted_cache_replace_leaves_no_partial_entry(self):
        entry = dict(self.inventory["components"][0], source="https://example.invalid/interrupted")
        with patch.object(A.urllib.request, "urlopen", return_value=io.BytesIO(self.source)):
            with patch.object(A.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    A.source_bytes(entry, self.cache, self.wheel)
        self.assertFalse(A.cache_path(entry, self.cache).exists())
        self.assertEqual(list(self.cache.glob("*.tmp")), [])

    def test_stack_exchange_body_and_unexpected_responses(self):
        entry = dict(self.inventory["components"][0], source="https://example.invalid/answer", jsonBody=True)
        response = json.dumps({"items": [{"body": self.source.decode("utf-8")}]}).encode()
        with patch.object(A.urllib.request, "urlopen", return_value=io.BytesIO(response)):
            self.assertEqual(A.source_bytes(entry, self.cache, self.wheel), self.source)
        A.cache_path(entry, self.cache).unlink()
        for document in ({"items": []}, {"error_id": 502}, {"items": [{}]}, {"items": [None]},
                         {"items": [{"body": None}]}, [], {"items": "wrong"}):
            with self.subTest(document=document):
                with patch.object(A.urllib.request, "urlopen",
                                  return_value=io.BytesIO(json.dumps(document).encode())):
                    with self.assertRaisesRegex(ValueError, "unexpected Stack Exchange response: https://"):
                        A.source_bytes(entry, self.cache, self.wheel)
        with patch.object(A.urllib.request, "urlopen", return_value=io.BytesIO(b"not JSON")):
            with self.assertRaisesRegex(ValueError, "unexpected Stack Exchange response"):
                A.source_bytes(entry, self.cache, self.wheel)

    def test_fragments_and_line_ranges_must_be_valid_and_in_the_component_section(self):
        entry = self.inventory["components"][0]
        text = self.source.decode("utf-8")
        texts = {"THIRD-PARTY-NOTICES": "example\n" + text}
        A.check_attribution(dict(entry, noticeLines=[1, 1]), text, texts)
        A.check_attribution(dict(entry, noticeFragments=text.splitlines()), text, texts)
        for bounds in ([2, 1], [0, 1], [1, 3], [], [1], [True, 2], "1,2"):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "invalid notice line range"):
                A.check_attribution(dict(entry, noticeLines=bounds), text, texts)
        for fragments in ([], [""], ["  \n"], [1], "Copyright"):
            with self.subTest(fragments=fragments), self.assertRaises(ValueError):
                A.check_attribution(dict(entry, noticeFragments=fragments), text, texts)
        for target in ("missing", None, [], 1):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "unknown packaged text"):
                A.check_attribution(dict(entry, packagedAs=target), text, texts)
        misplaced = {"THIRD-PARTY-NOTICES": "example\nother text\n" + A.NOTICE_SEPARATOR
                     + "\nother component\n" + text}
        for selectors in ({}, {"noticeLines": [1, 1]}, {"noticeFragments": text.splitlines()}):
            with self.subTest(selectors=selectors), self.assertRaisesRegex(ValueError, "upstream attribution missing"):
                A.check_attribution(dict(entry, **selectors), text, misplaced)

    def test_inventory_cannot_drop_any_required_text(self):
        jar = self.root / "missing.jar"
        for name in R.REQUIRED_NATIVE_LEGAL_TEXTS:
            inventory = dict(self.inventory, packagedTexts=dict(self.inventory["packagedTexts"]))
            del inventory["packagedTexts"][name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "inventory lacks required"):
                R.native_notice_report(jar, inventory)

    def test_license_equals_rules_remain_inside_the_component_section(self):
        text = "Copyright\n========\nPermission\n" + "=" * 72 + "\nDisclaimer\n"
        target = "example\n" + text + A.NOTICE_SEPARATOR + "\nother component\nother text\n"
        self.assertEqual(A.notice_sections(target)["example"], text.rstrip())
        A.check_attribution(self.inventory["components"][0], text,
                            {"THIRD-PARTY-NOTICES": target})

    def test_main_jar_scans_other_entries_but_keeps_native_payload_validation_separate(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        bootstrap = scripts / "bootstrap-native.sh"
        wheel_hash = self.inventory["wheelSha256"]
        pins = {"mlxMetalVersion": "0.32.4", "mlxcCommit": "a" * 40}
        bootstrap.write_text('MLX_METAL_WHEEL_SHA256="' + wheel_hash + '"\n')
        self.inventory["nativePins"] = pins
        (self.legal / "NOTICE").write_text(" ".join((*pins.values(), wheel_hash)), encoding="utf-8")
        self.inventory["packagedTexts"]["NOTICE"] = R.sha256(self.legal / "NOTICE")
        self.save()
        jar = self.root / "native.jar"
        prefix = "se/alipsa/jmlx/native/macos-aarch64/"
        def write_jar(leak=None):
            with zipfile.ZipFile(jar, "w") as archive:
                for name in self.inventory["packagedTexts"]:
                    archive.writestr("META-INF/" + name, (self.legal / name).read_bytes())
                for name in R.NATIVE_FILES:
                    archive.writestr(prefix + name, R.UPSTREAM_PATH if name == "libmlx.dylib" else b"payload")
                if leak:
                    archive.writestr(leak, b"/Users/runner/private/source")
        with patch.object(R, "native_report", return_value={"pins": pins}) as payload_check:
            write_jar()
            R.native_jar_report(jar, bootstrap, ["/Users/runner"])
            self.assertEqual(payload_check.call_args.args[2], ["/Users/runner"])
            payload_check.reset_mock()
            for entry in ("META-INF/MANIFEST.MF", "META-INF/debug.txt", "api/example.class",
                          prefix + "unexpected.txt"):
                with self.subTest(entry=entry):
                    write_jar(entry)
                    with self.assertRaisesRegex(ValueError, "personal path leaked in " + entry):
                        R.native_jar_report(jar, bootstrap, ["/Users/runner"])
            payload_check.assert_not_called()

    def test_native_main_missing_reviewed_text_fails_before_payload_parsing(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        jar = self.root / "native.jar"
        with zipfile.ZipFile(jar, "w") as archive:
            for name in self.inventory["packagedTexts"]:
                if name != "NOTICE":
                    archive.writestr("META-INF/" + name, (self.legal / name).read_bytes())
        with self.assertRaisesRegex(ValueError, "missing native legal text: NOTICE"):
            R.native_jar_report(jar, scripts / "bootstrap-native.sh", [])

    def test_native_cli_checks_companion_notices_and_personal_paths(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        jars = [self.root / f"native-{name}.jar" for name in ("sources", "javadoc")]
        def write_jar(jar, omit=False, leak=False):
            with zipfile.ZipFile(jar, "w") as archive:
                for name in self.inventory["packagedTexts"]:
                    if not omit or name != "NOTICE":
                        archive.writestr("META-INF/" + name, (self.legal / name).read_bytes())
                archive.writestr("api/example.html", b"/Users/release/private/source" if leak else b"API")
        for jar in jars:
            write_jar(jar)
        args = ["release_inputs.py", "native", str(self.root / "native.jar"), "--bootstrap",
                str(scripts / "bootstrap-native.sh"), "--forbid", "/Users/release"]
        for jar in jars:
            args += ["--documentation-jar", str(jar)]
        with patch.object(sys, "argv", args), patch.object(R, "native_jar_report", return_value={}):
            with redirect_stdout(io.StringIO()):
                R.main()
            for jar in jars:
                with self.subTest(jar=jar.name):
                    write_jar(jar, omit=True)
                    with self.assertRaisesRegex(ValueError, "missing native legal text: NOTICE"):
                        R.main()
                    write_jar(jar, leak=True)
                    with self.assertRaisesRegex(ValueError, "personal path leaked in api/example.html"):
                        R.main()
                    write_jar(jar)


if __name__ == "__main__":
    unittest.main()
