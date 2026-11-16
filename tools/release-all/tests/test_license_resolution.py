"""The dated comparison recipe preserves exact projections and extraction boundaries.

These live release selftests import the recipe from the dated evidence folder below.
Archiving or moving that folder requires updating RECIPE and the recorded verifier/evidence
paths in the same change, or retaining the recipe at its current path.
"""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RECIPE = (Path(__file__).resolve().parents[3]
          / "req/plans/evidence/native-license-resolution-2026-11-16/compare_literals.py")
SPEC = importlib.util.spec_from_file_location("license_resolution", RECIPE)
C = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "dont_write_bytecode", True):
    SPEC.loader.exec_module(C)


class LicenseResolutionTest(unittest.TestCase):
    def test_comment_projection_has_exact_ascii_serialization_and_api_order(self):
        document = {"has_more": False, "items": [
            {"comment_id": 2, "creation_date": 20, "body": "<p>© Björn</p>",
             "owner": {"user_id": 7, "display_name": "ignored"}, "score": 12},
            {"comment_id": 1, "creation_date": 10, "body": "plain"}]}
        expected = (b'[{"body":"<p>\\u00a9 Bj\\u00f6rn</p>","comment_id":2,"creation_date":20,'
                    b'"ownerUserId":7},{"body":"plain","comment_id":1,"creation_date":10,'
                    b'"ownerUserId":null}]')
        self.assertEqual(C.comment_projection(json.dumps(document).encode("utf-8")), expected)
        document["items"].reverse()
        self.assertNotEqual(C.comment_projection(json.dumps(document).encode("utf-8")), expected)

    def test_comment_projection_rejects_partial_or_malformed_responses(self):
        documents = [[], {"error_id": 502}, {"has_more": True, "items": []},
                     {"items": []}, {"has_more": False, "items": None},
                     {"has_more": False, "items": [{"body": None}]},
                     {"has_more": False, "items": [{"body": "x", "owner": None}]}]
        for document in documents:
            with self.subTest(document=document), self.assertRaises(ValueError):
                C.comment_projection(json.dumps(document).encode("utf-8"))

    def test_extraction_starts_at_name_and_only_attached_signs_are_literals(self):
        source = ("float sample(float a);\n"
                  "template <typename T> float sample(float a) {\n"
                  "  // ignored -9.0f\n  if (a) { a = -1.0f + 2.0e-3F; }\n"
                  "  return a - 3.0f + 4 + 0x12;\n}\nfloat other() { return 8.0f; }")
        function = C.function_text(source, "sample")
        self.assertTrue(function.startswith("sample(float a) {"))
        self.assertTrue(function.endswith("}"))
        self.assertNotIn("other", function)
        self.assertEqual(C.literals(function), ["-1.0", "0.0020", "3.0"])
        self.assertEqual(C.literals("1e-5f 2E3f 3.0f"), ["3.0"])
        for source in ("float sample(float a);", "float sample(float a) { return 1.0f;"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                C.function_text(source, "sample")

    def test_answer_projection_decodes_longest_code_and_checks_full_body_digest(self):
        body = "<p>text</p><code>x</code><pre><code>float f() { return 1.0f &lt; 2.0f; }</code></pre>"
        raw = json.dumps({"items": [{"body": body}]}).encode("utf-8")
        self.assertEqual(C.answer_code(raw, C.digest(body.encode("utf-8"))),
                         b"float f() { return 1.0f < 2.0f; }")
        with self.assertRaisesRegex(ValueError, "body differs"):
            C.answer_code(raw, "0" * 64)
        with self.assertRaisesRegex(ValueError, "unexpected Stack Exchange answer"):
            C.answer_code(b'{"items":[]}', "0" * 64)

    def test_url_bound_inputs_rehash_hits_and_never_cache_failed_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary)
            payload = b"example"
            expected = hashlib.sha256(payload).hexdigest()
            urls = ["https://example.invalid/one", "https://example.invalid/two"]
            with patch.object(C.urllib.request, "urlopen", side_effect=[io.BytesIO(payload),
                                                                      io.BytesIO(payload)]) as fetch:
                for url in urls:
                    self.assertEqual(C.read_source(url, expected, lambda raw: raw, inputs, True), payload)
                self.assertEqual(fetch.call_count, 2)
            with patch.object(C.urllib.request, "urlopen", side_effect=AssertionError("unexpected fetch")):
                for url in urls:
                    C.read_source(url, expected, lambda raw: raw, inputs, False)
            for path in inputs.iterdir():
                path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "projected source digest differs"):
                C.read_source(urls[0], expected, lambda raw: raw, inputs, False)
            with patch.object(C.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "projected source digest differs"):
                    C.read_source("https://example.invalid/three", expected, lambda raw: raw, inputs, True)
            self.assertEqual(len(list(inputs.iterdir())), 2)

    def test_cli_reports_missing_sources_and_unexpected_json_types_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "req/plans/evidence/fixture"
            folder.mkdir(parents=True)
            recipe = folder / "compare_literals.py"
            recipe.write_bytes(RECIPE.read_bytes())
            (root / "req/phase8-4-native-license-audit.json").write_text('{"sourceEvidence":[]}')
            cases = [
                ({"sources": []}, {"records": [{"id": "bsd-forum-missing"}]},
                 "research source missing for comparison record: bsd-forum-missing"),
                ({"sources": None}, {"records": [{"id": "bsd-forum-erf"}]},
                 "not iterable"),
                ({"sources": []}, {"records": [None]}, "comparison record must have a string id"),
                ({"sources": []}, {"records": [{"id": 1}]}, "comparison record must have a string id"),
            ]
            for research, comparison, message in cases:
                with self.subTest(research=research, comparison=comparison):
                    (folder / "research.json").write_text(json.dumps(research), encoding="utf-8")
                    (folder / "code-comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, str(recipe), "--root", str(root), "--inputs", str(root / "inputs"),
                         "--output", str(root / "output.json")], capture_output=True, text=True,
                        encoding="utf-8", timeout=15)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("license comparison: ", result.stderr)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
