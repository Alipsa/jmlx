"""Native license evidence rejects stale sources and incomplete companion artifacts."""

import hashlib
import importlib.util
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
from native_license_fixtures import install_policy, approved_candidate


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
        for name in R.REQUIRED_NATIVE_LEGAL_TEXTS:
            data = (b"example\n" + self.source + approved_candidate()
                    if name == "THIRD-PARTY-NOTICES" else self.source)
            (self.legal / name).write_bytes(data)
            self.inventory["packagedTexts"][name] = hashlib.sha256(data).hexdigest()
        install_policy(self.root, self.inventory["components"])
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

        entry.update(revisionGuid='reviewed-guid', ownerUserId=780717, contentLicense='CC BY-SA 4.0')
        valid = {'body': self.source.decode(), 'revision_guid': 'reviewed-guid',
                 'user': {'user_id': 780717}, 'content_license': 'CC BY-SA 4.0'}
        for field, wrong in (('revision_guid', 'other'), ('user', {'user_id': 1}),
                             ('content_license', 'CC BY-SA 3.0')):
            response = json.dumps({'items': [dict(valid, **{field: wrong})]}).encode()
            with self.subTest(field=field), patch.object(A.urllib.request, 'urlopen',
                                                       return_value=io.BytesIO(response)):
                with self.assertRaisesRegex(ValueError, 'upstream revision identity differs'):
                    A.source_bytes(entry, self.cache, self.wheel)
                self.assertFalse(A.cache_path(entry, self.cache).exists())

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

class AuthoredNativeLicensesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        install_policy(self.root)
        self.candidate = approved_candidate()
        self.section = self.candidate.split(R.NOTICE_SEPARATOR + b'\n', 1)[1]
        self.title = next(iter(R.REQUIRED_NATIVE_AUTHORED_SECTIONS))
        self.fixtures = Path(__file__).parent / 'fixtures/native-pom'

    def test_raw_sections_preserve_all_bytes_and_last_section_eof(self):
        for ending in (b'\n', b'\r\n'):
            for leading in (b'', ending, b' \t' + ending):
                raw = R.NOTICE_SEPARATOR + ending + leading + b'Title' + ending + b'body' + ending
                sections = R.raw_notice_sections(raw)
                self.assertEqual(sections['Title'], leading + b'Title' + ending + b'body' + ending)
                self.assertNotEqual(sections, R.raw_notice_sections(raw[:-len(ending)]))
        self.assertEqual(R.raw_notice_sections(self.candidate)[self.title], self.section)
        self.assertNotEqual(R.raw_notice_sections(self.candidate),
                            R.raw_notice_sections(self.candidate.replace(b'\n', b'\r\n')))
        for changed in (self.candidate.replace(b'\n\nCreator', b'\nCreator'),
                        self.candidate.replace(b'Creator:', b'Creator: '),
                        self.candidate.replace(b'expression', b'changed expression', 1)):
            self.assertNotEqual(R.raw_notice_sections(self.candidate), R.raw_notice_sections(changed))

    def test_raw_parser_rejects_malformed_and_unsupported_leading_whitespace(self):
        for suffix in (b' ', b'\t', b' \t'):
            for ending in (b'\n', b'\r\n', b''):
                with self.subTest(suffix=suffix, ending=ending), \
                        self.assertRaisesRegex(ValueError, 'malformed-separator at line 2'):
                    R.raw_notice_sections(b'preamble\n' + R.NOTICE_SEPARATOR + suffix + ending
                                         + (b'Title' if ending else b''))
        for whitespace in ('\f', '\v', '\u00a0'):
            with self.subTest(whitespace=whitespace), \
                    self.assertRaisesRegex(ValueError, 'unsupported-leading-whitespace'):
                R.raw_notice_sections(R.NOTICE_SEPARATOR + b'\n' + whitespace.encode() + b'\nTitle\n')
        with self.assertRaisesRegex(ValueError, 'duplicate native notice section'):
            R.raw_notice_sections(self.candidate + self.candidate)
        self.assertEqual(R.raw_notice_sections(R.NOTICE_SEPARATOR), {})
        with self.assertRaises(UnicodeDecodeError):
            R.raw_notice_sections(R.NOTICE_SEPARATOR + b'\n\xff\n')

    def test_authored_policy_and_section_deletion_cannot_hide_behind_inventory_hash(self):
        R.check_authored_attribution(self.candidate, self.root)
        policy_path = self.root / 'req/native-authored-attributions.json'
        policy = json.loads(policy_path.read_text())
        policy['sections'] = []
        policy_path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, 'policy coverage'):
            R.check_authored_attribution(self.candidate, self.root)
        with self.assertRaisesRegex(ValueError, 'required native authored notice section missing'):
            R.check_authored_attribution(b'unrelated section', self.root)
        policy_path.unlink()
        with self.assertRaisesRegex(ValueError, 'policy missing'):
            R.check_authored_attribution(self.candidate, self.root)

    def test_authored_required_fields_scope_digest_and_evidence_are_checked(self):
        for changed in (self.candidate.replace(b'Creator:', b'Author:'),
                        self.candidate.replace(b'No CC adapter', b'A CC adapter'),
                        self.candidate.rstrip(b'\n')):
            with self.subTest(changed=changed[:50]), self.assertRaisesRegex(ValueError, 'authored attribution'):
                R.check_authored_attribution(changed, self.root)
        policy_path = self.root / 'req/native-authored-attributions.json'
        policy = json.loads(policy_path.read_text())
        policy['sections'][0]['requirements'].pop('scope')
        policy_path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, 'required fields differ'):
            R.check_authored_attribution(self.candidate, self.root)
        policy = install_policy(self.root)
        evidence = self.root / policy['evidence'][0]['path']
        evidence.write_bytes(evidence.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'evidence differs'):
            R.check_authored_attribution(self.candidate, self.root)

    def test_cc_license_and_section_remain_required_when_inventory_is_updated(self):
        legal = Path(__file__).resolve().parents[3] / 'jmlx-native-macos-arm64'
        texts = {name: (legal / name).read_bytes() for name in R.REQUIRED_NATIVE_LEGAL_TEXTS}
        jar = self.root / 'native.jar'
        for deleted in ('LICENSE-CC-BY-SA-4.0', 'section'):
            changed = dict(texts)
            if deleted == 'section':
                changed['THIRD-PARTY-NOTICES'] = b'preamble\n'
            else:
                changed.pop(deleted)
            inventory = {'packagedTexts': {n: hashlib.sha256(b).hexdigest() for n, b in changed.items()}}
            with zipfile.ZipFile(jar, 'w') as archive:
                for n, b in changed.items():
                    archive.writestr('META-INF/' + n, b)
            with self.subTest(deleted=deleted), self.assertRaisesRegex(ValueError, 'required'):
                R.native_notice_report(jar, inventory, root=self.root)

    def test_pom_fixtures_cover_complete_scoped_tuples_and_permalink(self):
        for fixture in sorted(self.fixtures.glob('*.pom')):
            with self.subTest(fixture=fixture.name):
                if fixture.name == 'valid.pom':
                    self.assertEqual(R.native_pom_report(fixture.read_bytes())['licenses'], 6)
                else:
                    with self.assertRaises(ValueError):
                        R.native_pom_report(fixture.read_bytes())

    def test_independent_gradle_pom_comparator_on_same_fixtures(self):
        gradle = os.environ.get('JMLX_RELEASE_TEST_GRADLE')
        if not gradle:
            self.skipTest('JMLX_RELEASE_TEST_GRADLE is required for the real Gradle fixture')
        import shutil
        project = self.root / 'gradle-fixture'
        shutil.copytree(self.fixtures, project)
        script = Path(__file__).resolve().parents[2] / 'release-smoke/native-pom-licenses.gradle'
        result = subprocess.run([gradle, '--no-daemon', '-p', str(project),
                                 '-PlicenseScript=' + str(script),
                                 '-PrepositoryRoot=' + str(Path(__file__).resolve().parents[3]),
                                 'verifyFixturePoms', 'verifyLegalMutations'],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)

    def make_release_fixture(self):
        original = Path(__file__).resolve().parents[3]
        def git(*args):
            return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.PIPE)
        git('init', '--quiet', '--initial-branch=main')
        git('remote', 'add', 'origin', 'https://github.com/Alipsa/jmlx.git')
        (self.root / 'baseline.txt').write_text('baseline')
        git('add', 'baseline.txt')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '--quiet', '-m', 'baseline')
        legal = self.root / 'jmlx-native-macos-arm64'
        legal.mkdir(exist_ok=True)
        texts = {}
        for name in R.REQUIRED_NATIVE_LEGAL_TEXTS:
            data = (original / 'jmlx-native-macos-arm64' / name).read_bytes()
            (legal / name).write_bytes(data)
            texts[name] = hashlib.sha256(data).hexdigest()
        (self.root / 'req/native-license-inventory.json').write_text(json.dumps({'packagedTexts': texts}))
        jars = [self.root / ('native' + suffix + '.jar') for suffix in ('', '-sources', '-javadoc')]
        for jar in jars:
            with zipfile.ZipFile(jar, 'w') as archive:
                for name in texts:
                    archive.writestr('META-INF/' + name, (legal / name).read_bytes())
        git('add', 'jmlx-native-macos-arm64/THIRD-PARTY-NOTICES')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '--quiet', '-m', 'approved fixture notice')
        commit = git('rev-parse', 'HEAD').decode().strip()
        git('update-ref', 'refs/remotes/origin/main', commit)
        git('switch', '--quiet', '-c', 'candidate')
        permalink = ('https://github.com/Alipsa/jmlx/blob/' + commit +
                     '/jmlx-native-macos-arm64/THIRD-PARTY-NOTICES')
        # A standalone copy binds the same CLI to synthetic Git history. No checked-out
        # production history, candidate tag, native runtime or remote fetch is needed.
        script = self.root / 'tools/release-all/release_inputs.py'
        script.parent.mkdir(parents=True)
        script.write_text(Path(R.__file__).read_text().replace(R.APPROVED_NOTICE_PERMALINK, permalink))
        spec = importlib.util.spec_from_file_location('fixture_release_inputs', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pom = self.root / 'fixture.pom'
        pom.write_bytes((self.fixtures / 'valid.pom').read_bytes().replace(
            R.APPROVED_NOTICE_PERMALINK.encode(), permalink.encode()))
        return git, jars, module, pom, commit

    def test_shallow_history_is_unavailable_even_when_linked_commit_object_exists(self):
        git, jars, module, pom, commit = self.make_release_fixture()
        for number in (1, 2):
            (self.root / 'baseline.txt').write_text(str(number))
            git('add', 'baseline.txt')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '--quiet', '-m', 'later change')
        shallow = self.root / 'shallow'
        subprocess.check_output(['git', 'clone', '--quiet', '--depth', '1', '--branch',
                                 'candidate', self.root.as_uri(), str(shallow)], stderr=subprocess.PIPE)
        def shallow_git(*args):
            return subprocess.run(['git', '-C', str(shallow), *args],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(shallow_git('fetch', '--depth', '1', self.root.as_uri(), commit).returncode, 0)
        self.assertEqual(shallow_git('cat-file', '-e', commit + '^{commit}').returncode, 0)
        self.assertEqual(shallow_git('rev-parse', '--is-shallow-repository').stdout.strip(), b'true')
        self.assertEqual(shallow_git('merge-base', '--is-ancestor', commit, 'HEAD').returncode, 1)
        self.assertEqual(shallow_git('remote', 'set-url', 'origin',
                                     'https://github.com/Alipsa/jmlx.git').returncode, 0)
        self.assertEqual(shallow_git('update-ref', 'refs/remotes/origin/main', 'HEAD').returncode, 0)
        data = (self.root / 'jmlx-native-macos-arm64/THIRD-PARTY-NOTICES').read_bytes()
        with self.assertRaisesRegex(ValueError, 'history unavailable; shallow checkout'):
            module.verify_notice_permalink(shallow, data)
        # Filling in the same local fixture's history makes this exact check succeed.
        self.assertEqual(shallow_git('fetch', '--unshallow', self.root.as_uri(),
                                     'candidate').returncode, 0)
        self.assertEqual(module.verify_notice_permalink(shallow, data), commit)

    def test_origin_must_match_approved_notice_repository(self):
        git, jars, module, pom, commit = self.make_release_fixture()
        data = (self.root / 'jmlx-native-macos-arm64/THIRD-PARTY-NOTICES').read_bytes()
        for origin in ('https://github.com/Alipsa/jmlx.git', 'git@github.com:Alipsa/jmlx.git',
                       'ssh://git@github.com/Alipsa/jmlx.git', 'https://github.com/alipsa/jmlx'):
            with self.subTest(origin=origin):
                git('remote', 'set-url', 'origin', origin)
                self.assertEqual(module.verify_notice_permalink(self.root, data), commit)
        for origin in ('https://github.com/fork/jmlx.git', 'git@github.com:fork/jmlx.git',
                       'https://example.invalid/Alipsa/jmlx.git',
                       'https://github.com/Alipsa/jmlx-other.git'):
            with self.subTest(origin=origin):
                git('remote', 'set-url', 'origin', origin)
                with self.assertRaisesRegex(ValueError, 'origin must point to Alipsa/jmlx'):
                    module.verify_notice_permalink(self.root, data)

    def test_explicit_cli_and_permalink_history_freshness(self):
        git, jars, R, pom, commit = self.make_release_fixture()
        report = R.native_release_licenses(self.root, pom, jars)
        self.assertEqual(len(report['checkedJars']), 3)
        # Stale local main is irrelevant while the fetched remote and candidate contain it.
        git('update-ref', 'refs/heads/main', 'HEAD^')
        self.assertEqual(R.native_release_licenses(self.root, pom, jars)['noticeCommit'], commit)
        git('update-ref', 'refs/heads/main', commit)
        data = (self.root / 'jmlx-native-macos-arm64/THIRD-PARTY-NOTICES').read_bytes()
        for changed in (data.rstrip(b'\n'), data.replace(b'Creator:', b'Creator: ')):
            with self.assertRaisesRegex(ValueError, 'stale approved notice permalink'):
                R.verify_notice_permalink(self.root, changed)
        # Candidate ancestry is independent from the reviewed-main check.
        with self.assertRaisesRegex(ValueError, r'linked notice commit is not an ancestor of HEAD\^'):
            R.verify_notice_permalink(self.root, data, candidate_ref='HEAD^')
        git('update-ref', 'refs/remotes/origin/main', 'HEAD^')
        with self.assertRaisesRegex(
                ValueError, 'linked notice commit is not an ancestor of refs/remotes/origin/main'):
            R.verify_notice_permalink(self.root, data)
        git('update-ref', 'refs/remotes/origin/main', commit)
        cli = Path(R.__file__).resolve()
        args = [sys.executable, str(cli), 'licenses', '--root', str(self.root), '--pom', str(pom)]
        for jar in jars:
            args += ['--jar', str(jar)]
        good = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        bad_args = [*args]
        bad_args[bad_args.index('--pom') + 1] = str(self.fixtures / 'stale-permalink.pom')
        bad = subprocess.run(bad_args, text=True, capture_output=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn('literal tuples', bad.stderr)
        git('update-ref', 'refs/remotes/origin/main', 'HEAD^')
        unavailable = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(unavailable.returncode, 2)
        self.assertIn('linked notice commit is not an ancestor of refs/remotes/origin/main',
                      unavailable.stderr)
        self.assertNotIn('fetch', unavailable.stderr)
        git('update-ref', '-d', 'refs/remotes/origin/main')
        missing = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn('approved notice history unavailable; fetch origin/main', missing.stderr)



if __name__ == "__main__":
    unittest.main()
