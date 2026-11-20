"""Regression checks for the dated scoped-review evidence replay."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import call, patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'req/plans/evidence/scoped-license-review-2026-11-21'
SPEC = importlib.util.spec_from_file_location('scoped_review', HERE / 'reproduce.py')
C = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'dont_write_bytecode', True):
    SPEC.loader.exec_module(C)
EVIDENCE = json.loads((HERE / 'review.json').read_text())


def revision(number=1):
    return {'post_id': 42, 'revision_number': number, 'revision_guid': f'guid-{number}',
            'creation_date': 100 + number, 'content_license': 'CC BY-SA 4.0',
            'body': '<code>example</code>', 'user': {'user_id': 7}}


def zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class ScopedLicenseReviewTest(unittest.TestCase):
    def test_import_does_not_execute_prior_recipe(self):
        spec = importlib.util.spec_from_file_location('scoped_no_import', HERE / 'reproduce.py')
        module = importlib.util.module_from_spec(spec)
        with patch('builtins.exec', wraps=exec) as execute:
            with patch.object(sys, 'dont_write_bytecode', True):
                spec.loader.exec_module(module)
        self.assertEqual(execute.call_count, 1)  # This module only; the prior recipe is lazy.

    def test_tampered_dependency_cannot_run_or_supply_the_hash_function(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            marker = folder / 'executed'
            recipe = folder / 'prior.py'
            approved = b'value = 1\n'
            recipe.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
                              'def digest(data): return "approved"\n')
            with self.assertRaisesRegex(ValueError, 'prior comparison recipe changed'):
                C.load_prior({'priorRecipeSha256': hashlib.sha256(approved).hexdigest()}, recipe)
            self.assertFalse(marker.exists())
            self.assertFalse((folder / '__pycache__').exists())

    def test_verified_dependency_uses_hashed_bytes_without_rereading_or_bytecode(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe = Path(temp) / 'prior.py'
            approved = b'value = 7\n'
            # No actual file is needed: compile executes the bytes returned by this single read.
            with patch.object(Path, 'read_bytes', side_effect=[approved]) as read:
                module = C.load_prior({'priorRecipeSha256': C.sha256(approved)}, recipe)
            self.assertEqual(module.value, 7)
            self.assertEqual(read.call_count, 1)
            self.assertFalse((Path(temp) / '__pycache__').exists())

    def test_revision_projection_is_sorted_and_ignores_quota_and_profile_changes(self):
        doc = {'has_more': False, 'quota_remaining': 2, 'items': [revision(2), revision(1)]}
        first = C.revisions(json.dumps(doc).encode())
        self.assertEqual([r['revision_number'] for r in json.loads(first)], [1, 2])
        doc['items'].reverse()
        doc['quota_remaining'] = 999
        doc['items'][0]['user']['display_name'] = 'mutable'
        self.assertEqual(first, C.revisions(json.dumps(doc).encode()))

    def test_revision_projection_rejects_malformed_incomplete_and_duplicate_histories(self):
        invalid = [[], {'error_id': 502}, {'has_more': True, 'items': [revision()]},
                   {'has_more': False, 'items': None}, {'has_more': False, 'items': []},
                   {'has_more': False, 'items': [None]}]
        for key, value in [('post_id', True), ('creation_date', 'today'), ('body', None),
                           ('revision_guid', ''), ('content_license', 4), ('user', None),
                           ('revision_number', 0)]:
            item = revision()
            item[key] = value
            invalid.append({'has_more': False, 'items': [item]})
        for items in ([revision(1), revision(1)], [revision(1), revision(3)]):
            invalid.append({'has_more': False, 'items': items})
        mixed = [revision(1), revision(2)]
        mixed[1]['post_id'] = 43
        invalid.append({'has_more': False, 'items': mixed})
        duplicate_guid = [revision(1), revision(2)]
        duplicate_guid[1]['revision_guid'] = duplicate_guid[0]['revision_guid']
        invalid.append({'has_more': False, 'items': duplicate_guid})
        for doc in invalid:
            with self.subTest(document=doc), self.assertRaises(ValueError):
                C.revisions(json.dumps(doc).encode())

    def test_marker_scan_handles_variants_and_function_boundaries(self):
        for text in ('CC BY-SA', 'cc-by-sa', 'Cc_By_Sa', 'CC\u2011BY\u2011SA'):
            with self.subTest(text=text):
                self.assertTrue(C.marker_scan(text)['ccBySa'])
        for text in ('Creative Commons Attribution-ShareAlike',
                     'creative commons attribution share alike'):
            self.assertTrue(C.marker_scan(text)['creativeCommonsShareAlike'])
        self.assertFalse(C.marker_scan('native interface')['erf'])
        self.assertTrue(C.marker_scan('MY_ERFF(x)')['erf'])
        self.assertTrue(C.marker_scan('ERFINVF(x)')['erfinv'])

    def test_nested_bundle_scans_all_meta_inf_files_and_records_notices_and_pom(self):
        sources = {'mlx-expm1f': b'Juffa expm1f license', 'fastermath': b'fastermath license',
                   'neon-math': b'\n' * 6 + b'NEON license',
                   'cephes-readme': b'Netlib Cephes permission'}
        notice = '\n'.join(raw.decode() for raw in sources.values())
        jar = zip_bytes({'META-INF/THIRD-PARTY-NOTICES': notice,
                         'META-INF/NOTICE': 'cc-by-sa',
                         'META-INF/LICENSE': 'Creative Commons Attribution-ShareAlike',
                         'META-INF/extra.txt': 'ERFINV and erf',
                         'native.h': 'source'})
        prefix = 'se/alipsa/jmlx-native-macos-arm64/0.1.0/jmlx-native-macos-arm64-0.1.0'
        files = {prefix + suffix: jar for suffix in ('.jar', '-sources.jar', '-javadoc.jar')}
        files[prefix + '.pom'] = ('<project xmlns="http://maven.apache.org/POM/4.0.0">'
            '<licenses><license><name>MIT License</name><url>https://example.test/mit</url>'
            '<comments>Scoped comment</comments></license></licenses></project>')
        bundle = zip_bytes(files)
        result = C.packaging_findings(bundle, sources)
        self.assertEqual(result['nativePom']['licenses'], [{'name': 'MIT License',
            'url': 'https://example.test/mit', 'comments': 'Scoped comment'}])
        for entry in result['rc1Packaging']:
            self.assertTrue(all(entry['packagedNoticeMatchesSources'].values()))
            self.assertEqual(entry['nativeSourceEntries'], ['native.h'])
            meta = entry['metaInfFiles']
            self.assertTrue(meta['META-INF/NOTICE']['markers']['ccBySa'])
            self.assertTrue(meta['META-INF/LICENSE']['markers']['creativeCommonsShareAlike'])
            self.assertTrue(meta['META-INF/extra.txt']['markers']['erf'])
            self.assertTrue(meta['META-INF/extra.txt']['markers']['erfinv'])
        # Removing a source notice must be observable in the recorded evidence.
        sources['fastermath'] = b'missing copyright and license'
        result = C.packaging_findings(bundle, sources)
        self.assertFalse(result['rc1Packaging'][0]['packagedNoticeMatchesSources']['fastermath'])

    def test_newest_revision_must_match_prior_answer_and_cpu_order_is_explicit(self):
        prior = C.load_prior(EVIDENCE)
        sources, baseline = {}, {}
        for answer, kind, function in ((35148199, 'erf', 'my_erff'),
                                       (49743348, 'erfinv', 'my_erfinvf')):
            code = f'float {function}(float x) {{ return x * 1.0f + 2.0f; }}'
            item = {'post_id': answer, 'revision_number': 1, 'revision_guid': 'one',
                    'content_license': 'CC BY-SA 4.0', 'creation_date': 1, 'ownerUserId': 7,
                    'body': '<code>' + code + '</code>'}
            sources[f'revisions-{answer}'] = json.dumps([item]).encode()
            baseline['stackoverflow-' + kind] = {'inputSha256': C.sha256(code.encode())}
            baseline['mlx-metal-' + kind] = {'decimalFloatingLiterals': ['1.0', '2.0']}
        baseline['mlx-cpu-erfinv'] = {'decimalFloatingLiterals': ['2.0', '1.0']}
        result = C.revision_mapping(sources, baseline, prior)
        self.assertTrue(result[-1]['matchesPriorAuditAnswerCode'])
        self.assertTrue(result[-1]['cpuLiteralMultisetEqual'])
        self.assertFalse(result[-1]['cpuLiteralSequenceEqual'])
        baseline['stackoverflow-erf']['inputSha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'newest revision code differs'):
            C.revision_mapping(sources, baseline, prior)

    def test_main_inspects_the_verified_bundle_bytes_even_if_path_changes(self):
        data = b'the verified ZIP bytes'
        evidence = copy.deepcopy(EVIDENCE)
        evidence['candidate']['bundleSha256'] = C.sha256(data)
        evidence['reproducerSha256'] = C.sha256((HERE / 'reproduce.py').read_bytes())
        evidence['technicalSources'] = []
        original_loads = json.loads
        manifest_text = (HERE / 'review.json').read_text()
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / 'bundle.zip'
            bundle.write_bytes(data)
            output = Path(temp) / 'report.json'
            def inspect(sources, actual, prior):
                bundle.write_bytes(b'replaced after verification')
                self.assertEqual(actual, data)
                return json.loads((HERE / 'technical-findings.json').read_text())
            with patch.object(C, 'load_prior', return_value=object()), \
                    patch.object(C, 'technical_findings', side_effect=inspect), \
                    patch.object(C.json, 'loads', wraps=json.loads) as loads, \
                    patch.object(sys, 'argv', ['reproduce.py', '--inputs', temp,
                                              '--bundle', str(bundle), '--output', str(output)]):
                # Replace only the review manifest; all other JSON uses the real parser.
                loads.side_effect = lambda raw: (evidence if raw == manifest_text
                                                 else original_loads(raw))
                with patch.object(C, 'sha256', wraps=C.sha256) as digest:
                    # Source fetch integrity is exercised by the full replay, not this race test.
                    C.main()
                self.assertTrue(output.exists())
                self.assertEqual(digest.call_args_list.count(call(data)), 1)
