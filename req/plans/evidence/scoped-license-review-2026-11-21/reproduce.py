#!/usr/bin/env python3
"""Replay the bounded technical evidence for the dated scoped license review."""

import argparse
from collections import Counter
import datetime
from decimal import Decimal
import hashlib
import http.client
import io
import json
from pathlib import Path
import re
import tarfile
import types
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RECIPE = ROOT / 'req/plans/evidence/native-license-resolution-2026-11-16/compare_literals.py'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_prior(evidence, recipe=None):
    recipe = RECIPE if recipe is None else recipe
    data = recipe.read_bytes()
    if sha256(data) != evidence['priorRecipeSha256']:
        raise ValueError('prior comparison recipe changed')
    # Execute the same bytes that were hashed; compilation does not write __pycache__.
    module = types.ModuleType('prior_comparison')
    module.__file__ = str(recipe)
    exec(compile(data, str(recipe), 'exec'), module.__dict__)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def revisions(raw):
    document = json.loads(raw)
    if not isinstance(document, dict) or 'error_id' in document:
        raise ValueError('unexpected revision response')
    items = document.get('items')
    if document.get('has_more') is not False or not isinstance(items, list) or not items:
        raise ValueError('revision response is incomplete or empty')
    records = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('unexpected revision record')
        for key in ('post_id', 'revision_number', 'creation_date'):
            if type(item.get(key)) is not int or item[key] < 1:
                raise ValueError('invalid revision integer: ' + key)
        for key in ('revision_guid', 'content_license', 'body'):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError('invalid revision string: ' + key)
        user = item.get('user')
        if not isinstance(user, dict) or type(user.get('user_id')) is not int:
            raise ValueError('invalid revision owner')
        record = {key: item[key] for key in ('post_id', 'revision_number', 'revision_guid',
                  'creation_date', 'content_license', 'body')}
        record['ownerUserId'] = user['user_id']
        records.append(record)
    numbers = sorted(record['revision_number'] for record in records)
    if numbers != list(range(1, len(records) + 1)):
        raise ValueError('revision numbers must be unique and contiguous from 1')
    if len({record['post_id'] for record in records}) != 1:
        raise ValueError('revision response mixes posts')
    if len({record['revision_guid'] for record in records}) != len(records):
        raise ValueError('revision GUIDs must be unique')
    return canonical(sorted(records, key=lambda record: record['revision_number']))


def magnitude_set(text, prior):
    return {abs(Decimal(value)) for value in prior.literals(text)}


def revision_mapping(sources, baseline, prior):
    mapping = []
    for answer, kind, function in ((35148199, 'erf', 'my_erff'),
                                   (49743348, 'erfinv', 'my_erfinvf')):
        items = json.loads(sources[f'revisions-{answer}'])
        if any(item['post_id'] != answer for item in items):
            raise ValueError('revision list belongs to a different answer')
        newest = max(item['revision_number'] for item in items)
        for item in items:
            blocks = prior.CodeBlocks()
            blocks.feed(item['body'])
            block = max(blocks.blocks, key=len).encode('utf-8')
            matches_audit = sha256(block) == baseline['stackoverflow-' + kind]['inputSha256']
            if item['revision_number'] == newest and not matches_audit:
                raise ValueError('newest revision code differs from prior audit answer')
            code = prior.function_text(block.decode('utf-8'), function)
            literals = prior.literals(code)
            metal = baseline['mlx-metal-' + kind]['decimalFloatingLiterals']
            cpu = baseline['mlx-cpu-erfinv']['decimalFloatingLiterals']
            mapping.append({
                'answerId': answer, 'revision': item['revision_number'],
                'revisionGuid': item['revision_guid'], 'apiLicense': item['content_license'],
                'date': datetime.datetime.fromtimestamp(item['creation_date'],
                          datetime.timezone.utc).isoformat(),
                'ownerUserId': item['ownerUserId'],
                'bodySha256': sha256(item['body'].encode('utf-8')),
                'codeBlockSha256': sha256(block), 'matchesPriorAuditAnswerCode': matches_audit,
                'functionSha256': sha256(code.encode('utf-8')),
                'metalLiteralSequenceEqual': metal == literals,
                'metalLiteralMultisetSubset': not (Counter(metal) - Counter(literals)),
                'cpuLiteralSequenceEqual': cpu == literals if kind == 'erfinv' else None,
                'cpuLiteralMultisetEqual': Counter(cpu) == Counter(literals)
                    if kind == 'erfinv' else None,
            })
    return mapping


def technical_findings(sources, bundle_bytes, prior, old):
    baseline = {record['id']: record for record in old['records']}
    mapping = revision_mapping(sources, baseline, prior)
    # Explicitly verify both MLX comparison inputs against the existing audited file hashes.
    for name, record in (('mlx-metal', baseline['mlx-metal-erf']),
                         ('mlx-simd', baseline['mlx-cpu-erfinv'])):
        if sha256(sources[name]) != record['inputSha256']:
            raise ValueError('MLX source differs from original comparison')
    mlx = sources['mlx-simd'].decode('utf-8')
    coefficients = {
        'exp2f': ['1.535336188319500e-4', '1.339887440266574e-3',
                 '9.618437357674640e-3', '5.550332471162809e-2',
                 '2.402264791363012e-1', '6.931472028550421e-1'],
        'sinf': ['-1.9515295891e-4', '8.3321608736e-3', '-1.6666654611e-1',
                '2.443315711809948e-5', '-1.388731625493765e-3',
                '4.166664568298827e-2'],
    }
    cephes = []
    with tarfile.open(fileobj=io.BytesIO(sources['cephes-single']), mode='r:gz') as archive:
        for function, values in coefficients.items():
            member = './' + function + '.c'
            data, text, release = cephes_member(archive, member)
            expected = {abs(Decimal(value)) for value in values}
            cephes.append({'member': member, 'sha256': sha256(data),
                           'releaseLabel': release,
                           'comparison': 'six selected polynomial coefficient magnitudes; '
                                         'not control flow, sign/evaluation or '
                                         'permission equivalence',
                           'selectedCoefficients': values,
                           'allSelectedMagnitudesInNetlibFile':
                               expected <= magnitude_set(text, prior),
                           'allSelectedMagnitudesInMlxSimdFunction': expected <= magnitude_set(
                               prior.function_text(mlx,
                                   'exp' if function == 'exp2f' else 'sincos'), prior)})
    references = {
        'fastermathDescribesCephes28': bool(re.search(
            r'based on the cephes math library\s+2\.8',
            sources['fastermath'].decode('utf-8'), re.I)),
        'mlxSimdCitesFastermath': 'github.com/akohlmey/fastermath/blob/master/src/exp.c' in mlx,
        'mlxSimdCitesNeonMath': 'github.com/JishinMaster/simd_utils/blob/' in mlx
                              and '/neon_mathfun.h' in mlx,
    }
    return {'revisionMapping': mapping, 'cephesCoefficientChecks': cephes,
            'sourceReferences': references, **packaging_findings(bundle_bytes, sources)}


def cephes_member(archive, member):
    info = archive.getmember(member)
    if not info.isfile():
        raise ValueError('Cephes member is not a regular file: ' + member)
    with archive.extractfile(info) as handle:
        data = handle.read()
    text = data.decode('ascii')
    match = re.search(r'Cephes Math Library Release [^\n]+', text)
    if match is None:
        raise ValueError('Cephes release label missing: ' + member)
    return data, text, match.group()


# Presence markers may overlap: "inverse error function" sets both erf and erfinv.
# A hit flags text for review; it does not establish attribution to a specific function.
MARKERS = {
    'erf': r'\b(?:(?:my_)?erfc?f?|error[\s_\-\u2010-\u2015]+functions?)\b',
    'erfinv': r'\b(?:(?:my_)?erf_?inv(?:erse)?_?f?'
              r'|inverse[\s_\-\u2010-\u2015]+error[\s_\-\u2010-\u2015]+functions?)\b',
    'ccBySa': r'\bcc[\s_\-\u2010-\u2015]*by[\s_\-\u2010-\u2015]*sa\b',
    'creativeCommonsShareAlike': r'\bcreative\s+commons\s+attribution'
                                r'[\s_\-\u2010-\u2015]*share[\s_\-\u2010-\u2015]*alike\b',
    'ccUrl': r'creativecommons\.org/licenses/by-sa\b',
    'shareAlike': r'\bshare[\s_\-\u2010-\u2015]*alike\b',
    'bySa': r'\bby[\s_\-\u2010-\u2015]*sa\b',
    'stackOverflow': r'\bstackoverflow\.com\b|\b(?:35148199|49743348)\b',
    'juffa': r'\bjuffa\b',
    'expm1f': r'\bexpm1f\b',
    'fastermath': r'\bfastermath\b',
    'neonMath': r'\b(?:simd_utils|neon_mathfun)\b',
    'cephes': r'\bcephes\b',
    'moshier': r'\bmoshier\b',
    'netlib': r'\bnetlib\b',
}


def marker_scan(text):
    return {name: bool(re.search(pattern, text, re.I)) for name, pattern in MARKERS.items()}


def normalized_text(text):
    return ' '.join(text.split())


def packaging_findings(bundle_bytes, sources):
    # Match the actual packaged notice text to the independently pinned source fragments.
    fragments = {'juffaExpm1f': ('mlx-expm1f', 1, 33),
                 'fastermath': ('fastermath', 1, 28),
                 'neonMath': ('neon-math', 7, 26),
                 'cephesPermission': ('cephes-readme', 1, None)}
    packaging = []
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as outer:
        prefix = 'se/alipsa/jmlx-native-macos-arm64/0.1.0/jmlx-native-macos-arm64-0.1.0'
        for suffix in ('.jar', '-sources.jar', '-javadoc.jar'):
            entry = prefix + suffix
            data = outer.read(entry)
            with zipfile.ZipFile(io.BytesIO(data)) as jar:
                notice = jar.read('META-INF/THIRD-PARTY-NOTICES').decode('utf-8')
                source_matches = {}
                for name, (source, first, last) in fragments.items():
                    lines = sources[source].decode('utf-8').splitlines()[first - 1:last]
                    fragment = normalized_text('\n'.join(lines))
                    source_matches[name] = bool(fragment) and fragment in normalized_text(notice)
                meta = {}
                for member in jar.infolist():
                    if member.filename.upper().startswith('META-INF/') and not member.is_dir():
                        raw = jar.read(member)
                        meta[member.filename] = {'sha256': sha256(raw),
                            'markers': marker_scan(raw.decode('utf-8', errors='replace'))}
                packaging.append({'entry': entry, 'sha256': sha256(data),
                    'noticeSha256': sha256(notice.encode('utf-8')),
                    'packagedNoticeMatchesSources': source_matches,
                    'metaInfFiles': meta,
                    'nativeSourceEntries': [member for member in jar.namelist()
                        if member.lower().endswith(('.h', '.hpp', '.c', '.cc', '.cpp',
                                                   '.m', '.mm', '.inc', '.metal'))]})
        pom = outer.read(prefix + '.pom')
        ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
        licenses = [{key: item.findtext('m:' + key, namespaces=ns)
                     for key in ('name', 'url', 'comments')}
                    for item in ET.fromstring(pom).findall('m:licenses/m:license', ns)]
    return {'rc1Packaging': packaging, 'nativePom': {'sha256': sha256(pom), 'licenses': licenses}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--fetch-missing', action='store_true')
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evidence = json.loads((HERE / 'review.json').read_text())
    # Drift check against the reviewed manifest; not authentication of this script.
    if sha256(Path(__file__).read_bytes()) != evidence['reproducerSha256']:
        raise ValueError('scoped review recipe changed')
    prior = load_prior(evidence)
    bundle_bytes = args.bundle.read_bytes()
    if sha256(bundle_bytes) != evidence['candidate']['bundleSha256']:
        raise ValueError('bundle differs from frozen RC1')
    prior_evidence = RECIPE.parent / 'code-comparison.json'
    prior_bytes = prior_evidence.read_bytes()
    if sha256(prior_bytes) != evidence['priorComparisonSha256']:
        raise ValueError('prior comparison evidence changed')
    expected_path = HERE / 'technical-findings.json'
    expected_bytes = expected_path.read_bytes()
    if sha256(expected_bytes) != evidence['technicalFindingsSha256']:
        raise ValueError('committed technical report changed')
    sources = {}
    for source in evidence['technicalSources']:
        projection = revisions if source['projection'] == 'revisions' else lambda raw: raw
        data = prior.read_source(source['url'], source['sha256'], projection,
                                 args.inputs, args.fetch_missing)
        if sha256(data) != source['sha256']:
            raise ValueError('source digest differs: ' + source['id'])
        sources[source['id']] = data
    actual = technical_findings(sources, bundle_bytes, prior, json.loads(prior_bytes))
    expected = json.loads(expected_bytes)
    if actual != expected:
        raise ValueError('technical findings differ from the committed review')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(actual, indent=2) + '\n')
    print('Seven revision comparisons, two Cephes checks and three RC1 JAR checks reproduced; '
          'this is technical evidence, not legal clearance.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, tarfile.TarError,
            zipfile.BadZipFile, http.client.HTTPException) as error:
        raise SystemExit(f'scoped license review: {error}') from error
