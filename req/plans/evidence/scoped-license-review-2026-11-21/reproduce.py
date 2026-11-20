#!/usr/bin/env python3
"""Replay the bounded technical evidence for the dated scoped license review."""

import argparse
from collections import Counter
import datetime
from decimal import Decimal
import http.client
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RECIPE = ROOT / 'req/plans/evidence/native-license-resolution-2026-11-16/compare_literals.py'
spec = importlib.util.spec_from_file_location('prior_comparison', RECIPE)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def revisions(raw):
    document = prior.response_json(raw)
    if document.get('has_more') is not False:
        raise ValueError('revision response is incomplete')
    records = []
    for item in document['items']:
        record = {key: item[key] for key in ('post_id', 'revision_number', 'revision_guid',
                  'creation_date', 'content_license', 'body')}
        record['ownerUserId'] = item['user']['user_id']
        records.append(record)
    return canonical(sorted(records, key=lambda record: record['revision_number']))


def magnitude_set(text):
    return {abs(Decimal(value)) for value in prior.literals(text)}


def technical_findings(sources, bundle):
    old = json.loads((RECIPE.parent / 'code-comparison.json').read_text())
    baseline = {record['id']: record for record in old['records']}
    mapping = []
    for answer, kind, function in ((35148199, 'erf', 'my_erff'),
                                   (49743348, 'erfinv', 'my_erfinvf')):
        for item in json.loads(sources[f'revisions-{answer}']):
            blocks = prior.CodeBlocks()
            blocks.feed(item['body'])
            code = prior.function_text(max(blocks.blocks, key=len), function)
            literals = prior.literals(code)
            metal = baseline['mlx-metal-' + kind]['decimalFloatingLiterals']
            cpu = baseline['mlx-cpu-erfinv']['decimalFloatingLiterals']
            mapping.append({
                'answerId': answer, 'revision': item['revision_number'],
                'revisionGuid': item['revision_guid'], 'apiLicense': item['content_license'],
                'date': datetime.datetime.fromtimestamp(item['creation_date'],
                          datetime.timezone.utc).isoformat(),
                'ownerUserId': item['ownerUserId'],
                'bodySha256': prior.digest(item['body'].encode('utf-8')),
                'functionSha256': prior.digest(code.encode('utf-8')),
                'metalLiteralSequenceEqual': metal == literals,
                'metalLiteralMultisetSubset': not (Counter(metal) - Counter(literals)),
                'cpuLiteralMultisetEqual': Counter(cpu) == Counter(literals)
                    if kind == 'erfinv' else None,
            })
    # Explicitly verify both MLX comparison inputs against the existing audited file hashes.
    for name, record in (('mlx-metal', baseline['mlx-metal-erf']),
                         ('mlx-simd', baseline['mlx-cpu-erfinv'])):
        if prior.digest(sources[name]) != record['inputSha256']:
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
            data = archive.extractfile(member).read()
            text = data.decode('ascii')
            release = re.search(r'Cephes Math Library Release [^\n]+', text).group()
            expected = {abs(Decimal(value)) for value in values}
            cephes.append({'member': member, 'sha256': prior.digest(data),
                           'releaseLabel': release,
                           'comparison': 'six selected polynomial coefficient magnitudes; '
                                         'not control flow, sign/evaluation or '
                                         'permission equivalence',
                           'selectedCoefficients': values,
                           'allSelectedMagnitudesInNetlibFile': expected <= magnitude_set(text),
                           'allSelectedMagnitudesInMlxSimdFunction': expected <= magnitude_set(
                               prior.function_text(mlx,
                                   'exp' if function == 'exp2f' else 'sincos'))})
    packaging = []
    with zipfile.ZipFile(bundle) as outer:
        for suffix in ('.jar', '-sources.jar', '-javadoc.jar'):
            name = ('se/alipsa/jmlx-native-macos-arm64/0.1.0/'
                    'jmlx-native-macos-arm64-0.1.0' + suffix)
            data = outer.read(name)
            with zipfile.ZipFile(io.BytesIO(data)) as jar:
                notice = jar.read('META-INF/THIRD-PARTY-NOTICES').decode('utf-8')
                packaging.append({'entry': name, 'sha256': prior.digest(data),
                    'noticeSha256': prior.digest(notice.encode('utf-8')),
                    'noticeContainsErfinv': 'erfinv' in notice,
                    'noticeContainsCcBySa': 'CC BY-SA' in notice,
                    'nativeSourceEntries': [name for name in jar.namelist()
                        if name.endswith(('.h', '.cpp', '.c', '.metal'))]})
    return {'revisionMapping': mapping, 'cephesCoefficientChecks': cephes,
            'rc1Packaging': packaging}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--fetch-missing', action='store_true')
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evidence = json.loads((HERE / 'review.json').read_text())
    if prior.digest(Path(__file__).read_bytes()) != evidence['reproducerSha256']:
        raise ValueError('scoped review recipe changed')
    if prior.digest(RECIPE.read_bytes()) != evidence['priorRecipeSha256']:
        raise ValueError('prior comparison recipe changed')
    if prior.digest(args.bundle.read_bytes()) != evidence['candidate']['bundleSha256']:
        raise ValueError('bundle differs from frozen RC1')
    prior_evidence = RECIPE.parent / 'code-comparison.json'
    if prior.digest(prior_evidence.read_bytes()) != evidence['priorComparisonSha256']:
        raise ValueError('prior comparison evidence changed')
    expected_path = HERE / 'technical-findings.json'
    if prior.digest(expected_path.read_bytes()) != evidence['technicalFindingsSha256']:
        raise ValueError('committed technical report changed')
    sources = {}
    for source in evidence['technicalSources']:
        projection = revisions if source['projection'] == 'revisions' else lambda raw: raw
        sources[source['id']] = prior.read_source(source['url'], source['sha256'], projection,
                                                args.inputs, args.fetch_missing)
    actual = technical_findings(sources, args.bundle)
    expected = json.loads((HERE / 'technical-findings.json').read_text())
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
