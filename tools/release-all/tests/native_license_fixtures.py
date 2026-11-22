"""Shared synthetic archive setup retaining real approved authored-attribution bytes."""

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]


def install_policy(root, upstream_sources=None):
    policy = json.loads((ROOT / 'req/native-authored-attributions.json').read_text())
    for entry in policy['evidence']:
        target = root / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry['path'], target)
    target = root / policy['decision']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / policy['decision'], target)
    if upstream_sources is not None:
        policy['sections'][0]['upstreamSources'] = upstream_sources
    target = root / 'req/native-authored-attributions.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(policy))
    return policy


def approved_candidate():
    return (ROOT / 'req/plans/evidence/juffa-scope-wording-2026-11-21/'
            'juffa-notice-candidate.txt').read_bytes()
