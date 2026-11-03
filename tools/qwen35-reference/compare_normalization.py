#!/usr/bin/env python3
"""Compare paired stock/diagnostic captures; never creates acceptance goldens."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stock = json.loads(args.stock.read_text())
    diagnostic = json.loads(args.diagnostic.read_text())
    assert "NEVER acceptance" in diagnostic["kind"]
    for name in ("inputs_sha256", "device", "tf32", "packages", "parameter_dtypes"):
        assert stock[name] == diagnostic[name], name
    if diagnostic.get("teacher_history_sha256"):
        assert diagnostic["teacher_history_sha256"] == digest(args.stock)
    paths = [args.capture_dir / r["captures"]["file"] for r in (stock, diagnostic)]
    for path, record in zip(paths, (stock, diagnostic)):
        assert digest(path) == record["captures"]["sha256"]
    records = []
    with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        phases = sorted({key.removesuffix("_logits") for key in a.files if key.endswith("_logits")})
        for phase in phases:
            keys = [key for key in a.files if key.startswith(phase + "_")]
            differences = {}
            for kind in ("q", "k", "output", "state", "logits"):
                differences[kind] = max((float(np.max(np.abs(a[key]-b[key])))
                                         for key in keys if key.endswith("_"+kind)), default=None)
            logit_a = a[phase+"_logits"][0]
            logit_b = b[phase+"_logits"][0]
            largest = np.sort(logit_a)[-2:]
            margin = float(largest[-1]-largest[-2])
            records.append(dict(phase=phase, normalization_contribution_max_abs=differences,
                                stock_greedy=int(np.argmax(logit_a)),
                                diagnostic_greedy=int(np.argmax(logit_b)),
                                stock_margin=margin,
                                convention_only_margin_eligible=margin > 2*differences["logits"]+1e-4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        kind="normalization attribution only; Java/reference remainder still unmeasured",
        stock_sha256=digest(args.stock), diagnostic_sha256=digest(args.diagnostic),
        source_sha256=digest(Path(__file__)), phases=records), indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
