#!/usr/bin/env python3
"""Report Java/stock and normalization-only remainder separately; never acceptance goldens."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-report", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    java = json.loads(args.java_report.read_text())
    stock_record = json.loads((root/"evidence/27b-candidates-stock.json").read_text())
    diagnostic_record = json.loads((root/"evidence/27b-candidates-diagnostic.json").read_text())
    stock_file = args.captures/stock_record["captures"]["file"]
    diagnostic_file = args.captures/diagnostic_record["captures"]["file"]
    assert digest(stock_file) == stock_record["captures"]["sha256"]
    assert digest(diagnostic_file) == diagnostic_record["captures"]["sha256"]
    stock, diagnostic = np.load(stock_file, allow_pickle=False), np.load(diagnostic_file, allow_pickle=False)
    phases = []
    for case in java["cases"]:
        for step in case["phases"]:
            phase = step["phase"]
            path = args.java_report.parent/step["actual_vector_file"]
            assert digest(path) == step["actual_vector_sha256"]
            actual = np.fromfile(path, dtype="<f4")
            original, aligned = stock[phase+"_logits"][0], diagnostic[phase+"_logits"][0]
            phases.append(dict(phase=phase,
                java_stock_max_abs=float(np.max(np.abs(actual-original))),
                normalization_contribution_max_abs=float(np.max(np.abs(aligned-original))),
                java_diagnostic_remainder_max_abs=float(np.max(np.abs(actual-aligned))),
                stock_greedy=int(original.argmax()), diagnostic_greedy=int(aligned.argmax()),
                java_greedy=int(actual.argmax()), actual_sha256=digest(path)))
    record = dict(kind="Attribution only, not an acceptance reference or golden; stock acceptance is reported independently",
        source_sha256=digest(__file__), java_report_sha256=digest(args.java_report),
        java_source_commit=java["git_commit"], stock_capture_sha256=digest(stock_file),
        diagnostic_capture_sha256=digest(diagnostic_file), phases=phases,
        maximum_java_stock_error=max(p["java_stock_max_abs"] for p in phases),
        maximum_normalization_contribution=max(p["normalization_contribution_max_abs"] for p in phases),
        maximum_java_diagnostic_remainder=max(p["java_diagnostic_remainder_max_abs"] for p in phases),
        independently_stated_arithmetic_allowance=java["arithmetic_epsilon"])
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print({k: v for k, v in record.items() if k.startswith("maximum_")})


if __name__ == "__main__":
    main()
