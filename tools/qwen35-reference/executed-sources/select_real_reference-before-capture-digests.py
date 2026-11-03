#!/usr/bin/env python3
"""Select reference-only prompts under the fixed global envelope; preserve every failed case."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    stock_file = root/"evidence/27b-candidates-stock.json"
    diagnostic_file = root/"evidence/27b-candidates-diagnostic.json"
    stock_record = json.loads(stock_file.read_text())
    diagnostic_record = json.loads(diagnostic_file.read_text())
    assert diagnostic_record["teacher_history_sha256"] == digest(stock_file)
    stock = np.load(args.captures/stock_record["captures"]["file"], allow_pickle=False)
    diagnostic = np.load(args.captures/diagnostic_record["captures"]["file"], allow_pickle=False)
    candidates = []
    for case in stock_record["cases"]:
        name = case["name"]
        phases = [("prefill", case["prefill_margin"])] + [
            (f"decode{i}", step["margin"]) for i, step in enumerate(case["decode_steps"])]
        measurements = []
        for phase, gap in phases:
            key = name+"_"+phase+"_logits"
            a, b = stock[key], diagnostic[key]
            measurements.append(dict(phase=phase, gap=gap,
                normalization_contribution_max_abs=float(np.max(np.abs(a-b))),
                stock_greedy=int(a.argmax()), diagnostic_greedy=int(b.argmax())))
        candidates.append(dict(name=name, phases=measurements,
            maximum_normalization=max(p["normalization_contribution_max_abs"] for p in measurements),
            minimum_gap=min(p["gap"] for p in measurements)))
    image = next(c for c in candidates if c["name"] == "image")
    pairs = []
    for candidate in candidates:
        if candidate["name"] == "image":
            continue
        epsilon = .001+max(candidate["maximum_normalization"], image["maximum_normalization"])
        minimum_gap = min(candidate["minimum_gap"], image["minimum_gap"])
        pairs.append(dict(cases=[candidate["name"], "image"], epsilon=epsilon,
                          margin=.0001, minimum_gap=minimum_gap,
                          eligible=minimum_gap > 2*epsilon+.0001))
    eligible = [p for p in pairs if p["eligible"]]
    record = dict(kind="Independent reference-only eligibility; no Java measurements used",
        source_sha256=digest(__file__), stock_sha256=digest(stock_file),
        diagnostic_sha256=digest(diagnostic_file), arithmetic_epsilon=.001, margin=.0001,
        scope="prefill and all eight recorded decode outputs, including the extra post-budget output",
        candidates=candidates, pairs=pairs, selected=eligible[0]["cases"] if eligible else None)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    assert eligible, "Keep these failures; add reference candidates without changing bounds"
    print(record["selected"], eligible[0])


if __name__ == "__main__":
    main()
