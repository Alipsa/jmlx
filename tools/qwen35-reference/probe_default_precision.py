#!/usr/bin/env python3
"""Compare ordinary mlx-vlm precision with qualified FLOAT32 under identical histories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    candidates = json.loads((root/"evidence/27b-candidate-inputs.json").read_text())
    selected = json.loads((root/"evidence/27b-reference-selection.json").read_text())["selected"]
    inputs_file = root/"evidence/27b-default-precision-inputs.json"
    inputs_file.write_text(json.dumps(dict(kind="Default precision; same independently selected acceptance inputs",
        source_sha256=digest(__file__), cases=[c for c in candidates["cases"] if c["name"] in selected]), indent=2)+"\n")
    default_file = root/"evidence/27b-default-precision-stock.json"
    qualified_file = root/"evidence/27b-candidates-stock.json"
    environment = dict(os.environ)
    environment.pop("MLX_ENABLE_TF32", None)
    subprocess.run([str(root/"mlx/.venv/bin/python"), str(root/"probe_reload.py"),
        "--model", str(args.model), "--inputs", str(inputs_file), "--output", str(default_file),
        "--logits-only", "--decode-steps", "8", "--teacher-history-from", str(qualified_file),
        "--capture-dir", str(args.captures)], env=environment, check=True)
    ordinary = json.loads(default_file.read_text())
    qualified = json.loads(qualified_file.read_text())
    assert not ordinary["floating_parameter_upcast"] and ordinary["tf32"] == "unset"
    qualified_capture = args.captures/qualified["captures"]["file"]
    assert digest(qualified_capture) == qualified["captures"]["sha256"]
    a = np.load(args.captures/ordinary["captures"]["file"], allow_pickle=False)
    b = np.load(qualified_capture, allow_pickle=False)
    phases = []
    for case in ordinary["cases"]:
        for phase in ["prefill"]+[f"decode{i}" for i in range(8)]:
            key = case["name"]+"_"+phase+"_logits"
            phases.append(dict(phase=key, maximum_precision_difference=float(np.max(np.abs(a[key]-b[key]))),
                default_greedy=int(a[key].argmax()), qualified_float32_greedy=int(b[key].argmax())))
    args.output.write_text(json.dumps(dict(kind="Precision-configuration comparison only; qualified acceptance remains FLOAT32/TF32 off",
        source_sha256=digest(__file__), default_record_sha256=digest(default_file),
        qualified_record_sha256=digest(qualified_file),
        default_parameter_dtypes=ordinary["parameter_dtypes"], qualified_parameter_dtypes=qualified["parameter_dtypes"],
        maximum_logit_difference=max(p["maximum_precision_difference"] for p in phases),
        all_recorded_choices_agree=all(p["default_greedy"] == p["qualified_float32_greedy"] for p in phases),
        scope="Same FLOAT32-stock teacher histories; free-running equivalence follows only while all earlier choices agree",
        phases=phases), indent=2)+"\n")


if __name__ == "__main__":
    main()
