#!/usr/bin/env python3
"""Qualify stored affine weights with independent HF dequantization and stock MLX references.

The separately labeled diagnostic contributes attribution/eligibility measurements only. Its
arrays are never exported as acceptance goldens. Bounds and selection precede Java comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import Qwen3_5ForConditionalGeneration


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Choose an empty output directory; prior evidence is never overwritten")
    args.output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    cases = [dict(name=f"text{i}", input_ids=[[1]+[3+(i*7+j*13) % 90 for j in range(3+i % 9)]])
             for i in range(16)]
    inputs = args.output_dir / "inputs.json"
    inputs.write_text(json.dumps(dict(cases=cases), indent=2)+"\n")
    arrays, records = {}, []
    for family in ("vlm", "lm"):
        checkpoint = root / "fixtures" / (family+"-affine")
        for diagnostic in ((False, True) if family == "vlm" else (False,)):
            label = family + ("-diagnostic" if diagnostic else "-stock")
            command = [str(root/"mlx/.venv/bin/python"), str(root/"probe_reload.py"),
                       "--family", family, "--model", str(checkpoint), "--inputs", str(inputs),
                       "--output", str(args.output_dir/(label+".json")), "--decode-steps", "2"]
            if diagnostic:
                command += ["--normalization-diagnostic", "--teacher-history-from",
                            str(args.output_dir/"vlm-stock.json")]
            else:
                command += ["--greedy-history"]
            subprocess.run(command, env={**os.environ, "MLX_ENABLE_TF32": "0"}, check=True)
        stock_record = json.loads((args.output_dir/(family+"-stock.json")).read_text())
        stock = np.load(args.output_dir/(family+"-stock.npz"))
        diagnostic = np.load(args.output_dir/"vlm-diagnostic.npz") if family == "vlm" else None
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            root/"fixtures/hf", dtype=torch.float32, attn_implementation="eager").eval()
        raw = {}
        for shard in sorted(checkpoint.glob("*.safetensors")):
            raw.update(load_file(shard))
        state = {}
        for name, value in raw.items():
            if not name.startswith("language_model.") or name.endswith((".scales", ".biases")):
                continue
            if value.dtype == np.uint32:
                base = name[:-6]
                bits = ((value[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15)
                values = bits.reshape(value.shape[0], -1).astype(np.float32).reshape(value.shape[0], -1, 64)
                value = (values*raw[base+"scales"][..., None]+raw[base+"biases"][..., None]).reshape(value.shape[0], -1)
            else:
                value = value.astype(np.float32)
            if name.startswith("language_model.model."):
                key = "model.language_model."+name[len("language_model.model."):]
            else:
                key = "lm_head."+name[len("language_model.lm_head."):]
            if name.endswith("conv1d.weight"):
                value = value.transpose(0, 2, 1)
            if key.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight")) or key == "model.language_model.norm.weight":
                value = value-1
            state[key] = torch.from_numpy(value.copy())
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("model.visual.") for k in missing)
        with torch.no_grad():
            for case in stock_record["cases"]:
                stem = family+"."+case["name"]
                ids = torch.tensor(case["input_ids"], dtype=torch.int32)
                output = model(input_ids=ids, use_cache=True)
                cache = output.past_key_values
                phases = [("prefill", None, case["prefill_margin"])] + [
                    (f"decode{i}", step["input_id"], step["margin"])
                    for i, step in enumerate(case["decode_steps"])]
                steps = []
                for phase, token, margin in phases:
                    if token is not None:
                        ids = torch.tensor([[token]], dtype=torch.int32)
                        output = model(input_ids=ids, past_key_values=cache, use_cache=True)
                    hf_logits = output.logits[:, -1].numpy()
                    stock_logits = stock[case["name"]+"_"+phase+"_logits"]
                    contribution = float(np.max(np.abs(stock_logits-diagnostic[case["name"]+"_"+phase+"_logits"]))) if diagnostic is not None else 0.0
                    remainder = float(np.max(np.abs(hf_logits-(diagnostic[case["name"]+"_"+phase+"_logits"] if diagnostic is not None else stock_logits))))
                    assert remainder < 1e-4, (stem, phase, remainder)
                    # Declare fixed independent arithmetic remainder, plus measured normalization.
                    epsilon = contribution+1e-4
                    eligible = margin > 2*epsilon+1e-5
                    arrays[stem+"."+phase+".ids"] = ids.numpy().copy()
                    arrays[stem+"."+phase+".hf_logits"] = hf_logits.copy()
                    arrays[stem+"."+phase+".stock_logits"] = stock_logits.copy()
                    steps.append(dict(phase=phase, normalization_max_abs=contribution,
                                      independent_hf_reference_remainder=remainder,
                                      stock_epsilon=epsilon, stability_margin=1e-5,
                                      top_two_gap=margin, eligible=eligible,
                                      greedy=int(stock_logits.argmax())))
                records.append(dict(name=stem, family=family, steps=steps,
                                    eligible=all(s["eligible"] for s in steps)))
        assert any(r["eligible"] for r in records if r["family"] == family)
    target = args.output_dir/"acceptance.safetensors"
    save_file(arrays, target, metadata={"reference": "independent HF same stored packed weights and unmodified stock MLX; diagnostic arrays excluded"})
    record = dict(kind="Bounds and margin eligibility fixed before Java acceptance",
                  arithmetic_epsilon=1e-4, source_sha256=digest(__file__),
                  safetensors_sha256=digest(target), cases=records,
                  files={p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()})
    (args.output_dir/"acceptance.json").write_text(json.dumps(record, indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
