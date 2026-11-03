#!/usr/bin/env python3
"""Pinned CPU FLOAT32 real-artifact prefill/cached/fresh reference, opt-in."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tensors = {}
    cases = []
    with torch.no_grad():
        for case in json.loads(args.inputs.read_text())["cases"]:
            ids = torch.tensor(case["input_ids"])
            original_count = ids.shape[1]
            kwargs = {}
            if case.get("pixel_values_file"):
                path = args.inputs.parent / case["pixel_values_file"]
                assert digest(path) == case["pixel_values_sha256"]
                kwargs = dict(pixel_values=torch.from_numpy(np.load(path, allow_pickle=False)),
                              image_grid_thw=torch.tensor(case["image_grid_thw"]),
                              mm_token_type_ids=(ids == model.config.image_token_id).long())
            output = model(input_ids=ids, use_cache=True, **kwargs)
            cache = output.past_key_values
            logits = output.logits[:, -1]
            tensors[case["name"] + "_prefill"] = logits.numpy()
            record = dict(name=case["name"], prefill_greedy=logits.argmax(-1).item(), decode=[])
            greedy = []
            for step in range(args.steps):
                token = logits.argmax(-1).item()
                # These selected cases must not introduce another visual placeholder.
                assert token not in (model.config.image_token_id, model.config.video_token_id)
                greedy.append(token)
                ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
                output = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = output.logits[:, -1]
                if kwargs:
                    kwargs["mm_token_type_ids"] = torch.cat((
                        (ids[:, :original_count] == model.config.image_token_id).long(),
                        torch.zeros((1, step+1), dtype=torch.long)), dim=1)
                fresh = model(input_ids=ids, use_cache=False, **kwargs).logits[:, -1]
                error = (logits-fresh).abs().max().item()
                tensors[f'{case["name"]}_decode{step}'] = logits.numpy()
                top = logits.sort().values
                record["decode"].append(dict(input_id=token, greedy=logits.argmax(-1).item(),
                                              margin=(top[0, -1]-top[0, -2]).item(),
                                              cached_vs_fresh_max_abs=error,
                                              cached_vs_fresh_1e4_pass=error < 1e-4))
            record.update(greedy_ids=greedy, decoded=tokenizer.decode(greedy, skip_special_tokens=False),
                          eos_token_id=tokenizer.eos_token_id)
            cases.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = args.output.with_suffix(".npz")
    np.savez_compressed(capture, **tensors)
    record = dict(kind="HF CPU FLOAT32 reference; Java acceptance pending", cases=cases,
                  platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                  torch_threads=1, inputs_sha256=digest(args.inputs),
                  source_sha256=digest(__file__), hf_source_sha256=digest(hf.__file__),
                  logits_file=capture.name, logits_sha256=digest(capture))
    record["cached_vs_fresh_1e4_pass"] = all(
        step["cached_vs_fresh_1e4_pass"] for case in cases for step in case["decode"])
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
