#!/usr/bin/env python3
"""Record HF defaults/legacy metadata and the explicit partial rotary path."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
from probe_hf import tiny_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261102)
    torch.set_num_threads(1)
    baseline = tiny_config().to_dict()
    cases = []
    for name in ("nested_only", "matching_duplicate", "conflicting_duplicate",
                 "missing_nested", "missing_both", "missing_interleaved", "nondefault_type"):
        config = copy.deepcopy(baseline)
        text = config["text_config"]
        text.pop("partial_rotary_factor", None)
        if name == "matching_duplicate":
            text["partial_rotary_factor"] = .25
        elif name == "conflicting_duplicate":
            text["partial_rotary_factor"] = .5
        elif name == "missing_nested":
            text["rope_parameters"].pop("partial_rotary_factor")
            text["partial_rotary_factor"] = .5
        elif name == "missing_both":
            text["rope_parameters"].pop("partial_rotary_factor")
        elif name == "missing_interleaved":
            text["rope_parameters"].pop("mrope_interleaved")
        elif name == "nondefault_type":
            text["rope_parameters"]["rope_type"] = "linear"
            text["rope_parameters"]["factor"] = 2.
        try:
            parsed = Qwen3_5Config.from_dict(config)
            rotary = impl.Qwen3_5TextRotaryEmbedding(parsed.text_config)
            cases.append(dict(name=name, hf_accepted=True,
                              rope_parameters=parsed.text_config.rope_parameters,
                              inverse_frequency_count=rotary.inv_freq.numel()))
        except Exception as error:
            cases.append(dict(name=name, hf_accepted=False, error=str(error)))
    config = tiny_config()
    config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(config).eval().float()
    ids = torch.tensor([[1, 3, 4, 5]])
    with torch.no_grad():
        absent = model(input_ids=ids, use_cache=False).logits
        present_config = copy.deepcopy(config)
        present_config.text_config.output_gate_type = "swish"
        present = Qwen3_5ForConditionalGeneration(present_config).eval().float()
        present.load_state_dict(model.state_dict())
        gate_error = (absent-present(input_ids=ids, use_cache=False).logits).abs().max().item()
        assert gate_error == 0
        positions = torch.tensor([[[0, 2, 7]], [[1, 4, 3]], [[5, 0, 9]]])
        q = torch.randn(1, 2, 3, 64)
        k = torch.randn(1, 1, 3, 64)
        rotary = impl.Qwen3_5TextRotaryEmbedding(config.text_config)
        cos, sin = rotary(q, positions)
        q_rot, k_rot = impl.apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.equal(q[..., 16:], q_rot[..., 16:])
        assert torch.equal(k[..., 16:], k_rot[..., 16:])
    record = dict(kind="WP0 config/rotary probes, not Java acceptance",
                  configuration_cases=cases, output_gate_swish_vs_absent_max_abs=gate_error,
                  attention_gate="sigmoid (inspected source); output_gate_type not consulted",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_source_sha256=hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
                  partial_rotary=dict(positions=positions.tolist(), q=q.tolist(), k=k.tolist(),
                                      cos=cos.tolist(), sin=sin.tolist(),
                                      q_rotated=q_rot.tolist(), k_rotated=k_rot.tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
