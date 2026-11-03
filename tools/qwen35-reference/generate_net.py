#!/usr/bin/env python3
"""Generate HF CPU FLOAT32 complete DeltaNet convolution/gate/cache goldens."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from probe_hf import tiny_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    config = tiny_config()
    config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(config).eval().bfloat16().float()
    net = model.model.language_model.layers[0].linear_attn
    tensors = {"weights."+key: value.detach().contiguous() for key, value in net.state_dict().items()}
    cases = []
    with torch.no_grad():
        for length in (1, 3, 4, 5, 63, 64, 65, 127, 128, 129):
            stem = f"t{length}."
            cache = DynamicCache(config=config.text_config)
            x = torch.randn(1, length, 128)
            output = net(x, cache_params=cache)
            tensors[stem+"input"] = x
            tensors[stem+"output"] = output.clone()
            tensors[stem+"state"] = cache.get_recurrent_state(0).clone()
            tensors[stem+"convolution"] = cache.get_conv_state(0)[:, :, -3:].transpose(1, 2).contiguous()
            token = torch.randn(1, 1, 128)
            decode = net(token, cache_params=cache)
            tensors[stem+"decode_input"] = token
            tensors[stem+"decode_output"] = decode.clone()
            tensors[stem+"decode_state"] = cache.get_recurrent_state(0).clone()
            tensors[stem+"decode_convolution"] = cache.get_conv_state(0)[:, :, -3:].transpose(1, 2).contiguous()
            fresh = net(torch.cat((x, token), dim=1))[:, -1:]
            error = (fresh-decode).abs().max().item()
            assert error < 1e-4, error
            cases.append(dict(length=length, hf_cached_vs_fresh_max_abs=error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"reference": "HF 5.22.1 CPU FLOAT32, BF16-rounded weights"})
    record = dict(seed=20261102, cases=cases, convolution_history="last K-1 projected real positions, initial zeros not charged",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  hf_source_sha256=hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest(),
                  safetensors_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
