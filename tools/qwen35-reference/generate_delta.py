#!/usr/bin/env python3
"""Generate independent HF FLOAT32 delta-rule component fixtures, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    tensors = {}
    records = []
    for length in (1, 3, 4, 5, 63, 64, 65, 127, 128, 129):
        for hv in (2, 6):
            b, hk, dk, dv = 1, 2, 32, 64
            q = torch.randn(b, length, hk, dk)
            k = torch.randn_like(q)
            q = q / torch.sqrt((q*q).sum(-1, keepdim=True)+1e-6)
            k = k / torch.sqrt((k*k).sum(-1, keepdim=True)+1e-6)
            v = torch.randn(b, length, hv, dv)
            g = -torch.rand(b, length, hv) * .2
            beta = torch.sigmoid(torch.randn(b, length, hv))
            initial = torch.randn(b, hv, dk, dv) * .03
            qr = torch.repeat_interleave(q, hv//hk, dim=2)
            kr = torch.repeat_interleave(k, hv//hk, dim=2)
            kwargs = dict(initial_state=initial, output_final_state=True)
            with torch.no_grad():
                y, state = hf.torch_chunk_gated_delta_rule(qr, kr, v, g, beta, **kwargs)
                yr, sr = hf.torch_recurrent_gated_delta_rule(qr, kr, v, g, beta, **kwargs)
            error = (y-yr).abs().max().item()
            state_error = (state-sr).abs().max().item()
            assert max(error, state_error) < 1e-4
            stem = f"t{length}h{hv}"
            for name, tensor in dict(q=q/dk**.5, k=k, v=v, g=g, beta=beta,
                                     initial=initial, output=y, state=state).items():
                tensors[f"{stem}.{name}"] = tensor.contiguous()
            records.append(dict(name=stem, length=length, value_heads=hv,
                                hf_chunk_vs_recurrence_max_abs=error,
                                hf_chunk_vs_recurrence_state_max_abs=state_error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"reference": "HF 5.22.1 CPU FLOAT32"})
    record = dict(seed=20261102, cases=records,
                  q_convention="L2 normalized with sum-space epsilon 1e-6, scaled by 1/sqrt(Dk)",
                  state_orientation="B,Hv,Dk,Dv", gate_convention="g is log-space decay",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  hf_source_sha256=hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest(),
                  safetensors_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
