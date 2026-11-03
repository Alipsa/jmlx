#!/usr/bin/env python3
"""HF stress fixtures for correlated keys, decay extremes and preprocessing-batch boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    tensors = {}
    records = []
    for name, batch, length, dk, mode in (
            ("correlated", 1, 129, 32, "identical"),
            ("near_correlated", 1, 129, 128, "near"),
            ("strong_decay", 1, 129, 32, "decay"),
            ("batch_boundary", 2, 1025, 32, "random"),
            ("long_boundary", 1, 2049, 32, "random")):
        hk, hv, dv = 2, 6, 64
        q = torch.randn(batch, length, hk, dk)
        k = torch.randn_like(q)
        beta = torch.full((batch, length, hv), .6)
        g = torch.full_like(beta, -.02)
        if mode == "identical":
            q.zero_(); k.zero_(); q[..., 0] = 1; k[..., 0] = 1
            beta.fill_(1); g.zero_()
        elif mode == "near":
            base = torch.randn(batch, 1, hk, dk)
            k = base + k * 1e-3
            q = base + q * 1e-3
            beta.fill_(.9999); g.fill_(-1e-6)
        elif mode == "decay":
            g.fill_(-80)
        q = q / torch.sqrt((q*q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k*k).sum(-1, keepdim=True) + 1e-6)
        v = torch.randn(batch, length, hv, dv)
        initial = torch.randn(batch, hv, dk, dv) * .03
        qr = torch.repeat_interleave(q, hv//hk, dim=2)
        kr = torch.repeat_interleave(k, hv//hk, dim=2)
        kwargs = dict(initial_state=initial, output_final_state=True)
        with torch.no_grad():
            output, state = hf.torch_chunk_gated_delta_rule(qr, kr, v, g, beta, **kwargs)
            recurrence, recurrent_state = hf.torch_recurrent_gated_delta_rule(qr, kr, v, g, beta, **kwargs)
        error = max((output-recurrence).abs().max().item(), (state-recurrent_state).abs().max().item())
        assert error < 1e-4, (name, error)
        for key, value in dict(q=q/dk**.5, k=k, v=v, g=g, beta=beta, initial=initial,
                               output=output, state=state).items():
            tensors[name+"."+key] = value.contiguous()
        records.append(dict(name=name, batch=batch, length=length, dk=dk, hf_chunk_vs_recurrence_max_abs=error))
    # A valid fully correlated, zero-decay, beta=1 system is a lower-triangular all-ones matrix.
    # Measure the proposed doubling inverse separately; it never supplies a golden.
    system = torch.ones(64, 64).tril()
    eye = torch.eye(64)
    power = -(system-eye)
    doubled = eye+power
    for _ in range(5):
        power = power @ power
        doubled = doubled @ (eye+power)
    reference_inverse = torch.linalg.solve_triangular(system, eye, upper=False, unitriangular=True)
    doubling_error = (doubled-reference_inverse).abs().max().item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"reference": "HF 5.22.1 CPU FLOAT32 stress"})
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        seed=20261103, source_sha256=digest(__file__), hf_source_sha256=digest(hf.__file__),
        safetensors_sha256=digest(args.output), cases=records,
        doubling_probe=dict(kind="rejected numerical experiment, never a golden",
                            inverse_max_abs_error=doubling_error,
                            eligible_1e4=doubling_error < 1e-4)), indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
