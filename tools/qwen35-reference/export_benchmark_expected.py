#!/usr/bin/env python3
"""Independent HF outputs/state for the exact saved four-path benchmark inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from safetensors.torch import save_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    for case in json.loads(args.input_manifest.read_text())["cases"]:
        path = args.inputs_dir / case["inputs_file"]
        assert digest(path) == case["inputs_sha256"]
        arrays = load_file(path)
        q, k, v, g, beta = [torch.from_numpy(arrays[name].copy()) for name in
                            ("q", "k", "v", "log_decay", "beta")]
        initial = torch.from_numpy(arrays["initial"].copy()).transpose(-1, -2).contiguous()
        ratio = v.shape[2] // q.shape[2]
        # Adapt the documented query convention; HF applies Dk**-0.5 internally.
        q = torch.repeat_interleave(q * q.shape[-1]**.5, ratio, dim=2)
        k = torch.repeat_interleave(k, ratio, dim=2)
        kwargs = dict(initial_state=initial, output_final_state=True,
                      use_qk_l2norm_in_kernel=False)
        with torch.no_grad():
            output, state = hf.torch_chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
            recurrent, recurrent_state = hf.torch_recurrent_gated_delta_rule(q, k, v, g, beta, **kwargs)
        output_error = (output-recurrent).abs().max().item()
        state_error = (state-recurrent_state).abs().max().item()
        assert max(output_error, state_error) < 1e-4
        assert torch.isfinite(output).all() and torch.isfinite(state).all()
        target = args.inputs_dir / (path.stem + "-expected.safetensors")
        save_file(dict(output=output.contiguous(), state=state.contiguous()), target,
                  metadata={"reference": "HF 5.22.1 CPU FLOAT32, no Q/K renormalization"})
        records.append(dict(inputs_file=path.name, inputs_sha256=digest(path),
                            expected_file=target.name, expected_sha256=digest(target),
                            hf_chunk_vs_recurrence_output_max_abs=output_error,
                            hf_chunk_vs_recurrence_state_max_abs=state_error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        kind="independent HF benchmark qualification; external tensor files",
        torch=torch.__version__, hf_source_sha256=digest(Path(hf.__file__)),
        source_sha256=digest(Path(__file__)), input_manifest_sha256=digest(args.input_manifest),
        adapters=dict(query="prepared_q * sqrt(Dk); HF rescales internally",
                      state="B,Hv,Dv,Dk transposed to B,Hv,Dk,Dv",
                      decay="saved log_decay; no Q/K renormalization"),
        cases=records), indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
