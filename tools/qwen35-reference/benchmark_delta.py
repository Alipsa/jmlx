#!/usr/bin/env python3
"""Four-path evaluated DeltaNet benchmark; no default is selected by this tool."""
import argparse
import datetime
import hashlib
import inspect
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.qwen3_5 import gated_delta as vlm


def host_state():
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                load_average=list(os.getloadavg()),
                vm_stat=subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout,
                processes=subprocess.run(["ps", "-Ao", "pid,pcpu,rss,comm", "-r"],
                                         capture_output=True, text=True, check=True).stdout.splitlines()[:16])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--shapes", nargs="+", choices=("tiny", "08b", "27b"), default=["tiny", "08b", "27b"])
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 256, 1024, 1280])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    mx.random.seed(20261102)
    paths = dict(chunked_ops=vlm.gated_delta_chunked,
                 token_recurrence=vlm.gated_delta_ops,
                 vlm_custom_metal=vlm.gated_delta_kernel,
                 mlx_builtin=mx.fast.gated_delta_update)
    record = dict(kind="WP1 reference benchmark; Java facade dispatch cost not measured",
                  platform=platform.platform(), mlx=mx.__version__,
                  tf32=os.environ.get("MLX_ENABLE_TF32", "unset"),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  vlm_source_sha256=hashlib.sha256(Path(vlm.__file__).read_bytes()).hexdigest(),
                  samples=args.samples, layers_per_sample=args.layers, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for shape in args.shapes:
        hk, hv, dk, dv = {"tiny": (2, 6, 32, 64), "08b": (16, 16, 128, 128),
                          "27b": (16, 48, 128, 128)}[shape]
        for length in args.lengths:
            q = mx.random.normal((1, length, hk, dk))
            k = mx.random.normal(q.shape)
            q = q * mx.rsqrt(mx.sum(q*q, axis=-1, keepdims=True)+1e-6) / dk**.5
            k = k * mx.rsqrt(mx.sum(k*k, axis=-1, keepdims=True)+1e-6)
            v = mx.random.normal((1, length, hv, dv))
            decay = mx.full((1, length, hv), .98)
            beta = mx.full((1, length, hv), .4)
            initial = mx.random.normal((1, hv, dv, dk)) * .01
            mx.eval(q, k, v, decay, beta, initial)
            args.inputs_dir.mkdir(parents=True, exist_ok=True)
            input_path = args.inputs_dir / f"{shape}-{length}.safetensors"
            if not input_path.exists():
                mx.save_safetensors(str(input_path), dict(q=q, k=k, v=v, decay=decay,
                                                         log_decay=mx.log(decay), beta=beta,
                                                         initial=initial))
            saved = mx.load(str(input_path))
            q, k, v, decay, beta, initial = [saved[key] for key in
                                           ("q", "k", "v", "decay", "beta", "initial")]
            assert q.shape == (1, length, hk, dk) and v.shape == (1, length, hv, dv)
            mx.eval(q, k, v, decay, beta, initial)
            inputs = (q, k, v, decay, beta, initial)
            reference, reference_state = vlm.gated_delta_ops(*inputs)
            mx.eval(reference, reference_state)
            case = dict(shape=shape, dimensions=[hk, hv, dk, dv], length=length,
                        inputs_file=input_path.name,
                        inputs_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        logical_input_bytes=sum(x.nbytes for x in inputs), paths={})
            for name, fn in paths.items():
                observation = host_state()
                mx.clear_cache()
                mx.reset_peak_memory()
                active = mx.get_active_memory()
                start = time.perf_counter_ns()
                output, state = fn(*inputs)
                mx.eval(output, state)
                first_ns = time.perf_counter_ns() - start
                output_error = mx.max(mx.abs(output-reference)).item()
                state_error = mx.max(mx.abs(state-reference_state)).item()
                case["paths"][name] = dict(first_evaluated_ns=first_ns,
                                          first_peak_above_active=mx.get_peak_memory()-active,
                                          output_max_abs=output_error, state_max_abs=state_error,
                                          eligible=output_error < 1e-4 and state_error < 1e-4,
                                          first_host_state=observation, warmed=[])
            # Rotate execution order to avoid a fixed thermal/order advantage.
            names = [name for name in paths if case["paths"][name]["eligible"]]
            for sample in range(args.samples):
                for name in names[sample % 4:] + names[:sample % 4]:
                    observation = host_state()
                    mx.reset_peak_memory()
                    active = mx.get_active_memory()
                    start = time.perf_counter_ns()
                    for _ in range(args.layers):
                        output, state = paths[name](*inputs)
                        mx.eval(output, state)
                    elapsed = time.perf_counter_ns() - start
                    case["paths"][name]["warmed"].append(dict(
                        evaluated_ns=elapsed, peak_above_active=mx.get_peak_memory()-active,
                        host_state=observation))
            for name in names:
                case["paths"][name]["median_evaluated_ns"] = statistics.median(
                    x["evaluated_ns"] for x in case["paths"][name]["warmed"])
            record["cases"].append(case)
            args.output.write_text(json.dumps(record, indent=2)+"\n")
            print(shape, length, {name: round(case["paths"][name]["median_evaluated_ns"]/1e6, 3)
                                  for name in names}, flush=True)


if __name__ == "__main__":
    main()
