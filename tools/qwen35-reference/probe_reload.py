#!/usr/bin/env python3
"""Reload stock converter output; optionally run the labeled normalization diagnostic.

Diagnostic output is attribution evidence only, never an acceptance golden.
"""
import argparse
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import textwrap
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.qwen3_5 import language
from mlx_vlm.models.qwen3_5 import gated_delta
from mlx_vlm.utils import load_model
from mlx_lm.models import qwen3_5 as lm_language, gated_delta as lm_delta, qwen3_next as lm_next


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(fn):
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("vlm", "lm"), default="vlm")
    parser.add_argument("--normalization-diagnostic", action="store_true")
    parser.add_argument("--float32", action="store_true", help="Upcast floating parameters only")
    parser.add_argument("--logits-only", action="store_true", help="Keep path observations but omit large Q/K/output/state capture arrays")
    parser.add_argument("--capture-dir", type=Path, help="External storage for large real-model captures")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--greedy-history", action="store_true")
    parser.add_argument("--teacher-history-from", type=Path)
    args = parser.parse_args()
    if args.decode_steps < 1:
        parser.error("--decode-steps must be positive")
    teacher_cases = ({case["name"]: case for case in
                      json.loads(args.teacher_history_from.read_text())["cases"]}
                     if args.teacher_history_from else {})
    if args.family != "vlm" and args.normalization_diagnostic:
        parser.error("The diagnostic applies only to mlx-vlm's Q/K convention")
    active_language = language if args.family == "vlm" else lm_language
    active_delta = gated_delta if args.family == "vlm" else lm_delta
    gate_owner = language if args.family == "vlm" else lm_next
    gate_name = "_precise_swiglu" if args.family == "vlm" else "precise_swiglu"
    record = dict(
        kind="normalization-only diagnostic; NEVER acceptance/goldens" if args.normalization_diagnostic
             else "stock reference probe; not yet a declared acceptance golden",
        platform=platform.platform(), python=platform.python_version(),
        device=str(mx.default_device()), stream=str(mx.default_stream(mx.gpu)),
        metal_available=mx.metal.is_available(),
        tf32=os.environ.get("MLX_ENABLE_TF32", "unset"),
        inputs_sha256=digest(args.inputs),
        teacher_history_sha256=digest(args.teacher_history_from) if args.teacher_history_from else None,
        packages={p: importlib.metadata.version(p) for p in ("mlx", "mlx-metal", "mlx-vlm", "mlx-lm")},
        sources={active_language.__name__: digest(active_language.__file__),
                 active_delta.__name__: digest(active_delta.__file__),
                 gate_owner.__name__: digest(gate_owner.__file__),
                 "probe_reload.py": digest(__file__)}, cases=[])
    original_call = language.Qwen3_5GatedDeltaNet.__call__
    if args.normalization_diagnostic:
        stock = textwrap.dedent(inspect.getsource(original_call))
        aligned = stock
        for name in ("q", "k"):
            pattern = f"mx.fast.rms_norm({name}, None, 1e-6)"
            assert aligned.count(pattern) == 1
            aligned = aligned.replace(pattern, f"mx.fast.rms_norm({name}, None, 1e-6 / self.head_k_dim)")
        namespace = dict(language.__dict__)
        exec(compile(aligned, "<normalization-only-diagnostic>", "exec"), namespace)
        language.Qwen3_5GatedDeltaNet.__call__ = namespace["__call__"]
        record["diagnostic"] = dict(
            stock_sha256=hashlib.sha256(stock.encode()).hexdigest(),
            aligned_sha256=hashlib.sha256(aligned.encode()).hexdigest(),
            diff="".join(difflib.unified_diff(stock.splitlines(True), aligned.splitlines(True),
                                           fromfile="stock", tofile="normalization-only")))
    arrays = {}
    events = []
    phase = ""
    original_delta = active_language.gated_delta_update
    original_conv = nn.Conv1d.__call__
    original_gate = getattr(gate_owner, gate_name)
    original_decode_conv = language._qwen3_5_decode_depthwise_conv
    dispatch_calls = []
    dispatch_originals = {}
    def observe_dispatch(name, fn):
        def call(*positional, **kwargs):
            dispatch_calls.append(dict(callable=fn.__module__+"."+name,
                                       source_sha256=source_hash(fn)))
            return fn(*positional, **kwargs)
        return call
    for name in ("gated_delta_kernel", "gated_delta_ops", "gated_delta_chunked",
                 "gated_delta_kernel_xtree", "gated_delta_kernel_unpacked"):
        if hasattr(active_delta, name):
            fn = getattr(active_delta, name)
            dispatch_originals[name] = fn
            setattr(active_delta, name, observe_dispatch(name, fn))
    def capture_delta(q, k, v, *positional, **kwargs):
        index = sum(event["kind"] == "delta" for event in events)
        dispatch_start = len(dispatch_calls)
        output, state = original_delta(q, k, v, *positional, **kwargs)
        mx.eval(q, k, output, state)
        stem = f"{phase}_delta{index}"
        if not args.logits_only:
            for name, value in (("q", q), ("k", k), ("output", output), ("state", state)):
                arrays[f"{stem}_{name}"] = np.asarray(value.astype(mx.float32))
        events.append(dict(kind="delta", phase=phase, q_dtype=str(q.dtype),
                           v_dtype=str(v.dtype), state_dtype=str(state.dtype),
                           use_kernel=kwargs.get("use_kernel", True),
                           selected=dispatch_calls[dispatch_start:],
                           callable_sha256=source_hash(original_delta)))
        return output, state
    def capture_conv(self, x):
        output = original_conv(self, x)
        events.append(dict(kind="conv1d", phase=phase, input_dtype=str(x.dtype),
                           weight_dtype=str(self.weight.dtype), output_dtype=str(output.dtype),
                           callable_sha256=source_hash(original_conv)))
        return output
    def capture_gate(*positional, **kwargs):
        output = original_gate(*positional, **kwargs)
        events.append(dict(kind="gated_norm_silu", phase=phase,
                           input_dtypes=[str(x.dtype) for x in positional],
                           output_dtype=str(output.dtype), accumulation="float32 (source)",
                           callable_sha256=source_hash(original_gate)))
        return output
    def capture_decode_conv(x, weight):
        output = original_decode_conv(x, weight)
        events.append(dict(kind="decode_depthwise_conv", phase=phase,
                           input_dtype=str(x.dtype), weight_dtype=str(weight.dtype),
                           output_dtype=str(output.dtype), accumulation="float32 (source)",
                           callable_sha256=source_hash(original_decode_conv)))
        return output
    active_language.gated_delta_update = capture_delta
    nn.Conv1d.__call__ = capture_conv
    setattr(gate_owner, gate_name, capture_gate)
    language._qwen3_5_decode_depthwise_conv = capture_decode_conv
    if args.normalization_diagnostic:
        namespace["gated_delta_update"] = capture_delta
    try:
        if args.family == "vlm":
            model = load_model(args.model)
        else:
            from mlx_lm import load
            model, _ = load(str(args.model))
        model.eval()
        if args.float32:
            model.set_dtype(mx.float32)
        record["floating_parameter_upcast"] = args.float32
        record["training"] = model.training
        record["parameter_dtypes"] = sorted({str(v.dtype) for _, v in
                                             __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())})
        for case in json.loads(args.inputs.read_text())["cases"]:
            if args.family == "lm" and case["name"] == "image":
                continue
            ids = mx.array(case["input_ids"])
            pixels = case.get("pixel_values")
            if case.get("pixel_values_file"):
                pixels_path = args.inputs.parent / case["pixel_values_file"]
                assert digest(pixels_path) == case["pixel_values_sha256"]
                pixels = np.load(pixels_path, allow_pickle=False)
            kwargs = dict(pixel_values=mx.array(pixels),
                          image_grid_thw=mx.array(case.get("image_grid_thw", [[1, 2, 2]]))) if pixels is not None else {}
            cache = model.language_model.make_cache() if args.family == "vlm" else model.make_cache()
            events.clear()
            phase = case["name"] + "_prefill"
            output = model(ids, cache=cache, **kwargs)
            logits = output.logits if args.family == "vlm" else output
            mx.eval(logits)
            last = logits[:, -1].astype(mx.float32)
            arrays[phase + "_logits"] = np.asarray(last)
            result = dict(name=case["name"], input_ids=case["input_ids"],
                          prefill_greedy=mx.argmax(last).item(),
                          prefill_margin=(mx.sort(last)[0, -1] - mx.sort(last)[0, -2]).item())
            if "prefill_logits" in case:
                result["prefill_hf_max_abs"] = mx.max(mx.abs(last - mx.array(case["prefill_logits"]))).item()
            # Match mlx-vlm.generate.ar: do not rerun multimodal get_input_embeddings here.
            decoder = model.language_model if args.family == "vlm" else model
            result["decode_steps"] = []
            for step in range(args.decode_steps):
                phase = case["name"] + f"_decode{step}"
                if case["name"] in teacher_cases:
                    forced_id = teacher_cases[case["name"]]["decode_steps"][step]["input_id"]
                else:
                    forced_id = mx.argmax(last).item() if args.greedy_history else case.get("forced_decode_id", 6)
                output = decoder(mx.array([[forced_id]]), cache=cache)
                logits = output.logits if args.family == "vlm" else output
                mx.eval(logits)
                last = logits[:, -1].astype(mx.float32)
                arrays[phase + "_logits"] = np.asarray(last)
                decode_record = dict(input_id=forced_id, greedy=mx.argmax(last).item(),
                                     margin=(mx.sort(last)[0, -1] - mx.sort(last)[0, -2]).item())
                if "decode_logits" in case and step == 0:
                    decode_record["hf_max_abs"] = mx.max(mx.abs(last - mx.array(case["decode_logits"]))).item()
                result["decode_steps"].append(decode_record)
            result["paths"] = list(events)
            record["cases"].append(result)
    finally:
        language.Qwen3_5GatedDeltaNet.__call__ = original_call
        active_language.gated_delta_update = original_delta
        nn.Conv1d.__call__ = original_conv
        setattr(gate_owner, gate_name, original_gate)
        for name, fn in dispatch_originals.items():
            setattr(active_delta, name, fn)
        language._qwen3_5_decode_depthwise_conv = original_decode_conv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture_path = (args.capture_dir / args.output.with_suffix(".npz").name
                    if args.capture_dir else args.output.with_suffix(".npz"))
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(capture_path, **arrays)
    record["captures"] = dict(file=capture_path.name, sha256=digest(capture_path),
                              external=bool(args.capture_dir), logits_only=args.logits_only)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
