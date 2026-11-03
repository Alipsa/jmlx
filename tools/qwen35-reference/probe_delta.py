#!/usr/bin/env python3
"""Probe stock delta dispatch and native boundaries; never writes model goldens."""
import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.qwen3_5 import gated_delta as vlm


def max_error(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    mx.random.seed(20261102)
    records = []
    for device in (mx.gpu, mx.cpu):
        mx.set_default_device(device)
        for dk, dv, hk, hv, length, dtype, masked in [
                (dk, dv, 2, 6, 3, mx.float32, False)
                for dk in (1, 16, 31, 32, 33, 64, 128)
                for dv in (32, 64, 128)] + [
                (32, 64, 2, 6, length, dtype, masked)
                for length in (1, 64, 65)
                for dtype in (mx.float32, mx.float16, mx.bfloat16)
                for masked in (False, True)] + [(32, 64, 2, 3, 3, mx.float32, False)]:
            record = dict(device=str(device), dk=dk, dv=dv, hk=hk, hv=hv,
                          length=length, dtype=str(dtype), masked=masked)
            q = (mx.random.normal((1, length, hk, dk)) * .1).astype(dtype)
            k = (mx.random.normal(q.shape) * .1).astype(dtype)
            v = mx.random.normal((1, length, hv, dv)).astype(dtype)
            decay = mx.full((1, length, hv), .8, dtype)
            beta = mx.full((1, length, hv), .4, dtype)
            state = mx.random.normal((1, hv, dv, dk)) * .01
            mask = mx.array([[i % 3 != 1 for i in range(length)]]) if masked else None
            try:
                result, final = mx.fast.gated_delta_update(q, k, v, decay, beta, state, mask)
                mx.eval(result, final)
                record.update(accepted=True, output_shape=list(result.shape),
                              state_shape=list(final.shape), state_dtype=str(final.dtype))
            except (ValueError, RuntimeError) as error:
                record.update(accepted=False, error=str(error))
                records.append(record)
                continue
            ref, ref_state = vlm.gated_delta_ops(q, k, v, decay, beta, state, mask)
            mx.eval(ref, ref_state)
            record.update(output_max_abs=max_error(result, ref),
                          state_max_abs=max_error(final, ref_state))
            active_error = max_error(mx.where(mask[..., None, None], result, 0),
                                     mx.where(mask[..., None, None], ref, 0)) if masked else record["output_max_abs"]
            record["active_output_max_abs"] = active_error
            assert active_error < .02, record
            assert record["state_max_abs"] < .02, record
            # The custom Metal path has a source-derived multiple-of-32 limit.
            if device == mx.gpu and dk % 32 == 0:
                custom, custom_state = vlm.gated_delta_kernel(q, k, v, decay, beta, state, mask)
                mx.eval(custom, custom_state)
                record.update(custom_output_max_abs=max_error(custom, ref),
                              custom_state_max_abs=max_error(custom_state, ref_state))
                custom_error = max_error(mx.where(mask[..., None, None], custom, 0),
                                         mx.where(mask[..., None, None], ref, 0)) if masked else record["custom_output_max_abs"]
                record["custom_active_output_max_abs"] = custom_error
                assert custom_error < .02, record
                assert record["custom_state_max_abs"] < .02, record
            records.append(record)
    mx.set_default_device(mx.gpu)
    paths = []
    # Trace dispatcher choices without changing any callable's arithmetic.
    originals = {name: getattr(vlm, name) for name in
                 ("gated_delta_ops", "gated_delta_chunked", "gated_delta_kernel")}
    selected = []
    def traced(name, fn):
        def call(*args, **kwargs):
            selected.append(name)
            return fn(*args, **kwargs)
        return call
    try:
        for name, fn in originals.items():
            setattr(vlm, name, traced(name, fn))
        for device in (mx.gpu, mx.cpu):
            mx.set_default_device(device)
            for use_kernel in (True, False):
                for length in (1, 3):
                    selected.clear()
                    q = mx.full((1, length, 2, 32), .1)
                    v = mx.full((1, length, 6, 64), .2)
                    a = b = mx.zeros((1, length, 6))
                    out, state = vlm.gated_delta_update(
                        q, q, v, a, b, mx.zeros((6,)), mx.zeros((6,)),
                        use_kernel=use_kernel)
                    mx.eval(out, state)
                    paths.append(dict(device=str(device), use_kernel=use_kernel,
                                      length=length, selected=list(selected)))
    finally:
        for name, fn in originals.items():
            setattr(vlm, name, fn)
    record = dict(kind="WP0 probes, not acceptance goldens", seed=20261102,
                  platform=platform.platform(), mlx=mx.__version__,
                  metal_available=mx.metal.is_available(),
                  gate_convention="linear-space decay for native and mlx-vlm",
                  state_orientation="B,Hv,Dv,Dk (transpose last axes for Java/HF)",
                  vlm_source_sha256=hashlib.sha256(Path(vlm.__file__).read_bytes()).hexdigest(),
                  probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_doc=inspect.getdoc(mx.fast.gated_delta_update),
                  boundaries=records, dispatch=paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
