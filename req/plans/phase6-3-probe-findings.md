# Phase 6.3 quantization and GGUF probe findings

Status: **float-only decision recorded after native projection probe** (2026-09-29).

`QuantizedCheckpointProbeTest` creates the seeded, 64-wide Llama checkpoint from Task 5,
quantizes one `q_proj.weight` with affine group 64 / 4-bit MLX quantization, saves and reloads
`*.weight`, `*.scales`, and `*.biases` in safetensors, and compares the loaded
`QuantizedLinear` output with the float projection. The test passed (1/1 executed) in the
[macOS native CI run](https://github.com/Alipsa/jmlx/actions/runs/36627571854), using the
runtime pinned by `scripts/bootstrap-native.sh`. Its 0.08 absolute tolerance is a
projection-level probe threshold, not a measured model-level tolerance. The remaining
whole-decoder logit comparison and quantized embedding/head behavior are unmeasured.

## Decision gate

The successful projection probe has **not** established that MLX-community quantized safetensors load into a complete
decoder without new native surface. Keep `quantization` and `quantization_config` as named
config errors, and keep the compatibility matrix at **float weights only**. Do not add Task
10a′ or claim quantized checkpoint support until a complete decoder/logit probe passes on
the pinned macOS/MLX runtime. The current run establishes that the existing native surface
can round-trip one packed projection; it does not establish quantized embedding/head support.

No GGUF quantization is chosen. `MLXIO.loadGguf` has not been tested here with Q4_0, Q8_0,
Q4_K, or another model artifact, and the decoder does not accept GGUF files. A GGUF support
claim would also require tensor-name mapping, q/k rotary-layout un-permutation,
tokenizer/vocabulary metadata handling, and per-type dequantization parity tests. Until
those exist, the checkpoint loader's safetensors requirement is the exact format rejection.

## Gathered MoE (`mlx_gather_mm`) probe

Status: **recorded 2026-09-30 on mlx 0.31.2** (the pinned `mlx-metal` wheel). Re-run with
`./tools/mlx-oracle/.venv/bin/python tools/mlx-oracle/probes/moe_gather_probe.py` after any
native pin change; it exits non-zero on a correctness regression.

- Correctness (PASS): per-token equivalence; sorted == unsorted forward and backward for
  top-2 and top-k == E; float indices rejected; gradient through router-derived indices
  raises `[GatherMM] Cannot calculate VJP with respect to indices.` unless the indices go
  through `stop_gradient`, after which grads equal the dense implementation exactly.
- **Not testable from Java (recorded, not asserted):** `sorted_indices=True` on unsorted
  indices gives the SAME result (the flag is only a hint); out-of-range indices raise NO error.
  If either flips on a future runtime, revisit `SwitchGlu`'s review items and bounds checks.
- Half precision vs exact closed form, `|a - r| / (|r| + 1)`: bf16 worst 0.0076, f16 worst
  0.0011. Gathered and dense differ at top-3 (bf16 0.031 absolute) because of reduction order.
- Load memory: per-layer stack + eval + close sources = 1.125x weights (peak and steady);
  keeping sources = 2.0x.
- Approximate timing (bf16, H=1024 F=3584 E=8 K=2; varies by machine and run):
  - Python probe, expert path only: T=1 dense ≈1.67 ms vs gathered ≈0.32 ms; T=128 dense
    ≈5.51 ms, gathered ≈5.47 ms, gathered-sorted ≈2.85 ms.
  - Java `MoeMlpBenchmarkTest`, whole `MoeMlp.forward` including routing: T=1 ≈2.1 ms dense
    vs ≈1.0 ms gathered, 1.88-2.29× across eight runs. T=128 (256 slots, sorted path)
    1.64-2.38×, sometimes below decode. The probe's "unsorted ≈ dense at T=128" does not
    describe Java prefill, which always sorts at this size. The ≈0.65 ms/call gathered-side
    overhead against the probe is not yet explained; the decode floor is 1.5×, and prefill
    is reported only.
  - Re-run while implementing this plan: probe T=1 dense 1.80 ms vs gathered 0.32 ms, T=128
    5.55 / 5.44 / 2.83 ms; Java benchmark T=1 2.34 vs 1.08 ms (2.16×), T=128 5.25 vs 3.44 ms
    (1.53×).
