# Phase 6.3 quantization and GGUF probe findings

Status: **decision pending native execution** (2026-09-29).

`QuantizedCheckpointProbeTest` creates the seeded, 64-wide Llama checkpoint from Task 5,
quantizes one `q_proj.weight` with affine group 64 / 4-bit MLX quantization, saves and reloads
`*.weight`, `*.scales`, and `*.biases` in safetensors, and compares the loaded
`QuantizedLinear` output with the float projection. The test is native-gated. It compiled on
Linux, but did not run here because the MLX native library is unavailable. Its 0.08 absolute
tolerance is a probe threshold, not a measured model-level tolerance. The remaining
whole-decoder logit comparison and quantized embedding/head behavior are unmeasured.

## Decision gate

The probe has **not** established that MLX-community quantized safetensors load into a complete
decoder without new native surface. Keep `quantization` and `quantization_config` as named
config errors, and keep the compatibility matrix at **float weights only**. Do not add Task
10a′ or claim quantized checkpoint support until the native projection probe and a complete
decoder/logit probe pass on the pinned macOS/MLX runtime. Record that run's native pin,
result, and tolerance here.

No GGUF quantization is chosen. `MLXIO.loadGguf` has not been tested here with Q4_0, Q8_0,
Q4_K, or another model artifact, and the decoder does not accept GGUF files. A GGUF support
claim would also require tensor-name mapping, q/k rotary-layout un-permutation,
tokenizer/vocabulary metadata handling, and per-type dequantization parity tests. Until
those exist, the checkpoint loader's safetensors requirement is the exact format rejection.
