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
