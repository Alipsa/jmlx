# Phase 6 compatibility matrix

This is the human-readable source of truth for local text inference support. “Implemented” means
the capability is present in a released API only after the Phase 6 gate says so; a synthetic test is
not evidence that an arbitrary Hugging Face artifact will load.

| Architecture | Status | Verification | Tokenizer / chat template | Checkpoint | Quantization | RoPE / context/cache policy | License / access notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama pre-norm decoder | implemented | verified-with-real-artifact on recorded Tier-B pin; scaled RoPE verified with synthetic fixtures | ByteLevel BPE; directory metadata/templates and tokenizer-backed text streaming | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | base, linear, dynamic NTK, Llama 3, YaRN RoPE; FULL cache, optionally capacity-bounded | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Qwen2 pre-norm GQA decoder | implemented | verified-with-real-artifact on recorded Tier-B pin; scaled RoPE verified with synthetic fixtures | ByteLevel BPE; directory metadata/templates and tokenizer-backed text streaming | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | base, linear, dynamic NTK, Llama 3, YaRN RoPE; FULL cache, optionally capacity-bounded; `use_sliding_window=true` rejected | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Qwen3 pre-norm GQA decoder (per-head QK norm) | implemented | verified-with-real-artifact on recorded Tier-B pin; strict float32 HF reference on the synthetic fixture | ByteLevel BPE; real Qwen3 chat template (live-verified SHA-256) embedded in the synthetic family bundle | float safetensors (0.6B tied embeddings, 8B untied) | float + MLX affine 4-bit (synthetic verification; local 8B 4-bit run) | explicit head dimension, base RoPE; FULL cache, optionally capacity-bounded; `use_sliding_window=true` rejected | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Mistral | implemented | verified-with-real-artifact on recorded Tier-B pin; synthetic Hugging Face tiny-checkpoint logits and window references | Real-artifact chat text/IDs and synthetic family tokenizer golden | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | FULL cache by default; explicit sliding eviction only when checkpoint `sliding_window` is non-null | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Gemma v1 | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer/template; text/ID golden | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | explicit head dimension and base RoPE; FULL cache, optionally capacity-bounded | Tier-B terms acceptance pending |
| Phi-3 | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer/template; text/ID golden | float safetensors with fused QKV and gate/up projections | float + MLX affine 4-bit (synthetic verification) | FULL cache by default; sliding eviction requires a checkpoint window; `longrope` rejected | Public MIT artifact; hosted-runner RAM limits Tier-B execution |
| Mixtral / MoE | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer with Mistral template; text/ID golden | float safetensors with expert tensors | float weights only | FULL cache by default; sliding eviction requires a checkpoint window; gathered top-k expert compute | Tier-B artifact/runner pending |
| Gemma 2/3, Phi-2, LongRoPE, shared-expert MoE | planned | no verification | — | — | — | unsupported capability named during config parsing | — |

The supported runtime is macOS on Apple Silicon with Java 25 and the MLX pins in
`scripts/bootstrap-native.sh`. Artifact licensing and access requirements are recorded alongside
each future Tier-B fixture; this matrix intentionally makes no claim that every public Hugging Face
checkpoint is compatible.

The independent pure-Java tokenizer module additionally verifies synthetic/reference fixtures for
Metaspace BPE, Metaspace Unigram, and Bert/WordPiece. Phase 6.3 family tokenizer bundles are synthetic
and establish component and chat-rendering behavior, not byte-level parity with a production tokenizer.
SentencePiece `Precompiled` normalizers are rejected. The [2026-09-29 macOS native CI run](https://github.com/Alipsa/jmlx/actions/runs/36627571854)
executed the Phase 6.3 attention, MoE, Llama/Qwen, Mistral, Gemma, Phi-3 and Mixtral suites;
the workflow checks their JUnit XML to prevent a silent native skip. The
[Llama](https://github.com/Alipsa/jmlx/actions/runs/36640308134),
[Qwen](https://github.com/Alipsa/jmlx/actions/runs/36640308134), and
[Mistral](https://github.com/Alipsa/jmlx/actions/runs/36640308134) pinned real-artifact jobs
asserted 16 greedy IDs on the recorded runner pin; the other rows retain
synthetic-fixture status.

MLX affine quantization (`quantization: {group_size, bits}`, tensors as `.weight`/`.scales`/`.biases`)
loads for Llama, Qwen2, Qwen3, Mistral, Gemma v1 and Phi-3, with the packed embedding table also
serving as the tied output head. Verification is **synthetic plus one local real artifact**:
`QuantizedDecoderTest`
compares each family's quantized logits against the same weights dequantized to float, and
`mlx-community/Llama-3.2-1B-Instruct-4bit` (not a pinned Tier-B artifact) produced the same greedy
token IDs as an independent MLX-Python reference on three prompts (24, 8 and 16 tokens). The
`mlx-community/Qwen3-8B-4bit` local run (throughput and memory, not Tier-B evidence) is recorded in
`req/phase6-4-benchmark.md`. Mixtral
quantization, per-layer overrides, non-affine modes, GPTQ/AWQ and GGUF are rejected with the key
named. Quantized KV cache remains unsupported.

## Phase 7.2 text tasks (2026-10-05)

| Task/family | Tier-A | Tier-B | Limits |
| --- | --- | --- | --- |
| Text encoder, BERT | Tiny HF encoder hidden states including padded rows; CLS/mean/max/L2 | all-MiniLM-L6-v2, actual sentence-transformers CPU vector | Absolute positions, GELU, unquantized; one supported Pooling plus optional Normalize |
| Sequence classification, BERT | Tiny HF sequence logits and labels | Licensed bert-base-uncased-sst2 fallback; stable-margin exact class | Regression rejected; pair API separately oracle-verified |
| Token classification, BERT | Tiny HF token logits for every row | No real task-head claim | Softmax/sigmoid selected by problem_type; no CRF |
| Seq2seq text, T5/Flan-T5 | Tied ReLU/untied gated-GELU; encoder, full/cached logits, buckets, generation | flan-t5-small complete greedy IDs, EOS and every CPU logit history | FULL cache only; no scheduler, beam search, decoder prefixes or quantized weights |

All three new real-artifact checks passed locally on M5 Max/macOS 27.0.1 with TF32 disabled.
Their first supported macOS 26 CI executions remain pending. See `phase7-2-implementation-report.md`
for pins, margins and measured resource use; synthetic head checks do not imply other model families.
