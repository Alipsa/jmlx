# Phase 6 compatibility matrix

This is the human-readable source of truth for local text inference support. “Implemented” means
the capability is present in a released API only after the Phase 6 gate says so; a synthetic test is
not evidence that an arbitrary Hugging Face artifact will load.

| Architecture | Status | Verification | Tokenizer / chat template | Checkpoint | Quantization | RoPE / context/cache policy | License / access notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama pre-norm decoder | implemented | verified-with-real-artifact on recorded Tier-B pin; scaled RoPE verified with synthetic fixtures | ByteLevel BPE; directory metadata/templates and tokenizer-backed text streaming | float safetensors, including index shards | float weights only | base, linear, dynamic NTK, Llama 3, YaRN RoPE; unbounded per-generation cache | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Qwen2 pre-norm GQA decoder | implemented | verified-with-real-artifact on recorded Tier-B pin; scaled RoPE verified with synthetic fixtures | ByteLevel BPE; directory metadata/templates and tokenizer-backed text streaming | float safetensors, including index shards | float weights only | base, linear, dynamic NTK, Llama 3, YaRN RoPE; `use_sliding_window=true` rejected | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Mistral | implemented | verified-with-real-artifact on recorded Tier-B pin; synthetic Hugging Face tiny-checkpoint logits | Real-artifact chat text/IDs and synthetic family tokenizer golden | float safetensors, including index shards | float weights only | optional sliding-window masking; cache unbounded until 6.4 | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Gemma v1 | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer/template; text/ID golden | float safetensors, including index shards | float weights only | explicit head dimension and base RoPE; cache unbounded | Tier-B artifact/license/access record pending |
| Phi-3 | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer/template; text/ID golden | float safetensors with fused QKV and gate/up projections | float weights only | optional sliding-window masking; `longrope` rejected; cache unbounded until 6.4 | Tier-B artifact/license/access record pending |
| Mixtral / MoE | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer with Mistral template; text/ID golden | float safetensors with expert tensors | float weights only | optional sliding-window masking; dense MoE compute; cache unbounded | Tier-B artifact/license/access record pending |
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
