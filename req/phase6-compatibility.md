# Phase 6 compatibility matrix

This is the human-readable source of truth for local text inference support. “Implemented” means
the capability is present in a released API only after the Phase 6 gate says so; a synthetic test is
not evidence that an arbitrary Hugging Face artifact will load.

| Architecture | Status | Verification | Tokenizer / chat template | Checkpoint | Quantization | RoPE / context/cache policy | License / access notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama pre-norm decoder | implemented | verified-with-synthetic-fixture | byte-level BPE; caller renders chat prompt | safetensors, including index shards | float weights only | base RoPE; no scaling; unbounded per-generation cache | Tier-B artifact and license/access record pending |
| Qwen2 pre-norm GQA decoder | implemented | verified-with-synthetic-fixture | byte-level BPE; caller renders chat prompt | safetensors, including index shards | float weights only | base RoPE; no scaling; unbounded per-generation cache | Tier-B artifact and license/access record pending |
| Mistral | planned (6.3) | no verification | tokenizer/template golden required | safetensors | undecided | sliding window required | Tier-B artifact/license/access record required |
| Gemma | planned (6.3) | no verification | tokenizer/template golden required | safetensors | undecided | descriptor-dependent | Tier-B artifact/license/access record required |
| Phi | planned (6.3) | no verification | tokenizer/template golden required | safetensors | undecided | descriptor-dependent | Tier-B artifact/license/access record required |
| Mixtral / MoE | planned (6.3) | no verification | tokenizer/template golden required | safetensors | undecided | descriptor-dependent | Tier-B artifact/license/access record required |

The supported runtime is macOS on Apple Silicon with Java 25 and the MLX pins in
`scripts/bootstrap-native.sh`. Artifact licensing and access requirements are recorded alongside
each future Tier-B fixture; this matrix intentionally makes no claim that every public Hugging Face
checkpoint is compatible.
