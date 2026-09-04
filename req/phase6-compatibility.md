# Phase 6 compatibility matrix

This is the human-readable source of truth for local text inference support. “Implemented” means
the capability is present in a released API only after the Phase 6 gate says so; a synthetic test is
not evidence that an arbitrary Hugging Face artifact will load.

| Architecture | Status | Tokenizer / chat template | Checkpoint | Quantization | RoPE / context | Evidence and notes |
| --- | --- | --- | --- | --- | --- | --- |
| Llama pre-norm decoder | implemented; synthetic fixture | byte-level BPE; caller renders chat prompt | safetensors, including index shards | float weights only | base RoPE only; no scaling or sliding window | tiny native load/generation test |
| Qwen2 pre-norm GQA decoder | implemented; synthetic fixture | byte-level BPE; caller renders chat prompt | safetensors, including index shards | float weights only | base RoPE only; no scaling or sliding window | tiny native load/generation test |
| Mistral | planned (6.3) | tokenizer/template golden required | safetensors | undecided | sliding window required | no claimed support before Tier-A and Tier-B evidence |
| Gemma | planned (6.3) | tokenizer/template golden required | safetensors | undecided | descriptor-dependent | no claimed support before Tier-A and Tier-B evidence |
| Phi | planned (6.3) | tokenizer/template golden required | safetensors | undecided | descriptor-dependent | no claimed support before Tier-A and Tier-B evidence |
| Mixtral / MoE | planned (6.3) | tokenizer/template golden required | safetensors | undecided | descriptor-dependent | no claimed support before Tier-A and Tier-B evidence |

The supported runtime is macOS on Apple Silicon with Java 25 and the MLX pins in
`scripts/bootstrap-native.sh`. Artifact licensing and access requirements are recorded alongside
each future Tier-B fixture; this matrix intentionally makes no claim that every public Hugging Face
checkpoint is compatible.
