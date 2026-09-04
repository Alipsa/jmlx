# Phase 5 M3 retrospective — decoder proof

Phase 5 delivered a deliberately narrow native decoder proof in `jmlx-models`.

## Delivered

- Loads Hugging Face `llama` and `qwen2` `config.json` files and `.safetensors` checkpoints,
  including indexed shards.
- Maps the shared pre-norm decoder architecture into `jmlx-core` embeddings, grouped-query
  attention, RMSNorm, SwiGLU, linear output heads, and per-layer KV caches.
- Validates required dimensions, tensor names, bias rules, unsupported RoPE scaling, and
  sliding-window attention before returning a model.
- Greedily generates from token IDs inside a child `MLXScope`; weights remain owned by the
  caller's model scope, while activations and caches close at the end of generation.
- Verifies tiny Llama and Qwen checkpoints natively, including grouped-query cache operation and
  representative failure diagnostics.

## Deliberately omitted

- Sampling, seeds, log probabilities, callback streaming, cancellation, and batching.
- RoPE scaling, sliding-window cache eviction, model families beyond Llama/Qwen2, MoE, and GGUF.
- Tokenizer families beyond the existing pure-Java byte-level BPE path.
- Public release guarantees, real-artifact fixture automation, performance targets, and serving.

Phase 6 replaces the proof's model-specific entry points with a stable text-generation contract;
the legacy methods remain compatibility adapters while output equivalence is tested.
