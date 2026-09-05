# Phase 6 Tier-A fixtures

Tier A is deterministic, credential-free, and committed or generated from reviewable literals. It
runs on every relevant pull request; no test downloads a model or tokenizer.

| Fixture | Input / seed | Golden | Test or verifier |
| --- | --- | --- | --- |
| Python MLX array oracle | GPU; FLOAT32 logits `[[1,3,2],[5,4,3]]`, axis 1 | FLOAT32 doubled/softmax `[2,3]`; UINT32 argmax `[2]` with values `[1,0]` | `verifyMlxOracleFixtures` |
| Llama tiny decoder | generated zero-valued 1-layer safetensors; prompt `[1]`; two decode steps | common and legacy token IDs `[1,0,0]` | `LlamaModelTest.legacyAndCommonGenerationAgree` |
| Qwen2 tiny decoder | generated zero-valued 1-layer GQA safetensors; prompt `[1]`; two decode steps | common and legacy token IDs `[1,0,0]` | `QwenModelTest.loadsCheckpointAndGeneratesWithGroupedQueryCache` |
| Qwen-style byte BPE | text `low the`, no special tokens | IDs `[13,16]`; decoded text `low the` | `HfTokenizerTest.qwenStyleEncodeDecodeRoundTrips` |
| Llama-style byte BPE | text `low the`, special tokens enabled | IDs `[128000,13,16]`; decoded text `low the` when special tokens are skipped | `HfTokenizerTest` Llama 3 golden tests |

The decoder tests generate safetensors through `MLXIO.saveSafetensors`; no opaque binary checkpoint
is committed. `LlamaModelTest.closesGenerationScopesOnEveryTerminalPath` covers repeated cleanup,
listener failure, cancellation, EOS, stop-token, and max-token paths. The Python oracle's input and
canonical expected output live under `tools/mlx-oracle/fixtures/`; its exact interpreter/package
provenance is `tools/mlx-oracle/provenance.json`.
