# Phase 6 Tier-A fixtures

Tier A is deterministic, credential-free, and committed or generated from reviewable literals. It
runs on every relevant pull request; no test downloads a model or tokenizer.

| Fixture | Input / seed | Golden | Test or verifier |
| --- | --- | --- | --- |
| Python MLX array oracle | GPU; FLOAT32 logits `[[1,3,2],[5,4,3]]`, axis 1 | FLOAT32 doubled/softmax `[2,3]`; UINT32 argmax `[2]` with values `[1,0]` | `verifyMlxOracleFixtures` |
| Python MLX sampling oracle | GPU; six canonical greedy, penalty, filtering, and combined policies with explicit keys | Ordered post-policy logits, split keys, selected IDs, and post-filter log probabilities | `verifyMlxOracleFixtures`; `SamplingOracleTest.matchesCommittedPythonSelectionsAndLogProbabilities` |
| Llama tiny decoder | generated zero-valued 1-layer safetensors; prompt `[1]`; two decode steps | common and legacy token IDs `[1,0,0]` | `LlamaModelTest.legacyAndCommonGenerationAgree` |
| Fresh-JVM seeded Llama sampling | same tiny decoder; prompt `[1]`; seeds 42 and 7; 5/20 decode steps | committed exact token/logprob streams; short request is a prefix | `LlamaModelTest.seededSamplingRepeatsAcrossFreshJvmsAndRetainsPrefixes` |
| Qwen2 tiny decoder | generated zero-valued 1-layer GQA safetensors; prompt `[1]`; two decode steps | common and legacy token IDs `[1,0,0]` | `QwenModelTest.loadsCheckpointAndGeneratesWithGroupedQueryCache` |
| Qwen-style byte BPE | text `low the`, no special tokens | IDs `[13,16]`; decoded text `low the` | `HfTokenizerTest.qwenStyleEncodeDecodeRoundTrips` |
| Llama-style byte BPE | text `low the`, special tokens enabled | IDs `[128000,13,16]`; decoded text `low the` when special tokens are skipped | `HfTokenizerTest` Llama 3 golden tests |
| Hugging Face tokenizer component oracle | committed WordPiece/Bert, Metaspace-BPE, Metaspace-Unigram, and added-token JSON; ASCII, accented, CJK, supplementary, unknown/byte-fallback, truncation/padding inputs | IDs, decoded text, UTF-8 byte offsets, type IDs, attention mask, and special-token mask | `verifyTokenizerOracleFixtures`; `TokenizerOracleTest.componentFamiliesMatchPinnedOracle` |
| Tokenizer directory/chat contracts | local tokenizer config and Jinja templates; reserved context and template-only IDs | metadata/template precedence, rendered text, collision failures, incremental split-UTF-8 decode | `Phase62TokenizerContractTest` |
| Tokenizer-backed tiny decoder | generated tiny Llama checkpoint and local tokenizer; normal and pre-cancelled requests | generated IDs, event deltas/flush, result text, contextual decode abort | `LlamaModelTest.tokenizerBackedRequestsStreamTextAndFlushPreCancelledRequests`; `LlamaModelTest.tokenizerDecodeFailureIsAContextualGenerationAbort` |
| Tokenizer-backed Qwen2 tiny decoder | generated zero-valued 1-layer GQA safetensors and local tokenizer | generated IDs `[0,0]`; generated text `aa` | `QwenModelTest.loadsCheckpointAndGeneratesWithGroupedQueryCache` |

`verifyMlxOracleFixtures` verifies that the pinned Python environment reproduces every committed
oracle output. The original array row remains an environment self-check; Phase 6.1 closes the Java
differential loop for the sampling row by consuming its vocabulary-ordered post-filter logits,
selected IDs, and log probabilities in `SamplingOracleTest`. The other intermediate stage arrays
remain directly reviewable evidence.

The decoder tests generate safetensors through `MLXIO.saveSafetensors`; no opaque binary checkpoint
is committed. `LlamaModelTest.closesGenerationScopesOnEveryTerminalPath` covers repeated cleanup,
listener failure, cancellation, EOS, stop-token, and max-token paths. The Python oracle's input and
canonical expected output live under `tools/mlx-oracle/fixtures/`; its exact interpreter/package
provenance is `tools/mlx-oracle/provenance.json`.

The tokenizer oracle is the pinned Python binding over Hugging Face's Rust implementation. It runs
offline against committed synthetic files under `tools/tokenizer-oracle/fixtures`; exact package,
platform, and source-file hashes are recorded in `tools/tokenizer-oracle/provenance.json`. Its raw
Python offsets are Unicode-code-point indices and the runner converts them to the public jmlx
half-open UTF-8 byte-offset contract before comparison. These fixtures prove tokenizer components,
not additional model architectures or unrestricted compatibility with public artifacts.
