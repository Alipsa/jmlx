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
| Phase 6.3 checkpoint tensor plan | pure-Java fabricated names and safetensors headers/indexes, including mismatches | named missing, forbidden, extra, and misindexed keys before native loading | `TensorPlanTest`; `CheckpointIndexTest` |
| Hugging Face RoPE oracle | pinned `transformers==4.57.6`; fixed 16-wide input at offsets 0, 17, 80, 160 | inverse frequencies, attention scaling, rotated values for base, linear, dynamic NTK, Llama 3, YaRN | `verifyHfReferenceGoldens`; `RopeReferenceTest` on macOS CI |
| Sliding-window mask and attention | generated query/key positions around window boundary | BOOL mask with `p - j < window`; prefill/decode attention equivalence | `AttentionMaskTest`; `DecoderAttentionTest` on macOS CI |
| Phi-3 fused projection mapping | 2-layer seeded Hugging Face checkpoint with fused QKV and gate/up tensors | prefill logits and two decode logits | `Phi3ModelTest` on macOS CI |
| Mistral, Gemma v1, Mixtral tiny decoders | committed 2-layer, seeded nonzero Hugging Face safetensors | full prefill logits and two decode logits per family | `MistralModelTest`, `GemmaModelTest`, `MixtralModelTest` on macOS CI |
| Phase 6.3 family tokenizer bundles | synthetic family-shaped tokenizer/config files and chats | Hugging Face rendered chat text and IDs plus tokenizers encode/decode IDs | `Phase63FamilyTokenizerTest`; `Phase63ModelTokenizerContractTest`; `verifyTokenizerOracleFixtures` |
| Phase 6.4 Mistral window boundaries | committed window-4 tiny checkpoint; prompts of lengths 3, 4, 6 and 12; chunked prefill and 16 decode steps | independent eager Hugging Face last logits and greedy IDs in `mistral-window.json`; existing `mistral.json` unchanged | `verifyHfReferenceGoldens`; `DecoderSlidingWindowTest` on macOS CI |

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

## Phase 7.3a vision fixtures (2026-10-13)

| Fixture | Input / seed | Golden | Test or verifier |
| --- | --- | --- | --- |
| Pillow image decode oracle | committed PNG/JPEG decode fixtures (palette/tRNS, grayscale, LA/RGBA, ICC/gAMA, EXIF-orientation JPEG, decode-limit inputs) under `tools/image-oracle/fixtures/decode/`, plus reject cases (truncated PNG/JPEG, CMYK, Adobe transform-2 JPEG, 16/1/4-bit PNG, BMP) under `tools/image-oracle/fixtures/reject/` | byte-exact decoded RGB against pinned Pillow 12.3.0 (max channel error `0`); every reject case must fail jmlx decoding with the exact `IOException` message pinned in `ImageDecoderTest` (the oracle only digest-pins these inputs and their `expectedRejected` flag, never running Pillow on them) | `verifyImageOracleFixtures` (digests and the `expectedRejected` flag); `ImageDecoderTest` |
| Pillow resize oracle | committed resize fixtures, bilinear/bicubic/LANCZOS (`resample=1` three-lobe) | byte-exact uint8 resized output | `verifyImageOracleFixtures`; `ImageOracleResizeTest`, `ImageTransformsTest` |
| Idefics3 processor full chain | committed opaque-RGB fixture image plus the pinned SmolVLM `preprocessor_config.json` | byte-exact uint8 resize/split output; normalized float32 tiles within `1e-6` | `verifyImageOracleFixtures`; `ImageOracleChainTest`, `SmolVlmImageProcessorTest` |
| Tiny Idefics3 checkpoint | committed 2-layer seeded synthetic checkpoint (`tools/hf-reference/goldens/checkpoints/idefics3`) with the SmolVLM processor and chat fixtures | patch/positioned embeddings, tower, connector, merged embeddings, prefill/cached/full-history logits and exact five-token greedy IDs, strict absolute `1e-4` against the CPU float32 reference in `idefics3.json` | `SmolVlmGoldenTest` in `:jmlx-models:float32GoldenTest` |
| SmolVLM tokenizer bundle | committed `smolvlm.tokenizer.json` plus `smolvlm.input.json` chat cases | IDs, decoded text and rendered chat against the pinned Rust tokenizer oracle | `verifyTokenizerOracleFixtures`; `TokenizerOracleTest` |

The image oracle's exact environment (Pillow 12.3.0, NumPy 2.5.3, Transformers 4.57.6) is pinned in
`tools/image-oracle/provenance.json` and re-verified by `verifyImageOracle`; `verifyHfReferenceGoldens`
SHA-256-pins the tiny checkpoint, the `idefics3.json` numerics, the Tier-B logits golden, the
manifest, the licensed input image and the generator. The decoded-image fixtures exercise
`jmlx-vision`'s pure-Java path (no native runtime needed); the tiny-checkpoint row exercises the
native `SmolVlmModel` and runs only where the MLX runtime is available.

## Phase 7.3b Qwen3-VL WP8 fixtures (2026-10-26)

| Fixture | Input / seed | Golden | Test or verifier |
| --- | --- | --- | --- |
| Tiny WP8 full-model checkpoint | `tools/hf-reference/goldens/checkpoints/qwen3vl-wp8/` (strict 10/18/14-key `config.json`, 16,483,496-byte `model.safetensors` with 104 bf16 tensors and tied head, 14-key `preprocessor_config.json`, the committed 21-token qwen3vl fixture tokenizer, two deterministic 192x128 / 128x192 PNGs); generator seed 20261025 | `qwen3vl-wp8.json` (schema `jmlx-phase7-3b-wp8-qwen3vl-model-v1`): per-case unexpanded/expanded IDs, grids, processed-pixel SHA-256, full three-axis prefill positions and delta, tower merged/DeepStack rows, prefill and every step's fresh full uncached reference logits | `Qwen3VlGoldenTest` in `:jmlx-models:float32GoldenTest` (strict absolute `1e-4`; measured maximum `2.3245811e-6`) |
| Default-precision WP8 evidence | the same six cases (four plan cases plus both forced generated-marker continuations) in the ordinary TF32 JVM | measured maximum absolute logit/feature error `0.0019590855` under the separately named bound `0.0025`, recorded in `req/phase6-golden-precision.md` | `Qwen3VlDefaultPrecisionTest` in the ordinary `:jmlx-models:test` |

The generator `tools/hf-reference/qwen3vl_wp8.py` is never invoked by Gradle and is
SHA-256-pinned, together with the golden and every checkpoint file, in
`tools/hf-reference/provenance.json` (enforced by `verifyHfReferenceGoldens`, part of root
`check`). The exact greedy-ID assertion is degenerate on this 21-token tied checkpoint (token 7
at every base-case step; recorded in the golden's `greedy_id_note`); the binding gate is the
full logit-vector comparison. See `req/plans/phase7-3b-reference-findings.md` §11.8 for the
case inventory, the shared-contract coverage decision and the remaining Tier-B/8B scope.

## Phase 7.3c Phi-3.5-Vision WP6 fixtures (2026-10-28)

| Fixture | Input / seed | Golden | Test or verifier |
| --- | --- | --- | --- |
| Tiny WP6 `Phi3VModel` checkpoint | `tools/hf-reference/goldens/checkpoints/phi3v-wp6/` (21-key `config.json`, 76 BF16 tensors with the single shared `embed_tokens` / `vision_embed_tokens.wte` embedding stored once, 14-key `preprocessor_config.json`, the committed 21-token phi3v fixture tokenizer, deterministic PNGs); generator seed 20261025 | `phi3v-wp6.json` (schema `jmlx-phase7-3c-wp6-phi3v-model-v1`): per-case unexpanded/expanded IDs with the sentinel marker runs, per-image HD geometry and processed-pixel SHA-256, the CLIP tower's merged hd-transform rows, and prefill plus every cached step's fresh full uncached CPU float32 logits; six cases (text-only, wide-cross-boundary, square, wide-long-prompt, tall, two-image) | `Phi3VGoldenTest` in `:jmlx-models:float32GoldenTest` (strict absolute `1e-4`; measured maximum `1.5497208e-6`) |
| Default-precision WP6 evidence | the same six cases in the ordinary TF32 JVM | measured maximum logit/projected-row error `9.266138e-4` under the separately named bound `0.0012`, recorded in `req/phase6-golden-precision.md` | `Phi3VDefaultPrecisionTest` in the ordinary `:jmlx-models:test` |

The generator `tools/hf-reference/phi3v_wp6.py` is never invoked by Gradle and is SHA-256-pinned,
together with the golden and every checkpoint file, in `tools/hf-reference/provenance.json`
(enforced by `verifyHfReferenceGoldens`, part of root `check`). The Tier-B acceptance of the
pinned real 4B artifact is documented in `req/phase6-tier-b-artifacts.md`; the WP6 and WP8
evidence are in `req/plans/phase7-3c-reference-findings.md` §14 and §16.
