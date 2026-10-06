# Phase 7.2 — Text encoders and encoder-decoder generation

- **Status:** implemented locally, 2026-10-05; new macOS 26 CI acceptance remains pending.
- **Evidence:** `req/phase7-2-implementation-report.md`.
- **Parent:** `req/plans/phase7-plan.md` §7.2, decisions 2–3 and 8.
- **Prerequisite:** 7.1's oracle, inventory and native CI gates pass. Its local implementation
  status alone is not acceptance. 7.0 is not a dependency.

## 1. Scope and delivery result

Deliver BERT hidden states, sentence embeddings, sequence classification and token classification,
plus T5/Flan-T5 generation through the existing `TextGenerationModel` contract. Add reusable
bidirectional attention, descriptor-selected encoder blocks and static cross-attention to core.
Keep the existing decoder families and scheduler behavior covered by their current tests.

Use the existing modules: `jmlx-core`, `jmlx-models`, `jmlx-tokenizer` and the generate-only
`tools/hf-reference` and `tools/tier-b`. The dated parent scope amendment includes tokenizer
pair encoding and SentencePiece Precompiled normalization, plus task-aware Tier-B infrastructure.
No new published module, native pin change or binding regeneration is planned.
Model packages use core facades exclusively. Training, quantized BERT/T5 checkpoints, beam search,
decoder-prefix inputs, seq2seq scheduler cohorts, cache fork/reorder, other encoder families and
vision remain outside this milestone. Reject unsupported checkpoint variants explicitly.

## 2. Baseline and reference gate

`TextGenerationModels.load` currently parses a decoder `ArchitectureDescriptor` before dispatch.
`ModelMetadata` is sealed and permits only `DecoderMetadata`. `DecoderModel.generate` seeds
penalties from the source prompt and budgets its length against the cache. `AttentionMask` only
builds causal masks. `MultiHeadAttention` combines self-attention with optional RoPE/cache;
`DecoderAttention` has a fixed default scale and decoder-specific behavior. Do not force BERT or
T5 into those assumptions by inventing decoder descriptors.

Before public APIs or numerical code land, extend the pinned Transformers 4.57.6 reference tool
with an architecture/semantics report. Record source hashes and checkpoint config keys alongside
the existing provenance. The upstream inspection anchors are
[BERT](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/bert/modeling_bert.py),
[T5](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/t5/modeling_t5.py)
and [generation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py).
The installed pinned sources and executable CPU probes are the final reference.

The report must settle:

- BERT absolute positions, embedding normalization, residual order, pooler and task-head inputs.
- T5 projection dimensions (`num_heads * d_kv` need not equal `d_model`), norm accumulation,
  attention scaling, FFN activation names and tied output-head behavior.
- Relative-bucket sign, exact/logarithmic boundary, saturation and cached query offsets.
- Actual decoder IDs passed to repetition processors, start-token resolution and greedy IDs.
- Sentence-transformers module order, token inclusion in pooling and maximum sequence length.
- Serialized tied-weight aliases, optional BERT position/type buffers and artifact tensor names.

Add small native attention probes for BOOL versus additive masks, padding plus relative bias,
unscaled attention and rectangular projections. Use existing `MLXFast.scaledDotProductAttention`
(`mlx_fast_scaled_dot_product_attention`) and existing core compositions. If the native fused path
cannot preserve T5 mask/bias semantics, implement an explicit matmul/add/softmax/matmul core path,
verified against the same references. Record the choice; never drop a mask or bias.

## 3. Public contracts

### Encoder and classification entry points

Add `TextEncoderModel`, `SequenceClassifier` and `TokenClassifier` interfaces with metadata and
the same caller-owned-scope lifecycle as current models. Add `TextEncoderModels.load`,
`SequenceClassifiers.load` and `TokenClassifiers.load` factories taking `(MLXScope, Path)`.
Typed BERT implementations share an internal backbone. Each factory validates the requested task
against `architectures` and its tensor plan; a classification checkpoint cannot silently become
an embedding model by discarding its head.

Use `TokenizerEncoding` as the fundamental single-sequence input, preserving its IDs, type IDs,
attention mask, special-token mask and UTF-8 byte offsets. Callers tokenize explicitly; this
milestone adds no encoder/classifier text convenience overloads. Paired inputs use
an additive tokenizer pair-encoding overload with sequence/type IDs. The current `HfTokenizer`
and `EncodingOptions` support single sequences only: implement pair post-processing, truncation,
padding and per-sequence offset alignment, with Rust tokenizer-oracle fixtures, before offering
MRPC Tier-B execution. Pair support is mandatory for the selected sentence-pair classifier.
Initial execution is one sequence per call; do not introduce
scheduler or public batching promises.

Proposed result records:

- `EncoderResult(TokenizerEncoding encoding, float[][] hiddenStates, float[] embedding)`;
  rows retain input alignment, including padding. Consumers use the encoding mask.
- `SequenceClassificationResult(float[] logits, float[] scores, List<String> labels)`.
- `TokenClassificationResult(TokenizerEncoding encoding, float[][] logits, float[][] scores,
  List<String> labels)`; prediction rows align with every input token. Special/padded rows are
  identifiable through the encoding and are not converted into entity spans.

Every array component is copied on construction and access, including nested rows. Records never
expose `MLXArray`. Labels are immutable and ordered by numeric `id2label` index; validate complete,
non-conflicting indices against head width. Without a mapping use documented `LABEL_n` defaults.
Single-label classification uses softmax; multi-label uses sigmoid when `problem_type` explicitly
selects it. Reject declared regression and missing `problem_type` with `num_labels == 1`;
otherwise an absent problem type defaults to single-label classification. Validate explicit
problem types and reject unknown values.

`Pooling` selects CLS, MEAN or MAX with an independent L2-normalization flag. Mean/max exclude
padding; attended special tokens remain included, matching the selected artifact pipeline. CLS
uses index 0, as do sentence-transformers CLS pooling and `BertPooler`, and requires that position
to be attended; never search for a CLS token ID. Reject all-masked input; zero vectors
remain zero under normalization. Without sentence-transformers metadata, default to CLS without
normalization; explicit caller policy overrides artifact defaults.

Read `modules.json`, the declared pooling module config (normally `1_Pooling/config.json`),
`sentence_bert_config.json` and any Normalize module. Support Transformer → one supported Pooling
→ optional Normalize. Reject multi-pooling concatenation and unsupported Dense/custom modules.
Resolve every non-empty module path as a canonical relative POSIX path beneath the model directory.
Reject absolute paths, dot/dot-dot or empty components, backslashes and symlink escapes; require
the resolved real path to remain beneath the model directory's real path before reading files.
An empty path is allowed only for the root Transformer module. Test the Java loader independently
of downloader protections with crafted local metadata and symlink fixtures.
Honor the artifact's maximum sequence length; core BERT additionally enforces position capacity.
Deliberate sentence-transformers difference: jmlx rejects over-limit encodings instead of silently
truncating to `max_seq_length` (256 for the MiniLM candidate, to be checked at its pinned revision).
Callers may opt into explicit tokenizer truncation before inference.

### Generation and metadata

Add `T5Model implements TextGenerationModel`, composing an encoder and a seq2seq decoder stack;
it does not extend `DecoderModel`. Dispatch `model_type=t5` before decoder-only parsing in
`TextGenerationModels.load`; preserve the existing typed decoder loaders.

Extend the sealed metadata permits list with internal encoder and seq2seq implementations.
Keep existing metadata methods; document `numHiddenLayers()` as decoder depth for generation
models and encoder depth for encoder-only models. Expose encoder depth separately for T5 via an
additive metadata accessor with a compatible default. No image payload or modality enum is added
here; those remain 7.3a work.

### New public model/tokenizer signatures

The following is the complete proposed model/tokenizer surface; result record components are
listed above. All loaders declare `throws IOException`; all models extend `AutoCloseable` with
the existing caller-owned-scope no-op `close()` convention.

```java
interface TextEncoderModel {
  ModelMetadata metadata();
  EncoderResult encode(TokenizerEncoding input); // artifact/default pooling
  EncoderResult encode(TokenizerEncoding input, Pooling pooling); // explicit override
}
interface SequenceClassifier {
  ModelMetadata metadata();
  SequenceClassificationResult classify(TokenizerEncoding input);
}
interface TokenClassifier {
  ModelMetadata metadata();
  TokenClassificationResult classify(TokenizerEncoding input);
}
record Pooling(Mode mode, boolean l2Normalize) { enum Mode { CLS, MEAN, MAX } }
record T5LoadOptions(int maxSourceTokens) { static T5LoadOptions defaults(); }
// Static factories on the corresponding plural utility classes:
TextEncoderModel TextEncoderModels.load(MLXScope scope, Path directory);
SequenceClassifier SequenceClassifiers.load(MLXScope scope, Path directory);
TokenClassifier TokenClassifiers.load(MLXScope scope, Path directory);
// Static factories on T5Model:
T5Model T5Model.load(MLXScope scope, Path directory);
T5Model T5Model.load(MLXScope scope, Path directory, T5LoadOptions options);
// Additive overload; options apply only to model_type=t5, otherwise fail explicitly:
TextGenerationModel TextGenerationModels.load(
    MLXScope scope, Path directory, T5LoadOptions options);
// Additive metadata method: BERT depth, T5 encoder depth, zero for decoder-only models:
default int ModelMetadata.numEncoderLayers();
// Pair strategy is explicit without changing Truncation's record components/equality:
enum PairTruncationStrategy { LONGEST_FIRST, ONLY_FIRST, ONLY_SECOND }
record PairEncodingOptions(EncodingOptions options, PairTruncationStrategy strategy) {}
// Additive instance overload on HfTokenizer:
TokenizerEncoding HfTokenizer.encode(String text, String textPair, PairEncodingOptions options);
```

The signature inventory uses qualified method names for clarity, not compilable Java declarations.
`T5LoadOptions` requires a positive limit; `defaults()` returns 512. Existing no-options loaders
delegate to defaults for T5. Pooling validates a non-null mode. Pair encoding offsets are UTF-8
byte ranges local to the corresponding input; BERT type IDs distinguish the two sequences, while
special-token masks identify synthetic tokens. This is explicitly a BERT-template limitation:
pair post-processors must assign type ID 0 to first-input tokens and 1 to second-input tokens.
Reject pair templates that cannot preserve that distinction; no generic sequence-ID accessor or
change to `TokenizerEncoding` record equality is promised. Special/padding tokens have no source
offset even when they carry a type ID. No new normalizer type leaks into public API.

`PairEncodingOptions` requires both components to be non-null. Its `EncodingOptions` carries
length, direction, special-token and padding policy; the separate strategy chooses which input
loses tokens. Reserve pair special-token capacity first. Let the locked Rust oracle determine
ONLY_FIRST/ONLY_SECOND failure boundaries, including whether removing the entire selected
sequence is forbidden, and LONGEST_FIRST tie-breaking. Disabled truncation ignores strategy.
Keep `Truncation` and its existing
single-sequence constructors unchanged. Preserve the configured tokenizer JSON strategy internally;
support all three named strategies for pair use. For single-sequence ONLY_SECOND, probe both
under-limit and over-limit cases; do not reject merely because truncation is enabled if the
reference returns before removing tokens. Explicit pair options override configured strategy.
Document the compatibility change in the tokenizer CHANGELOG: configured ONLY_SECOND formerly
failed at JSON load and will now load, with encode-time success/failure following the oracle.
Core constructor/method signatures are fixed by the §2 probe gate and documented before WP3;
they must include explicit masks, bias, projection dimensions and request-scope ownership.

## 4. Shared core modules and ownership

| Change | Contract |
| --- | --- |
| `AttentionMask.bidirectional` | BOOL `[B,1,T,S]`, true means attend; expand only the key-padding mask across all query rows, including padded queries. No causal restriction or query-row substitution. Reject an input with no valid key. Padded hidden states remain defined and are compared to HF. |
| Attention scale | Add overloads to `DecoderAttention` and `MultiHeadAttention` and a parameter on new attention modules. Existing constructors delegate with exactly the current `1/sqrt(headDim)` calculation. Validate finite positive scale; T5 passes `1.0f`. |
| `EncoderBlock` | Descriptor selects pre/post norm; explicit mask and optional additive position bias; registered attention, normalization and FFN children. No RoPE for BERT/T5. |
| `CrossAttention` | Separate query input and encoder K/V projections; explicit `headDim`, scale and source mask. Support rectangular projection/output dimensions. |
| `StaticKVCache` | Non-appending, initialized once per decoder layer with projected `[B,H,S,D]` arrays, owned by an explicit request scope. No eviction, append, reorder or fork. |

Use a dedicated bidirectional attention composition rather than changing the semantics of existing
`MultiHeadAttention.forward(x, cache, causal)`. Reuse projection/head helpers where useful. T5
self-attention needs an explicit no-RoPE, relative-bias path; implement it as a separate internal
composition if decoder-specific assumptions make reuse unsafe. Shared core modules must not depend
on T5 config types from `jmlx-models`.
The scale overloads on existing attention classes are required by the parent §7.2 even though
T5 cannot use their current RoPE/mask interfaces. They are additive compatibility work, not the
T5 execution path; T5 uses the new mask/bias-aware no-RoPE composition.

Parameters register through `param`, children through `child`; forward reads current parameters.
Derived weights and position tensors must not accumulate on model scopes. New modules follow
recursive train/eval state; inference uses eval and training failures follow Phase 7.1 policy.

All request state lives inside `try (MLXScope generation = modelScope.newChild())`. Encode once in
a short-lived child, explicitly hoist encoder output into `generation`, project static K/V once
per decoder layer in another child and hoist each result into `generation`. Evaluate retained
arrays before closing producer scopes and test consumption after those scopes close. Release the
encoder output in a nested owner scope when projections no longer need it, or retain it until
request close with its one-time memory cost documented. Never retain activation-scope arrays.
Self-attention caches also belong to `generation`; each token has a child activation scope.

No new raw binding is currently required. If implementation needs one, identify the exact mlx-c
declaration, add a core facade/native probe, update inventory overrides and regenerate the inventory
before using it. Generated sources remain untouched. Scope inference, confinement, per-thread
streams and `MLXException` remain unchanged.

## 5. BERT loader and numerical execution

Create separate encoder/task descriptors and mappings, reusing `CheckpointLoader` and `TensorPlan`
validation mechanics where possible. Do not expand the decoder descriptor into a union of every
model architecture. Parse all selected artifact keys into supported, validated-inert or rejected
categories; commit a mapping table. Reject decoder-mode BERT, relative-position variants, pruned
heads and unsupported activation/quantization settings.

| Tensor group | Expected shape / handling |
| --- | --- |
| word / position / token-type embeddings | `[vocab,hidden]`, `[max_positions,hidden]`, `[type_vocab,hidden]` |
| embedding and block LayerNorm | weight/bias `[hidden]`, config epsilon |
| Q/K/V and attention output | weights `[hidden,hidden]`, biases `[hidden]` |
| FFN input/output | `[intermediate,hidden]` / `[hidden,intermediate]`, respective biases |
| pooler | dense `[hidden,hidden]` plus bias, tanh |
| task classifier | `[num_labels,hidden]` plus bias |

Handle bare `BertModel` names versus the `bert.` task prefix explicitly. Pooler is optional for
backbone/token-head artifacts; sequence classification requires it. Unexpected heads fail.
Allow only pinned, verified non-parameter position/type buffers with shape/value checks.

Compose embedding lookup/sum/LayerNorm, absolute input-index positions, bidirectional attention,
post-LN residual blocks and exact GELU. Padding does not renumber absolute positions. Sequence
classification uses pooler output; token classification uses final hidden states. Sentence
pooling uses final hidden states rather than the BERT tanh pooler.

Before allocating any native arrays, every BERT entry point validates a non-empty encoding,
equal column cardinalities, `0 <= id < vocab_size`, `0 <= typeId < type_vocab_size`, binary
attention/special-token masks and at least one attended key, plus artifact sequence limit and
position capacity. Although `TokenizerEncoding` already checks cardinalities, retain that
contract at the model boundary. CLS pooling and pooler-based classification additionally require
index 0 to be attended. Invalid token/type IDs must fail in Java, without relying on native
gather bounds checking.

## 6. T5 architecture and dtype policy

Map separate encoder/decoder depths, `d_model`, `d_kv`, heads, `d_ff`, epsilon, bucket count,
maximum relative distance, `feed_forward_proj`, tied embeddings and token IDs. Support dense ReLU
and gated GELU with the pinned reference activation; reject other variants explicitly. Validate
all dimensions and serialized weight aliases before returning a model.

Tensor-plan groups are `shared.weight`; encoder/decoder block self-attention `q/k/v/o.weight`;
first-block relative-bias tables `[buckets,heads]`; decoder cross-attention `q/k/v/o.weight`;
layer norms and final norms; dense `wi/wo` or gated `wi_0/wi_1/wo`; and untied `lm_head.weight`.
Projection shapes follow `[heads*d_kv,d_model]` and `[d_model,heads*d_kv]`. Decoder FFN/norm layer
indices differ because cross-attention occupies layer 1. Tied aliases may be omitted by
safetensors serialization; derive only declared aliases and validate any duplicates for equality.

Implement bucket mapping as a pure Java function with exact edge tests and a reference fixture.
Encoder and decoder each own a first-layer table. Compute bias once per encoder forward or decoder
step and share it across that stack's layers; do not cache successive bias matrices on the model.
Cached decoding uses absolute query positions and retained key positions. Cross-attention has no
learned relative bias. Apply source padding and decoder causality explicitly.

T5 uses scale 1 and RMS normalization without mean subtraction or bias. Apply hidden-state
rescaling only when `tie_word_embeddings` is true; Flan-T5's untied configuration does not use it.
Verify these choices against the pinned implementation linked in §2.

**Dtype decision:** execute all BERT/T5 weights, activations, norms, logits and caches in FLOAT32
for initial support. FLOAT16/BFLOAT16 safetensors weights are promoted once during loading;
integer/affine-quantized weights fail. T5 intermediates can overflow float16; do not emulate a
reduced-precision path or silently clamp to make a test pass. FLOAT32 storage does not disable
TF32 matmul: both existing Gradle precision modes remain mandatory. Reduced-precision inference
requires a later measured amendment.

## 7. Seq2seq generation semantics

Reuse `SamplingPipeline`, `PenaltyInputs`, `IncrementalTokenDecoder`, cancellation and event/result
types. Extract small architecture-neutral output/lifecycle helpers from `DecoderModel` only where
they remove duplicated behavior. Do not initially refactor its entire inference loop.

| Concern | T5 contract |
| --- | --- |
| Source | Request prompt IDs are encoder input, never decoder history. Empty/out-of-vocabulary sources fail before native work. Token-ID requests attend every supplied ID, including a pad ID; do not infer padding from token values. HF-parity fixtures and Tier-B use `GenerationRequest.text` with `PromptSpecialTokens.ADD`, preserving the tokenizer's terminal `</s>`. `OMIT` remains allowed and is a documented non-HF prompt policy; raw IDs are never rewritten. |
| Start token | First non-null `decoder_start_token_id` in `generation_config.json`, then `config.json`; validate vocabulary range. Missing value fails at load. No BOS fallback. |
| Penalties | Initialize decoder frequencies with the start token once, then emitted decoder IDs. Encoder IDs never contribute. Capture HF processor inputs to verify this rule. Frequency/presence penalties are jmlx extensions, applied to the same decoder history. |
| Budget | Start token is fed but not emitted or counted. For `N > 0`, required self-cache positions are `1 + N - 1 = N`; for `N == 0`, zero. Checked long arithmetic, rejection before encoding for bounded FULL overflow. |
| Source limit | Default maximum 512 tokens, an explicit jmlx resource limit rather than a T5 architectural position limit. Add `T5LoadOptions.maxSourceTokens` for an intentional positive override, and validate before native work. After projection, before the first decoder step, check every layer's static K and V source dimension equals the encoder output length (itself equal to validated source length), with matching batch/head/d_kv dimensions. |
| Cache policies | Unbounded FULL and capacity-bounded FULL supported. Resolve the default to FULL. Reject both `SLIDING_WINDOW` and `SLIDING_WINDOW_FROM_MODEL` before calling `GenerationCachePolicy.resolve`, with `IllegalArgumentException` naming `t5` and the requested policy. |
| Result | Keep original source in prompt IDs; generated IDs/text contain only selected target tokens. `tokenIds()` retains the existing source-plus-generated meaning; it is not a decoder replay input. |
| Termination | Existing EOS precedence and EOS inclusion, explicit-stop exclusion, log-probability alignment, listener failure and terminal-event rules remain identical. |
| Zero/cancelled work | `maxNewTokens == 0` or cancellation before encoding returns without running the encoder. Check cancellation after encoding, after static projections and before every decoder step. |

HF supplies the initial decoder token through its
[decoder-input preparation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py).
Record executable repetition-processor inputs, not just generated output. Deliberate differences
from HF: jmlx has no BOS fallback, retains source-plus-generated result IDs, implements additional
frequency/presence penalties, and applies a configurable source resource limit. Do not import HF
beam/cache or generation options that the current `GenerationConfig` does not support.
`PromptSpecialTokens.OMIT` is also a deliberate prompt-format difference. The separate
sentence-transformers over-limit rejection difference is specified in §3.

Keep the existing request EOS policy resolution; convenience factories may read checkpoint EOS
defaults consistently with current loaders. Do not treat a start token equal to PAD/EOS as an
already-finished generation. Partial abort results preserve source IDs and emitted target IDs.

Tighten `BatchGenerationScheduler.acceptModel` to throw `SchedulerStartException` containing the
model type and “not supported by the batch scheduler” for a T5 factory. Keep null-factory diagnostics
safe. Assert failure closes worker resources and releases the process scheduler guard. Image
admission validation remains coupled to the 7.3a payload API.

## 8. Fixtures, artifact pins and licensing gate

Extend `tools/hf-reference/generate.py` with BERT backbone/sequence/token-head cases and T5 dense
ReLU, gated GELU, tied/untied head cases. Use deterministic tiny safetensors checkpoints and CPU
float32 eager attention. Preserve existing family files and pin every new file in provenance.
Add padded/paired BERT inputs, hidden states, pooler values, all pooling outputs, task logits and
Unicode offsets. T5 fixtures include encoder output, prefill, at least four cached steps, full
uncached comparison, bucket boundaries and processor-history traces.

Add WordPiece and Unigram family encode/decode goldens through the pinned tokenizer oracle.
Add pair-encoding and Precompiled-normalizer fixtures through the dedicated tokenizer WP2 below.
Neither BERT nor Flan-T5 needs an invented chat template: record ordinary paired/sequence encoding
and text-to-text prompt formatting; unsupported chat use is explicit.

| Tier-B artifact | Selection / current verification |
| --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | Pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; BERT, safetensors and Apache-2.0 verified, including pooling and sequence-limit files. Local actual sentence-transformers comparison passed. [Metadata](https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2). |
| `google/flan-t5-small` | Pinned revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`; Apache-2.0, safetensors and real Precompiled normalizer verified. Local full-logit and greedy comparison passed. |
| BERT sequence classifier | Selected fallback `yoshitomo-matsubara/bert-base-uncased-sst2`, revision `ce4cfd087e0c988beae0699f77a610bf7916742c`, Apache-2.0. Intel MRPC has no safetensors at its inspected pin; the fallback is single-sequence. BERT pair API and oracle work remain included. |

**Closed local artifact gate, 2026-10-05:** all consumed files have SHA-256/size manifests,
immutable full revisions and license evidence under `tools/tier-b/`. The SST-2 fallback satisfies
the classifier gate; no mutable main download is used. Real-checkpoint error and stable-margin
eligibility passed locally. The new macOS 26 native/weekly runs remain pending acceptance.

**Reference dependency decision (implemented with 5.1.2):** add a compatible, pinned
`sentence-transformers`
version to `tools/hf-reference/requirements.in`, regenerate the hash lock and provenance, and
review the environment/golden diff. WP1 selects the version against existing Torch/Transformers
pins; no guessed version is promised here. Generate MiniLM reference embeddings using the actual
`SentenceTransformer` pipeline, including pooling, Normalize and sequence-limit behavior; do not
use a Java-equivalent hand-reimplementation as the acceptance oracle. Intermediate BERT references
continue to come from Transformers. Record module configs and dependency/source hashes.

### Tier-B infrastructure work (WP4)

The current downloader rejects subpaths, and `TierBSmokeTest` assumes chat-based `DecoderModel`
generation of 16 tokens. These are implementation prerequisites, not reusable behavior for 7.2.

- Replace `download.py` with a Java build-tool helper, `TierBArtifactDownloader` in buildSrc,
  exposed through root `downloadTierBArtifact` using explicit manifest/target Gradle properties.
  Declare no task outputs and call
  `doNotTrackState("downloads pinned external artifacts with their own SHA-256 verification")`.
  Do not make the task cacheable or fingerprint/cache its potentially multi-GB target directory.
  The downloader's own per-file SHA-256 checks decide whether previously downloaded files are
  current on every invocation, including when Gradle build caching is enabled.
  Use Java `HttpClient`, NIO and `MessageDigest`. Extract the existing private
  `MlxApiInventory.JsonParser` into a package-private shared buildSrc class and extend it to
  full JSON: objects/arrays/strings, integers as overflow-checked `long`, decimals/exponents as
  `BigDecimal`, booleans and null. Support every JSON escape, including control-character
  escapes, `\uXXXX` and surrogate pairs; reject malformed escapes and unescaped control
  characters. Enforce JSON number grammar and reject integer overflow rather than rounding.
  Manifest schema validation requires integral, non-negative long byte caps and valid finite
  reference bounds; parsing a decimal must not silently make it an acceptable byte cap.
  Preserve the inventory's narrower contract using a strict parser mode that rejects numeric,
  boolean and null values and retains its existing escape restrictions. Keep existing inventory
  tests unchanged. Add shared-parser tests for each new value/escape kind, signed-long limits,
  overflow, fractional/exponent syntax, surrogate pairs and malformed input, plus manifest cases
  covering existing token arrays/chat newlines and new reference vectors/bounds.
  Do not introduce Groovy imports or a new JSON parser dependency.
  Support canonical relative POSIX subpaths such as
  `1_Pooling/config.json`. Reject absolute paths, empty/dot/dot-dot components, backslashes,
  duplicate normalized paths and symlink escapes; create safe parent directories and keep
  atomic per-file writes, hash verification and both size caps. Test valid nested downloads and
  traversal/symlink failures with local fixtures, including cleanup after failed downloads.
  Configure `HttpClient.Redirect.NORMAL`, permitting HTTPS redirects to CDN hosts and refusing
  HTTPS-to-HTTP downgrade. HEAD/GET size metadata must come from successful final responses,
  not intermediate redirect bodies; validate status and byte/hash caps on the final response.
  Use `BodyHandlers.ofInputStream()` and a bounded copy loop into the `.part` file. Enforce
  per-file and aggregate caps on every chunk, including cached-file accounting, and abort/close
  the response stream and delete partial files immediately on failure. Never buffer an artifact
  into a byte array or use an unbounded `ofFile` download in the Gradle daemon.
  Inject an internal transport interface for HEAD/GET responses; production uses the fixed Hub
  endpoint, while tests supply deterministic responses. Add redirect-chain, cross-host CDN,
  unfollowed final 3xx status, incorrect/missing final Content-Length, and streamed over-cap cases.
  Verify the production HttpClient redirect configuration separately from the injected transport;
  a fake that only returns a final 200 response cannot prove redirect behavior. Provide a
  package-private production-transport factory accepting a base URI; the Gradle task always
  supplies the fixed HTTPS Hub URI, while tests supply local plain-HTTP server URIs.
  Exercise redirect chains and cross-host redirects using `localhost` versus `127.0.0.1`
  with local servers (and separate ports as needed); do not require `127.0.0.2` to be configured.
  Assert the constructed client's `followRedirects() == Redirect.NORMAL`. HTTPS-to-HTTP
  downgrade non-following relies on that JDK policy contract, not an injected-transport or local
  TLS integration test. The downloader rejects the resulting unfollowed 3xx through its final
  non-2xx status check; test that status rejection and cleanup through the injected transport.
  No test certificates, keystores, keytool invocation or TLS trust override is needed; production
  TLS validation stays at its default.
  Add `TierBArtifactDownloaderTest` to buildSrc's existing JUnit suite. Its entry point is
  `./gradlew -p buildSrc test`. Follow the existing isolated-build convention: buildSrc tests
  remain outside root `check`/`build`, and the existing Java CI job's
  `./gradlew -p buildSrc check` runs the new suite and its style checks. Do not add a nested
  GradleBuild task or run the suite a second time through root `check`. Keep helper/test code
  compatible with buildSrc's Java 21 toolchain.
  No Python executable or oracle venv is needed for ordinary builds or downloader tests.
  Keep test execution independent of native staging. Cover byte/hash
  limits and partial-file cleanup as well as path containment; no live Hub access in these tests.
- Add manifest `task`: `embedding`, `classification`, `seq2seq` or `generation` (legacy manifests
  default to generation). Add task-specific inputs and expectations: pooling/vector references,
  single/pair texts plus class/logit references, or source text plus target IDs/finish reason.
  Include revision, tokenizer/config hashes, precision and prompt-special-token policy.
- Add `TierBEmbeddingTest`, `TierBClassificationTest` and `TierBSeq2SeqTest`; dispatch by manifest
  task before choosing a loader. Encoder tests use their task factories; seq2seq uses
  `TextGenerationModels.load` without a `DecoderModel` cast and `GenerationRequest.text(…ADD…)`.
  Compare the complete target sequence, including legitimate early EOS, and finish reason
  using the stable-margin reference cases below;
  remove any fixed 16-token requirement from the seq2seq path.
- Add an optional `JMLX_TIER_B_TASK` assertion. Model type remains an architecture assertion,
  not task identity; the two `bert` artifacts are distinguished by task and manifest repository.
- Add three named matrix rows in `tier-b.yml`, plus task and expected test-suite fields.
  Gradle selection and XML execution assertions must target the selected task's suite, require
  nonzero executed tests with no skips, and retain existing decoder matrix behavior.
  Switch every download invocation in tier-b.yml and README/AGENTS examples to the Gradle task;
  remove the Python downloader once migrated, avoiding two implementations of path/hash policy.
  Add a dated migration note to `req/plans/phase6-4-1-plan.md` beside its historical
  “tools/tier-b/download.py is unchanged” statement when the Python downloader is removed.
- Margin eligibility is a two-stage process. WP1 selects candidate inputs, not acceptance inputs.
  State a provisional per-logit absolute `epsilon = 1e-4` for TF32-off screening only; it is not
  an established Tier-B error bound and is not inferred from tiny Tier-A results. Record full HF CPU logits,
  top-1/top-2 IDs and their gap for each class decision and each generated step, including EOS,
  together with the exact decoder history. If the allowed per-logit absolute error is `epsilon`,
  require the minimum gap to exceed `2 * epsilon` plus a documented positive safety margin.
  Use a provisional safety margin of `1e-4` during WP1 screening. A gap merely exceeding
  `epsilon` is insufficient because both competing logits can move. WP5 measures the real
  MiniLM/classifier errors; WP7 measures Flan-T5 errors at every recorded decoder history.
  Record a reviewed allowed error bound and positive safety margin per real artifact, then
  re-run eligibility using those final values. Replace failing candidate inputs, regenerate
  their HF references and update manifest hashes before acceptance. A WP1 screening pass never
  closes this gate. Do not loosen the existing strict Tier-A `1e-4` assertions.
  For multi-label threshold 0 (sigmoid score 0.5), require `abs(logit) > epsilon + margin` for
  every asserted label decision, because only one logit moves relative to that threshold.
- Always compare embedding/logit values under recorded precision bounds. Exact class IDs and
  full greedy sequences are acceptance assertions only for cases that pass the margin gate;
  choose replacement inputs before accepting a near-tie case. Such near-tie cases may remain
  numerical diagnostics, teacher-forced over the recorded history, without host-independent ID
  assertions. Tuple-specific exact observations may be recorded separately, as with Qwen3.
- `tierBTest` already forces TF32 off; retain that mode for the new task tests. A new host must
  still meet the recorded numerical bound and stable-margin IDs. Do not assume FLOAT32 storage
  or TF32-off ensures bit-identical CPU/Metal arithmetic. Report numerical comparison, margin
  eligibility and tuple match separately; never silently downgrade to structural acceptance.
  A bound change requires reviewed new measurements and rechecking every margin. Preserve legacy
  generation-manifest behavior explicitly if those manifests are not migrated here.

Measure download size, JVM RSS and native peak memory before scheduling separate artifact jobs
on the recorded small Tier-B runner. No network/model download belongs to ordinary tests.

## 9. Verification and memory evidence

- Pure Java: descriptor/key handling, tensor plans/aliases, labels, record defensive copies,
  pooling metadata and module-path containment, bucket boundaries, source/start/capacity
  validation and UTF-8 offsets. BERT boundary cases include empty input, mismatched columns,
  negative/out-of-vocabulary IDs, invalid type IDs, masks and sequence/position overflow.
- Core native: bidirectional masking, additive bias, scale defaults versus explicit scale,
  rectangular cross-attention, pre/post norm and static-cache ownership/confinement. Prove K/V
  projections initialize once per static cache; test this locally in the core package.
- Models integration: add a package-private observer on T5's internal request-state/decoder
  assembly in `se.alipsa.jmlx.models`, injected through a models test-only constructor or factory.
  Notify it at the actual K/V projection-and-static-cache initialization call, not at unrelated
  generation entry. A `T5Model.generate` test in that package counts each layer once across
  several decode steps, then once again with distinct request/cache identity for a second source.
  All initialization calls route through that observed path. Production uses a no-op observer;
  no cross-package package-private constructor access or public instrumentation API is needed.
- Family correctness: `full-float32` suites in `float32GoldenTest`, strict absolute `1e-4`
  HF output/logit bound; exact greedy IDs. Preserve current strict assertions.
- Default TF32: separate named suites/bounds, measured per BERT/T5 variant and recorded in
  `req/phase6-golden-precision.md`. Never regenerate a reference to fit TF32 error.
- Seq2seq: source-only repeated token gets no first-step penalty; start token does; long-source /
  short-target and reverse capacity boundaries; zero/one/several generated tokens; start omitted
  from events/results; EOS/stop overlap; sampled reproducibility and filtered log probabilities;
  unsupported sliding/missing start; cancellation and listener/tokenizer/native failure cleanup.
- Cache agreement: cached versus full uncached T5 logits over identical decoder history in both
  precision modes, including relative buckets at long offsets and different source lengths.
- Regression: all current decoder and scheduler suites; existing constructors retain default scale.

Use `@EnabledIfNativeAvailable`, never a new file-presence gate. Add new mandatory native classes
to `.github/workflows/ci.yml`'s required-suite lists in the matching result directory. Classes
tagged only `full-float32` appear only under `float32GoldenTest`; do not require them in `test`.

Create `Seq2SeqMemoryTest` using `MLXMemory.activeBytes()` after synchronized evaluation and
warm-up. Repeated same/different sources and every terminal failure return to baseline within a
measured fixed allocator margin. Vary source length to distinguish request retention from model
growth. Reuse the scheduler memory test's early/late slope analysis.

For FLOAT32 self-cache, retained bytes per target position are
`2 * decoderLayers * heads * d_kv * 4` for batch 1. Static cross-K/V adds this same coefficient
times source length once; encoder output adds `sourceLength * d_model * 4` if retained. Document
temporary concatenation/evaluation overhead as a measured multiplier plus a fixed warm-up margin.
Apply the derived budget to new seq2seq tests. Changing `BatchSchedulerMemoryTest`'s existing
4 MiB budget is separate follow-up work, as recorded by the parent's existing 2026-10-05
amendment to public-contract decision 8; it is not a 7.2 dependency.
Bounded FULL must reject an over-budget request before encoding; it never
evicts and has no plateau requirement. T5 sliding retention is rejected, so no T5 plateau claim
is made. Existing decoder sliding tests retain their plateau requirement.

## 10. Work packages and exit gates

1. **References and pins:** close §8's artifact gate; produce config/tensor mapping tables,
   architecture/processor probes, deterministic fixture generator extensions and provenance;
   pin/install the actual sentence-transformers reference dependency and review its lock diff.
   Select Tier-B candidate inputs and record gaps using §8's stated provisional bound/margin.
   Final eligibility depends on real-checkpoint measurements in WP5/WP7, not a WP1 pass.
   Confirm 7.1 acceptance before native implementation proceeds.
2. **Tokenizer compatibility (mandatory):** inspect Flan-T5's pinned tokenizer JSON, including
   normalizer sequences. Implement pure-Java SentencePiece Precompiled charsmap decoding and
   application if present: validate base64/binary structure, reject malformed tables safely,
   preserve normalized-to-original UTF-8 alignment and compose with other normalizers. Add
   Rust-oracle fixtures for mappings, deletion/expansion, whitespace, combining characters,
   multibyte input, offsets and malformed payloads. If the pinned artifact has no Precompiled
   component, record the hashed evidence and explicitly narrow this task before implementation.
   Fixture source: commit a derived `*.tokenizer.json` that retains the real, byte-identical
   `precompiled_charsmap` from the immutable Flan-T5 tokenizer, rather than hand-writing a Darts
   table. Record source repository/revision, original file SHA-256/byte size, extraction script
   and retained charsmap hash; trim unrelated JSON only. Include the artifact's Apache-2.0
   notice and record the committed fixture's byte size. Pin both the derived tokenizer and
   `*.input.json` in `tools/tokenizer-oracle/provenance.json`; generate `*.expected.json` through
   the locked Rust oracle. Keep malformed-payload rejection tests separate from valid oracle
   fixtures. Review derivation and license against the finally selected revision in WP1.
   Add the §3 pair overload and `PairEncodingOptions`/`PairTruncationStrategy` with BERT pair
   templates/type IDs, padding, all three explicit truncation strategies,
   special masks and per-input offsets, including asymmetric lengths and Unicode. MRPC cannot
   proceed without pair fixtures. Extend `tools/tokenizer-oracle/runner.py` before creating them:
   accept optional `textPair` in `*.input.json`, pass it to Rust `Tokenizer.encode`, and pass
   each case's explicit strategy to `enable_truncation` (the existing keyword forwarding may
   be retained). Record `encoding.sequence_ids` as `sequenceIds` in enhanced expected cases.
   Convert each token's character offsets to UTF-8 bytes against `text` for sequence 0 or
   `textPair` for sequence 1; record null offsets for special/padding tokens with null sequence
   identity. Never convert second-sequence offsets against the first text.
   Update Java `TokenizerOracleTest` input/expected parsing to dispatch pair calls, map strategy
   options, assert sequence identity for supported BERT templates using type IDs plus masks,
   and compare null oracle offsets to the Java synthetic-token offset convention. No production
   sequence-ID API change is implied. Add non-ASCII pairs with different byte lengths, single
   ONLY_SECOND under/over limit, removal equal to a sequence length and equal-length
   LONGEST_FIRST cases. Record expected errors for failure cases, not fabricated encodings.
   Keep legacy single-sequence output schema unchanged; enhanced pair/edge cases opt into
   sequence identity/null-offset output, and Java tests accept both schemas. Because runner.py
   is a Gradle verification input, run `generateTokenizerOracleFixtures` for all existing
   fixtures and review every diff: legacy `*.expected.json` must remain byte-identical, while
   new enhanced cases have explicit reviewed additions. Repin modified input-source hashes in
   provenance and run both oracle verification tasks plus Java tests. Update
   `req/phase6-2-tokenizer-components.md` and the Phase 6.2
   pair deferral with dated amendments, tokenizer README/CHANGELOG and Java 21 bytecode checks.
   Public API changes belong to jmlx-tokenizer's independent next release and version process;
   do not publish or bump versions as part of implementing this milestone without a release task.
3. **Core attention:** land masks, additive-bias attention, scale overloads, encoder block and
   static cross-cache with native ownership and reference tests. Existing decoder checks pass.
4. **Tier-B infrastructure:** implement §8's safe nested downloader, task schema, task-specific
   tests, tuple/precision comparisons and matrix/XML assertions. Required before either family
   claims Tier-B acceptance; verify existing decoder manifest compatibility.
5. **BERT:** land descriptors/loaders, public results, sentence pooling and both task heads;
   WP2 tokenizer and strict/default-mode fixtures pass; run MiniLM and classifier Tier-B evidence
   using WP4. Compare padded rows with the key-only mask and actual sentence-transformers output.
   Measure real-checkpoint logit/vector errors and finalize margin eligibility for MiniLM and
   the classifier; replace candidate inputs and update manifests when needed before acceptance.
6. **T5 forward:** land T5 mappings, norms/FFNs, relative bias and encoder/decoder stacks;
   teacher-forced and cached/full-forward fixtures pass before generation integration.
7. **Generation:** integrate request scopes, sampler/output helpers, specified history and budget
   semantics; scheduler rejection and failure/memory tests pass; run Flan-T5 Tier-B through WP4
   after WP2 proves its pinned tokenizer can load and encode correctly.
   Measure Flan-T5 real-checkpoint per-step errors over reference histories and finalize margin
   eligibility; replace candidate sources and update manifests when needed before acceptance.
8. **Hardening/docs:** required-native CI lists, weekly Tier-B jobs, README/examples, CHANGELOGs
   and AGENTS architecture updates. Extend `req/phase6-compatibility.md` with “text encoder”
   and “seq2seq text” rows distinguishing Tier-A and Tier-B and naming unsupported variants.
   Record measurements and acceptance in a Phase 7.2 report; update parent milestone status.
   Document `downloadTierBArtifact`'s exact manifest/target property names in README, AGENTS and
   tools/tier-b documentation. Ordinary `check`/`build` retain their Java/Gradle prerequisites;
   Python remains confined to explicitly invoked reference/oracle tools. Preserve successful
   unstaged-checkout builds without Python, native bootstrap or oracle environments.

Required checks during implementation:

```sh
./gradlew :jmlx-core:check :jmlx-models:check :jmlx-tokenizer:check
./gradlew spotlessCheck verifyMlxApiCallSites verifyHfReferenceGoldens
./gradlew -p buildSrc check
./gradlew verifyTokenizerOracle verifyTokenizerOracleFixtures
./gradlew verifyMlxOracle verifyMlxOracleFixtures
./gradlew check
```

Inventory freshness is part of root `check`; run `generateMlxApiInventory` only when mappings
change and review its diff. WP4 changes buildSrc, so run its explicit check command above;
the existing Java CI step provides the same verification independently of root `check`. Oracle tools
require their documented installed environments and host profiles. A skipped native suite is not
acceptance: final evidence comes from supported macOS 26 Apple Silicon execution and native CI.

Completion requires every parent §7.2 gate: BERT embeddings/classification and Flan-T5 references,
pinned Tier-B runs, explicit seq2seq semantics with focused tests, static-cache leak evidence,
named scheduler rejection and compatibility rows. Outstanding artifact pins or unavailable native
evidence remain visible blockers to acceptance, not inferred successes.
