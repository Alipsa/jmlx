# Phase 6.2 implementation plan — tokenizer and prompt compatibility

**Roadmap:** `req/full-roadmap.md` §Phase 6.2

**Prerequisite:** Phase 6.1, merged in PR #23 (`92fa802`)

**Branch:** `phase6-2`

## Goal

Turn the current byte-level-BPE helper into the tokenizer side of the public local-text inference
contract. Phase 6.2 must preserve the existing Llama/Qwen byte-level BPE results, add component-based
support for the tokenizer families needed by the Phase 6 matrix, load the tokenizer metadata and
chat template shipped beside a checkpoint, and let `GenerationRequest` produce prompt IDs and
streamed text without double-adding special tokens.

The deliverable remains pure Java at runtime. A pinned Hugging Face reference implementation is a
fixture generator and CI oracle, not a production dependency. No tokenizer or model is downloaded by
an ordinary test.

## Baseline and scope boundary

The merged baseline has these constraints:

- `HfTokenizer` is a final, thread-safe facade, but its loader and fields are hard-wired to NFC +
  ByteLevel pre-tokenization + BPE + ByteLevel decoding.
- `encode(String, boolean)` returns IDs only. The parser rejects added-token stripping,
  single-word, and normalized matching, and it does not represent offsets, masks, truncation, or
  padding.
- `ChatTemplateRenderer` can render a caller-supplied template, but `HfTokenizer` does not load
  `tokenizer_config.json` or standalone `.jinja` templates.
- `GenerationRequest` accepts only token IDs. `GenerationEvent.textDelta()` is always null, and the
  old `DecoderModel.generateText` adapter tokenizes and decodes outside the common generation path.
- The compatibility matrix claims Llama/Qwen byte-level BPE only. Mistral, Gemma, Phi, and Mixtral
  remain Phase 6.3 architecture work.

Phase 6.2 implements three reference tokenizer model families:

1. BPE with either ByteLevel or Metaspace components. Existing ByteLevel behavior remains unchanged;
   Metaspace coverage establishes the SentencePiece-style BPE path used by Mistral/Mixtral artifacts.
2. Unigram with Metaspace and byte fallback, using fixtures whose normalizer pipeline does not
   contain SentencePiece `Precompiled`. This establishes the model/decoder path needed by
   Gemma-style artifacts without pretending to support the serialized SentencePiece charsmap.
3. WordPiece with the BERT normalizer/pre-tokenizer/decoder pipeline. This is a tokenizer capability,
   not a claim that a WordPiece-backed decoder architecture is supported in `jmlx-models`.

Do not add WordLevel, pair-sequence encoding, tokenizer training, Unigram sampling, arbitrary custom
regex dialect emulation, model downloads, or new model architectures. If a committed target fixture
uses a component outside the list below, fail the fixture audit first and amend this plan rather than
silently approximating it.

## Contract decisions

### 1. Keep `HfTokenizer` as the public facade

Do not create model-name subclasses such as `GemmaTokenizer` or `MistralTokenizer`. Introduce
package-private component strategies selected exclusively from `tokenizer.json` declarations:

- `NormalizerStep` transforms a `NormalizedText` while retaining alignment to original input;
- `PreTokenizerStep` returns aligned pre-token spans;
- `TokenizerModel` maps each pre-token to aligned model tokens;
- `DecoderStep` converts token strings/bytes into decoded output;
- the existing post-processor representation inserts special tokens and metadata.

Use sealed interfaces/records for parsed configuration so the loader has an exhaustive switch and
an unknown `type` fails at load time with its JSON path. The BPE strategy owns only BPE vocabulary and
merge behavior; ByteLevel/Metaspace transformations belong to their declared pipeline components.
`HfTokenizer.fromFile(tokenizerJson)` remains source-compatible. Add
`HfTokenizer.fromDirectory(modelDirectory)` for the complete tokenizer configuration and chat
template bundle.

`HfTokenizer` remains immutable and safe to share. Per-encode alignment, dynamic-programming, and
incremental-decode state must be request-local.

### 2. Add a real encoding result without breaking the ID convenience API

Add these public Java-21-compatible value types in `se.alipsa.jmlx.tokenizer`:

- `TokenOffset(int startByte, int endByte)` — half-open offsets into the original UTF-8 input;
- `TokenizerEncoding(List<Integer> ids, List<Integer> typeIds, List<Integer> attentionMask,
  List<Integer> specialTokensMask, List<TokenOffset> offsets)` — immutable, equal-length columns;
- `EncodingOptions(boolean addSpecialTokens, Truncation truncation, Padding padding)`;
- `Truncation(int maxLength, Direction direction)` and
  `Padding(int length, Direction direction, int padId, String padToken, int padTypeId)`.

Use explicit disabled values/factories rather than null public arguments. Validate positive lengths,
known pad IDs/tokens, and mutually consistent options before tokenization. The direction enum is
shared (`LEFT`, `RIGHT`). A top-level non-null `tokenizer.json` truncation/padding block is parsed as
the tokenizer's configured defaults. Add `encodeWithDefaults(String, boolean addSpecialTokens)` to
return a `TokenizerEncoding` using those defaults and the caller's special-token choice; a
one-argument convenience overload uses the HF-compatible default of `true`.
`encode(String, EncodingOptions)` overrides the defaults explicitly. The legacy
`encode(String, boolean)` deliberately uses unbounded/unpadded options and returns only `ids()`, so
all existing callers and goldens retain their behavior. If a configured default cannot be expressed
by the supported single-sequence options, reject it at load time rather than retaining inert config.

Keep parsed tokenizer-JSON truncation/padding config separate from the public value types until
validation completes. The parsed records retain `strategy`, `stride`, `direction`, `max_length`,
`pad_to_multiple_of`, `pad_id`, `pad_token`, and `pad_type_id` as applicable, so pair-only strategy,
non-zero stride, unsupported multiples, and malformed values cannot disappear before the loader
rejects them or converts them into supported `Truncation`/`Padding` defaults.

For the supported single-sequence API, truncation happens after model tokenization but reserves the
post-processor's special-token budget; post-processing follows; padding is last. Special and padding
tokens use offset `(0,0)`, special mask `1`, padding attention mask `0`, and the declared type ID.
Reject pair-only strategies, non-zero overflow stride, and inconsistent configured lengths rather
than returning partial encodings.

Track alignment through normalization instead of reconstructing offsets after tokenization.
`NormalizedText` stores Unicode-scalar boundaries mapped to original UTF-8 byte boundaries; every
normalizer composes that map. The Java API independently defines offsets as half-open UTF-8 byte
ranges into the original input, never Java UTF-16 indices. Slice 1 must audit the pinned Python
binding's actual unit and ByteLevel behavior on ASCII, combining, and supplementary inputs; if it
surfaces another unit, convert the oracle output explicitly before comparison rather than changing
the Java contract. Added-token consumed whitespace follows that converted reference result. Tests
must include combining characters, supplementary code points, invalid-looking byte fallback
sequences, left/right truncation, left/right padding, and special-token offsets.

### 3. Implement declared components, not look-alike behavior

Support only the following shapes in this milestone, each selected from JSON and independently
golden-tested:

- normalizers: null, NFC/NFD/NFKC/NFKD, Lowercase, StripAccents, Replace, Strip, Prepend,
  BertNormalizer, and Sequence composition. `Precompiled` is deliberately unsupported in 6.2:
  reject it with its JSON path rather than approximating the SentencePiece DoubleArray charsmap;
- pre-tokenizers: ByteLevel including `add_prefix_space` and `trim_offsets`; Metaspace including
  `replacement`, `prepend_scheme`, and `split`; Whitespace/WhitespaceSplit, BertPreTokenizer, Split,
  and Sequence composition;
- models: BPE, Unigram, and WordPiece;
- post-processors: ByteLevel, TemplateProcessing, BertProcessing, and Sequence composition;
- decoders: ByteLevel, Metaspace, WordPiece, Replace, Strip, ByteFallback, Fuse, and Sequence.

Parsing a recognized component includes every field that affects IDs, text, or offsets. Reject an
unsupported field value with its component and field name; do not accept a config merely because its
top-level type is known.

Expand `AddedToken` to retain `single_word`, `lstrip`, `rstrip`, `normalized`, and `special`.
Implement matching in the same order as the HF pipeline: normalized and non-normalized added-token
tries around normal text, longest match followed by declaration order, word-boundary enforcement,
and whitespace consumption. Remove the current loader rejections only when each behavior has an
isolated reference golden.

This widens the public `AddedToken` record and is intentionally source-breaking while
`jmlx-tokenizer` remains unpublished at `0.1.0-SNAPSHOT`; call it out in the API compatibility review
rather than treating extra record components as source compatible.

Split model parsing by declared model type before constructing shared vocabulary views. BPE alone
parses `vocab` as token-to-ID and requires `merges`; Unigram parses its ordered
`[[token, score], ...]` array so list position is the ID and retains scores; WordPiece parses its own
token-to-ID object without BPE-only validation. Do not apply BPE's `merges` or inert-field checks to
Unigram/WordPiece. Generalize `Vocabulary` with an id-indexed construction path while retaining its
public token/ID lookup contract; Unigram scores remain owned by the model strategy rather than
changing vocabulary equality.

For BPE, honor `unk_token`, `fuse_unk`, `byte_fallback`, `continuing_subword_prefix`,
`end_of_word_suffix`, and `ignore_merges`, with isolated fixtures for every value that removes a
current loader rejection. Because runtime tokenization must be deterministic, accept `dropout` only
when absent, null, or numeric zero and reject every other value. Validate that declared unknown and
byte-fallback tokens resolve to vocabulary entries.

For Unigram, run Viterbi over Unicode-scalar boundaries, maximizing the sum of serialized token
scores with deterministic HF-compatible tie-breaking. Use a vocabulary trie, with the longest
serialized piece as an explicit bound, so each input position considers only real vocabulary
prefixes rather than every substring. Add an adversarial long-input scaling test to prevent a
quadratic implementation. Honor `unk_id` and `byte_fallback`; fuse unknown spans exactly as the
fixture establishes. For WordPiece, count Unicode scalars for
`max_input_chars_per_word`, use greedy longest-prefix matching and the declared continuation prefix,
and emit the configured unknown token for the entire word when any suffix cannot be segmented.

Byte fallback parses only canonical `<0xHH>` vocabulary tokens and decodes buffered bytes with a
stateful UTF-8 decoder. Malformed byte-token declarations fail at load time; incomplete final byte
sequences flush with the pinned replacement behavior. Cover both cases with focused tests and oracle
goldens.

### 4. Load tokenizer metadata and chat templates with explicit precedence

`HfTokenizer.fromDirectory` reads `tokenizer.json` plus optional `tokenizer_config.json` and template
files. Add an immutable `TokenizerMetadata` exposing configured BOS, EOS, pad, unknown, separator,
classification, and mask tokens; model maximum length when meaningful; padding/truncation side; and
available chat-template names. Special-token config values may be strings or HF `{content: ...}`
objects; all configured tokens must resolve to the loaded vocabulary.

Follow Hugging Face's documented template loading precedence:

1. legacy `tokenizer_config.json.chat_template` string or named-template array;
2. root `chat_template.jinja` overriding the `default` entry;
3. `additional_chat_templates/*.jinja`, keyed by filename stem.

Reject duplicate/ambiguous names and path traversal. When more than one template exists, require an
explicit name unless `default` is present. Do not silently choose `tool_use`; callers pass the name
and tool context explicitly.

Add `ChatTemplateOptions(String templateName, boolean addGenerationPrompt,
Map<String,Object> extraContext)` with immutable context copying, plus
`HfTokenizer.renderChat(List<Map<String,Object>> messages, ChatTemplateOptions options)`. Require
non-empty roles and textual content for Phase 6.2; multimodal content blocks and
`continue_final_message` remain out of scope. Rendering exposes the configured special-token values
to Jinja, then tokenizes the rendered prompt with special-token insertion disabled. This enforces
the documented rule that a chat template already owns its BOS/EOS markers and prevents duplication.

The reserved render context contains `messages`, `add_generation_prompt`, and every configured
standard token under its HF key: `bos_token`, `eos_token`, `pad_token`, `unk_token`, `sep_token`,
`cls_token`, and `mask_token`. Omit an unconfigured optional token rather than binding it to an
invented value. Caller `extraContext` supplies `tools` and other template-specific data.
`ChatTemplateOptions` rejects any reserved key in `extraContext` at construction with the key named;
do not silently discard an attempted override. Generalize
`ChatTemplateRenderer` to assemble context from a token/context map instead of only two token
parameters, while retaining its existing public overload as a source-compatible delegate.

The relevant upstream contracts are the Hugging Face tokenizer pipeline/component documentation and
the Transformers chat-template loading/`add_generation_prompt` documentation:

- <https://huggingface.co/docs/tokenizers/main/api/tokenizer>
- <https://github.com/huggingface/tokenizers/blob/main/docs/source-doc-builder/components.mdx>
- <https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/tokenizer/mod.rs>
- <https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs>
- <https://huggingface.co/docs/transformers/chat_templating>
- <https://huggingface.co/docs/transformers/en/chat_templating_writing>

### 5. Compose tokenization into `GenerationRequest`

Do not make `TextGenerationModels.load` own or implicitly discover a tokenizer. Model tensors remain
owned by the caller's `MLXScope`, while an immutable tokenizer can be shared independently. A request
that wants text behavior carries the tokenizer it used; an existing token-ID request does not.

Add `PromptSpecialTokens` with `ADD`, `OMIT`, and `PRETOKENIZED`:

- the existing `GenerationRequest(int[], ...)` constructor remains and records `PRETOKENIZED`;
- `GenerationRequest.text(HfTokenizer, String, PromptSpecialTokens, GenerationConfig,
  CancellationToken)` accepts only `ADD` or `OMIT`, tokenizes immediately, and stores the tokenizer
  for output decoding;
- `GenerationRequest.chat(HfTokenizer, List<Map<String,Object>>, ChatTemplateOptions,
  GenerationConfig, CancellationToken)` renders and tokenizes with `OMIT` by construction.

Expose defensive prompt IDs and the recorded policy, but not mutable encode/decode state. Empty text
is allowed only when its post-processor/template produces at least one prompt ID; the existing model
request validation still rejects a truly empty prompt. Before opening a native prefill scope,
`DecoderModel` validates all produced IDs. For tokenizer-backed requests only, it additionally
requires `tokenizer.vocabSize() <= config.vocabSize()`; pretokenized requests have no tokenizer to
check. This permits the documented checkpoint-output-head tail gap but rejects a tokenizer whose
known ID range exceeds the checkpoint. It is a bounds-compatibility check, not proof that two
same-sized vocabularies are semantically identical.

The text and chat factories are eager pure-Java operations. Tokenization errors therefore occur at
request construction, not after a model generation scope has opened.

### 6. Stream decoded text without corrupting split Unicode

Add a request-local `IncrementalTokenDecoder` created by `HfTokenizer.newIncrementalDecoder(boolean
skipSpecialTokens)`. It accepts one generated ID at a time and returns the stable text delta now safe
to emit; `finish()` flushes any buffered incomplete byte sequence using the same replacement behavior
as full decode. `DecoderModel.generate` always passes `true` for tokenizer-backed requests, matching
legacy `generateText` and the full-decode acceptance gate; Phase 6.2 does not add a request-level
skip-special-tokens option. The decoder never owns native state.

For tokenizer-backed requests:

- accepted generated tokens carry a non-null `textDelta` (possibly empty while bytes are buffered);
- explicit stop tokens remain excluded and are never passed to the decoder;
- EOS remains an emitted token and is decoded or skipped according to the tokenizer vocabulary's
  own special-token predicate; do not assume that a configured EOS is flagged special;
- the terminal event may carry a final non-null delta from `finish()`, but still has no token ID or
  log probability;
- the incremental decoder applies full `decode`'s predicates in the same order: skip an ID when
  special-token skipping is requested and the vocabulary marks it special; otherwise decode it when
  `vocabulary.hasId(id)` (including a TemplateProcessing-only registration above the base maximum);
  otherwise drop it when it is above the tokenizer's known base-vocabulary maximum; otherwise throw
  for an in-range vocabulary hole;
- concatenating every non-null token and terminal delta equals
  `tokenizer.decode(result.generatedTokenIds(), true)` for EOS, stop, max-token, and cancellation
  completion whenever decoding succeeds.

If incremental decoding fails after native generation has started, `DecoderModel.generate` aborts
before delivering that token to the listener, closes the active/generation scopes, and throws
`GenerationAbortedException` with the original `TokenizerException` as cause plus prompt length,
generated-token count, failing token ID, and decode-stage context. The selected failing ID remains in
the exception's generated-token list, but no event for it and no terminal event are sent. Generalize
`GenerationAbortedException`'s listener-specific javadoc/message and internal construction to cover
both listener and output-decoder abort stages without weakening its existing prompt/generated-ID
evidence. Pure-Java request-construction failures remain direct tokenizer/configuration exceptions;
the statement that tokenization errors happen at request construction applies only to prompt
encoding, not output decoding.

Relax `GenerationEvent` only enough to permit terminal flush text. Add factories for token events
with text and optional log probability and for terminal events with trailing text. Token-ID requests
retain null deltas byte-for-byte. Listener failure semantics do not change: a token-listener failure
aborts immediately and sends no terminal flush event.

On successful termination, call `IncrementalTokenDecoder.finish()` before constructing either
terminal artifact. Append that flush text to the accumulated generated text, construct
`GenerationResult`, then invoke the listener with a terminal event carrying the exact same flush
delta. Tests assert both that concatenated token/terminal event deltas equal full decode and that
`GenerationResult.generatedText()` equals that same concatenation, including when the only final
text comes from `finish()`.

If cancellation is already set at the pre-loop check, no incremental decoder exists and `finish()`
is not called. A tokenizer-backed request still returns `generatedText == ""` and emits a terminal
event whose `textDelta` is the same non-null empty string; a pretokenized request retains null for
both fields. Pin this never-started path in the event/result contract tests.

Add nullable `generatedText` to `GenerationResult`; it is null for pretokenized requests and non-null
(including empty) for tokenizer-backed requests. Keep a four-argument compatibility constructor for
existing source callers, while documenting that the added record component necessarily changes
record `equals`, `hashCode`, and `toString`. The record validates only its own cardinality/nullability
invariants; `DecoderModel.generate` validates that result text is present exactly when the request
path collected deltas. `DecoderModel.generateText` becomes a thin delegate to
`GenerationRequest.text` and `GenerationResult.generatedText`, with no separate encode/decode logic.

Incremental decoders must not call full `decode(allIds)` once per token. ByteLevel and byte-fallback
paths stream bytes through a `CharsetDecoder`; Metaspace and WordPiece steps retain only the minimal
pending prefix/cleanup state. A long generated stream test guards against quadratic accumulation.
Add a randomized split-equivalence test that feeds valid and malformed byte streams through every
possible incremental split and compares the collected output with one-shot
`new String(bytes, UTF_8)`, pinning replacement-character behavior rather than assuming the JDK's
incremental and one-shot paths coincide.

## Reference fixtures and oracle

Create `tools/tokenizer-oracle/` with a lockfile pinning the official Hugging Face tokenizer runtime
used to create/verify fixtures: the official Python `tokenizers` package backed by the Hugging Face
Rust implementation. Pin Python, the exact wheel version and hashes in
`tools/tokenizer-oracle/requirements.lock`, and record those values plus fixture-source hashes in
provenance. The lock covers the complete transitive dependency set and all selected wheels for both
Ubuntu `manylinux_x86_64` and macOS `arm64`; installation uses `--require-hashes` and
`--only-binary=:all:`. Provide install and environment-verification scripts; CI provisions the pinned
Python before installing the hash-locked oracle, and verification asserts the resolved OS,
architecture, Python version, and package versions. Fixture generation consumes only committed local
files and runs with network access disabled after installation; installing `huggingface_hub` as a
transitive dependency does not authorize downloads. Do not use a JavaScript port as the reference:
its offset and component coverage is not equivalent to the Rust implementation exercised by the
Python binding.

The public UTF-8 byte-offset contract is a jmlx API choice. Slice 1 records what the pinned Python
binding actually returns, including `ByteLevel.trim_offsets` and `add_prefix_space`; the oracle
adapter converts those values into jmlx's original-input UTF-8 byte ranges when necessary. Treat any
reference-version change that alters raw or converted offsets or component behavior as an explicit
repin with a reviewable fixture diff.

Use the same paired `*.input.json`/`*.expected.json`, generate-only vs read-only-verify discipline as
the MLX oracle. Generation is the only task allowed to rewrite expected output. Verification declares
all inputs/outputs so adding, removing, or renaming a fixture invalidates Gradle up-to-date state.
Stale output uses the compact pretty-printed unified-diff diagnostic established in Phase 6.1.

Commit small, license-compatible tokenizer JSON/config/template fixtures and provenance for:

- existing Qwen2.5 ByteLevel-BPE text and ChatML prompts;
- existing Llama-3-style ByteLevel-BPE text and instruct prompts;
- Metaspace-BPE with byte fallback and unknown input;
- Unigram/Metaspace with competing segmentations, tie behavior, unknown and byte fallback, selected
  only from configurations with no `Precompiled` normalizer;
- WordPiece/Bert with case folding, accents, Chinese boundaries, continuation pieces, punctuation,
  unknown input, and cleanup;
- added-token flags, non-ASCII offsets, truncation, padding, and post-processor metadata;
- incremental decoding where one Unicode scalar spans multiple generated tokens, plus a generated
  TemplateProcessing-only ID that is known even though it lies above the base-vocabulary maximum.

Each expected case records tokens, IDs, decoded text, offsets, type IDs, attention/special masks,
rendered chat text where applicable, and incremental deltas. Do not commit full new model weights or
credential-gated artifacts. Existing real Qwen tokenizer/template files remain Tier A; every external
source/revision/hash is recorded in the fixture manifest.

## Delivery slices

### Slice 1 — Evidence, schema audit, and strategy boundary

1. At the start of the slice, update Ubuntu CI's `actions/setup-java` configuration to install both
   Java 21 and 25 explicitly. Bridge the hosted-tool-cache variables into Gradle discovery with
   `org.gradle.java.installations.fromEnv=JAVA_HOME_21_X64,JAVA_HOME_25_X64`, set
   `org.gradle.java.installations.auto-download=false` only through the CI job environment or command
   line, never in committed `gradle.properties`, and then run
   `:jmlx-models:check`. First prove with `--info` that both configured installations are discovered,
   no toolchain is provisioned, and the command succeeds on a Linux checkout with no
   `native/install/lib`; native-gated tests must skip while all pure-Java request/event/result tests
   execute. Keep foojay available for developer machines, but do not rely on it in CI.
2. Pin the tokenizer oracle, add generation/verification tasks, provision its pinned Python on
   Ubuntu, and run its environment/fixture verification in that job before production changes.
3. Commit the component matrix extracted from every selected fixture. Resolve field semantics and
   reference tie/offset behavior before implementing algorithms. Record the Python binding's raw
   offset unit and ByteLevel behavior and the oracle adapter's conversion to jmlx UTF-8 byte ranges.
   Reject any selected Unigram fixture containing `Precompiled` during this audit.
4. Introduce parsed component interfaces/config records and refactor the existing ByteLevel-BPE path
   behind them.
5. Prove all existing Qwen/Llama IDs, decode text, errors, and thread-safety tests are unchanged.

This slice is accepted only if unknown model/component types fail specifically and the old fixtures
remain byte-identical.

### Slice 2 — General aligned pipeline and added tokens

1. Add `NormalizedText`, aligned pre-token/model-token values, and composed normalizer/pre-tokenizer
   sequences.
2. Implement all added-token flags and preserve alignment through them.
3. Add `TokenizerEncoding`, explicit options, post-processor metadata, truncation, and padding.
4. Pin every result column against oracle fixtures, including non-ASCII offsets.

Do not proceed while offsets are reconstructed heuristically or any accepted JSON field is ignored.

### Slice 3 — Metaspace BPE and Unigram

1. Split tokenizer model parsing by type, add id-indexed/scored Unigram vocabulary loading, and keep
   BPE-only merges/fields out of the Unigram and WordPiece paths.
2. Generalize BPE input symbols away from ByteLevel-only coding, implement Metaspace components, and
   honor the enumerated deterministic BPE unknown/fallback/prefix/suffix/ignore-merges fields.
3. Implement trie-bounded Unigram Viterbi, unknown handling, byte fallback, decoder sequences, and
   incremental byte decoding.
4. Add isolated algorithm, adversarial scaling, and end-to-end oracle cases.
5. Keep architecture matrix rows planned; this slice proves tokenizer capability only.

### Slice 4 — WordPiece/Bert pipeline

1. Implement BertNormalizer/BertPreTokenizer, WordPiece model segmentation, BertProcessing, and the
   WordPiece decoder.
2. Cover Unicode, punctuation, continuation, unknown, maximum-word-length, offsets, special tokens,
   truncation, and padding against the oracle.
3. Document WordPiece as available in `jmlx-tokenizer`, without claiming a compatible Phase 6 model.

### Slice 5 — Tokenizer config and chat prompts

1. Add `fromDirectory`, `TokenizerMetadata`, special-token parsing, and template precedence.
2. Generalize `ChatTemplateRenderer` to map-based context assembly, preserve its existing overload,
   reject reserved-key collisions, and expose the exact standard-token/extra-context set above.
3. Add named-template selection and `renderChat`.
4. Match the committed Qwen/Llama rendered-text and encoded-ID goldens for plain, system, multi-turn,
   and generation-prompt conversations.
5. Add negative tests for missing templates, ambiguous names, unknown configured tokens, duplicate
   special tokens, unsupported multimodal messages, and attempted double special-token insertion.

### Slice 6 — Generation request and text streaming

1. Add explicit prompt-special-token policy and text/chat request factories.
2. Wire incremental decoding into the common `DecoderModel.generate` path.
3. Extend events/results and generalize `GenerationAbortedException` with the scoped text/error
   contracts above; make legacy `generateText` delegate.
4. Add pure-Java contract tests, including above-range IDs, in-range holes, every byte split, and
   decode-abort context, plus native tiny-model integration tests for Llama and Qwen.

No architecture-specific tokenizer branch may appear in `DecoderModel` or
`TextGenerationModels`.

### Slice 7 — Documentation, matrix, and release checks

1. Update `jmlx-tokenizer/README.md`, `jmlx-models/README.md`, public Javadocs, and examples with raw
   text, rendered chat, explicit special-token policy, streaming deltas, offsets, and padding.
2. Update `req/phase6-compatibility.md` and `req/phase6-tier-a-fixtures.md` with exact evidence. Do
   not upgrade an architecture to real-artifact verification based only on a tokenizer fixture.
3. Update `jmlx-tokenizer` publication metadata from “byte-level BPE” to the exact supported family
   list and keep Java 21 bytecode/publication dependency checks green.
4. Audit that the Slice-1 Ubuntu JDK 21/25, `:jmlx-models:check`, pinned-Python tokenizer oracle, and
   existing Node/Jinja gates still cover the final task graph. Native model-text integration remains
   in the macOS job.

## Verification matrix

Run on every platform-independent change:

```text
./gradlew -p buildSrc check
./gradlew :check
./gradlew :jmlx-jinja:check
./gradlew :jmlx-tokenizer:check
./gradlew :jmlx-models:check
./gradlew verifyTokenizerOracle verifyTokenizerOracleFixtures
```

Run the existing pinned Node/Jinja verification tasks when chat-template behavior or CI wiring
changes. Run Spotless, Checkstyle, Javadocs, publication metadata/dependency checks, Java-21 bytecode
verification, and `git diff --check` before every slice is committed.

On macOS Apple Silicon after native bootstrap, additionally run:

```text
./gradlew --no-build-cache :jmlx-models:check
```

The native suite must prove text/chat requests for both tiny decoder families preserve token IDs,
emit correct deltas and terminal flushes, retain log-probability alignment, handle EOS/stop/max-token/
cancellation, and close all generation resources. Tokenizer algorithm and template tests stay pure
Java and run on Ubuntu.

## Acceptance gate

Accept Phase 6.2 only when:

- all pre-6.2 ByteLevel-BPE goldens remain unchanged;
- BPE/ByteLevel, BPE/Metaspace, Unigram/Metaspace, and WordPiece/Bert fixtures match the pinned HF
  oracle for IDs, tokens, text, offsets, masks, truncation, padding, and relevant special tokens;
- Qwen and Llama tokenizer directories load their configured special tokens and documented chat
  templates, and rendered prompt text/IDs match committed references;
- raw-text/chat requests state their special-token policy and cannot add template-owned tokens twice;
- incremental decoding matches full decode's ordered special-skip, known-ID decode,
  unknown-above-base-range drop, and in-range-hole error behavior; concatenated deltas and
  `GenerationResult.generatedText()` both equal successful full generated-ID decode for every
  terminal path, including split UTF-8 and cancellation flush;
- output-decode failures are contextual `GenerationAbortedException`s, release native resources,
  and emit neither the failing token nor a terminal event;
- existing pretokenized requests retain null text fields and unchanged generated IDs;
- malformed or unsupported tokenizer components fail at load/request construction with the JSON
  component/field named;
- runtime tokenization is pure Java, credential-free, uses trie-bounded Unigram matching and
  linear-state incremental decoding, and introduces no native resource ownership;
- compatibility and Tier-A documents claim only the evidence actually exercised; and
- Ubuntu Java/oracle CI and macOS native integration CI are green.

## Explicitly deferred

- pair/batch tokenization, overflow encodings with stride, training, vocabulary mutation, and Unigram
  sampling;
- WordLevel and arbitrary normalizer/pre-tokenizer/decoder plugins not present in selected fixtures;
- SentencePiece `Precompiled` normalizer/charsmap decoding; selected 6.2 Unigram fixtures must not
  contain it;
- multimodal chat content, `continue_final_message`, automatic `tool_use` template selection, and
  model-specific prompt heuristics;
- tokenizer/model artifact downloading, caching, authentication, and license acceptance;
- new decoder architectures, RoPE scaling, sliding-window attention, and checkpoint changes (6.3);
- real Mistral, Mixtral, Gemma, and Phi tokenizer-directory/chat-template goldens. This deliberately
  narrows the roadmap's broad “every Phase-6 model artifact” tokenizer exit gate to Qwen/Llama in
  6.2; the remaining architecture artifacts land with their 6.3 loaders rather than being claimed
  from synthetic tokenizer fixtures;
- cache/performance work (6.4), batched serving/back-pressure (6.5), and asynchronous publishers.
