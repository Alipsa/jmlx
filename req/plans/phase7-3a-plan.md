# Phase 7.3a — Vision foundation and SmolVLM-256M

- **Status:** WP1–WP2 delivered via PR #41 (merge `6e82590`), WP3 via PR #42 (merge `0dac174`),
  WP4 via PR #43 (merge `d658309`), WP5 via PR #44 (pending merge as of 2026-10-09); WP6 is the
  next work package.
- **Parent:** `req/plans/phase7-plan.md` §7.3a, public decisions 1–4, 6–8 and
  Cross-cutting verification.
- **Prerequisite:** 7.1 layer/oracle/inventory acceptance. 7.2 cross-attention and 7.0 are
  not dependencies. Reuse the delivered encoder modules where their semantics fit.

## 1. Scope and outcome

Deliver a published, pure-Java `jmlx-vision` module and local SmolVLM-256M image+text generation
through `TextGenerationModel`. Support PNG/JPEG input, RGB conversion, bilinear/bicubic/LANCZOS
resize, rescale/normalize, processor-driven image splitting, structured chat content,
placeholder expansion and the Idefics3 vision tower/connector. Return ordinary Java results and
preserve sampling, streaming, cancellation, stop and log-probability semantics.

The target is `HuggingFaceTB/SmolVLM-256M-Instruct`. Its current [model
card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) identifies an Apache-2.0,
Idefics3-based model with a SmolLM2 backbone. This is candidate discovery, not a revision pin or
proof that the local loader supports it. WP1 pins the exact artifact and resolves its actual
`model_type`, processor class, configurations and tensors before implementation claims.

Exclude Qwen3-VL execution, LLaVA, video, WebP, remote image fetching, image generation,
training, public model batching and multimodal scheduler cohorts. Start with float checkpoint
loading; reject affine/other quantized vision checkpoints and unsupported per-layer overrides
explicitly. No native pin change or generated-binding edit is planned. Qwen smart-resize,
multi-axis RoPE and DeepStack remain 7.3b work; provide reusable image primitives without
implementing those algorithms.

## 2. Current integration points

- `DecoderModel.normalizedHiddenStatesBatch` currently performs embedding lookup, optional
  `EmbeddingHook`, embedding scaling, mask/RoPE construction, blocks and final normalization.
  `stepLogits` performs cache preflight/postflight, last-position selection and output projection.
  Preserve these invariants and cache poisoning when factoring an embedding entry point.
- `GenerationRequest` eagerly encodes text/chat and owns its token buffer. Add images by copying
  the request, preserving every existing field in both `withImages` and `withCachePolicy`.
- `GenerationResult` is a five-component record with a four-argument compatibility constructor.
  Adding position accounting must retain both signatures and document record equality and
  deconstruction compatibility changes.
- `ModelMetadata` is sealed over decoder, encoder and seq2seq metadata. Extend its permits list
  for vision-language metadata and give all existing implementations TEXT-only defaults.
- `HfTokenizer` rejects non-string message content. `ChatTemplateRenderer` already passes maps
  into Jinja. Extend validation without making the tokenizer responsible for image preprocessing.
- `settings.gradle` has eight projects; six publish. Release smoke explicitly enumerates them.
  Add the seventh published artifact throughout the build and consumer verification.

## 3. Reference and artifact gate (WP1)

Use pinned Transformers 4.57.6, CPU float32, evaluation mode, and explicitly recorded attention
implementation in `tools/hf-reference`. Inspect both [Idefics3
modeling](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/idefics3/modeling_idefics3.py)
and [SmolVLM
processing](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/smolvlm/processing_smolvlm.py).
Inspect the SmolVLM modeling/image-processing classes too: dispatch and processor selection must
come from the pinned files, not the model's marketing name. The current, unpinned
[config](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/config.json) and
[processor
config](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/preprocessor_config.json)
seed WP1 with the following facts; reverify them at the selected immutable revision:

| Area | Current artifact values |
| --- | --- |
| Outer | `idefics3`, `Idefics3ForConditionalGeneration`, scale factor 4, image ID 49190, BF16 storage |
| Text | `llama`, hidden 576, 30 layers, 9 query/3 KV heads, 8192 positions, untied embeddings |
| Vision | hidden 768, patch 16, image 512, 12 layers, `gelu_pytorch_tanh` → `GELU_TANH` |
| Processor | `Idefics3ImageProcessor`, resample 1 (LANCZOS), longest edge 2048, tile longest edge 512 |

Both Python tools explicitly select the slow PIL processor (`use_fast=False`, or direct
`Idefics3ImageProcessor` construction). Assert the resolved class and record it in provenance;
never rely on AutoImageProcessor defaults or substitute the Torch/torchvision fast variant.

Pin the reference loading path to raw `PIL.Image.open` → `load()` → slow processor, retaining
original mode, palette and image metadata until processor entry. Do not call
`transformers.image_utils.load_image`, `exif_transpose` or preliminary `convert("RGB")`. Record
that path in both tools' provenance and use it for Tier-B and chat/image references too. EXIF
orientation therefore remains unapplied. Probe alpha, LA and palette/tRNS through this exact
path; processor palette reconstruction may discard transparency, so the fixture determines the
actual tRNS result rather than assuming white compositing. The Java decoder's RGB conversion
must match this path's observed processor conversion before resize.

Produce `req/plans/phase7-3a-reference-findings.md` before WP4/WP5, containing:

1. Full 40-character HF revision, license files, every required file's SHA-256, download size,
   shard index and tensor inventory. Include tokenizer JSON/config, special-token mapping,
   generation config, processor config and template location (including processor-owned templates).
2. A config-key table for top-level, nested text/vision, processor and generation configs. Mark
   every key supported, explicitly inert or rejected; no unknown computational field merely warns.
   Verify tied aliases and required/optional/forbidden tensors, biases and shapes.
3. Decode/color/alpha/EXIF behavior; resize rounding and antialiasing; intermediate uint8 rounding;
   splitting thresholds, tile/global ordering, padding and pixel masks; token count per tile.
4. Patch convolution and learned-position bucket assignment for valid rectangular patch grids;
   padding-mask reduction; attention scale/bias, pre/post normalization, final norm and activation.
   Idefics3's variable-resolution positions must not be replaced by fixed square indices.
5. Connector pixel-shuffle permutation, scale-factor divisibility, linear dimensions and image
   feature order. Record tiny index-coded examples to expose permutations that shape checks miss.
6. Exact unexpanded template syntax, processor-only tokens, expanded IDs, row/column numbering,
   separators/newlines, and text-only/adjacent/multiple-image behavior. Special IDs are resolved
   through tokenizer metadata and validated; never copied from an example artifact.
7. Actual expanded `input_ids` passed to HF repetition processors; prefill/decode image-feature
   reuse, cache positions, EOS/stop behavior and whether text-only execution skips the tower.

Pin Python/Pillow/NumPy/Transformers and all transitive dependencies with hashes. Verify
existing 4.57.6 compatibility; any bump is a separate reviewed provenance/golden change. The
existing HF lock already pins Pillow 12.3.0 and NumPy 2.5.3. Make Pillow a direct
`requirements.in` pin at that version during WP1; preserve the reviewed lock. In WP2, once the
image-oracle lock exists, add `verifyImageOracleDependencyPins` to root check: parse both locks
and fail on any Pillow, NumPy or Transformers version disagreement, missing pin or inconsistent
environment marker. Both generators consume the same hash-pinned pixel-tensor artifact for
model-math goldens and record its hash; matching dependency versions alone do not prove
identical pixels. Probe tiny CPU examples rather than relying only on source inspection. No
Torch installation or execution becomes an ordinary Gradle check.

WP1 cannot close until this table is populated. If the artifact differs from the anticipated
Idefics3 mapping, amend this plan with the evidence before architecture code lands.

## 4. Vision module and preprocessing contract (WP2)

Add `jmlx-vision`, package `se.alipsa.jmlx.vision`, Java 21, independent of tokenizer/core/FFI.
Initial version `0.1.0-SNAPSHOT`. Apply existing publishing, formatting, lint, Javadoc, license,
signing and bytecode verification conventions; copy the common `release.sh` byte-for-byte.
`jmlx-models` declares `api project(':jmlx-vision')`. In this same PR, update settings,
release-smoke `run.sh` module publication list and `repositories.gradle` jmlxVersions, all
consumer artifact/license/POM checks, and `tools/release-smoke/app/build.gradle` source/Javadoc
artifact counts (six → seven), `verifyReleaseScriptsMatch` coverage, Ubuntu
`:jmlx-vision:check`, and AGENTS.md module, architecture and release sections. Run release-smoke
ci before merging; WP8 must not be the first point at which the new transitive dependency can
resolve.

Proposed public surface:

```java
final class RgbImage {
  RgbImage(int width, int height, byte[] pixels); // interleaved unsigned RGB, row-major
  int width();
  int height();
  byte[] pixels();
  void copyPixelsTo(byte[] destination, int offset);
}
final class ImageDecoder {
  // No-options overloads use ImageDecodeLimits(16_777_216L, 16_384).
  static RgbImage decode(Path path) throws IOException;
  static RgbImage decode(InputStream input) throws IOException;
  static RgbImage decode(Path path, ImageDecodeLimits limits) throws IOException;
  static RgbImage decode(InputStream input, ImageDecodeLimits limits) throws IOException;
}
record ImageDecodeLimits(long maxPixels, int maxDimension) {}
record ImageTensor(float[] values, int[] shape, Layout layout) {
  void copyValuesTo(float[] destination, int offset);
  // Explicit defensive accessors and content equals/hashCode/toString.
}
enum Layout { HWC, NHWC }
final class ImageTransforms {
  static RgbImage resize(RgbImage image, int width, int height, Resampling method);
  static ImageTensor normalize(RgbImage image, float scale, float[] mean, float[] std);
}
enum Resampling { BILINEAR, BICUBIC, LANCZOS }
```

`RgbImage` is a final class with content equality/hashCode. Validate positive dimensions,
`Math.multiplyExact(width, height)` and multiplication by three, and exact buffer length; copy
on construction/access. The bulk-copy method validates destination bounds and exposes no backing
buffer. `ImageTensor` also copies values and shape on construction/access; enforce positive,
overflow-checked dimensions, channel count three, valid layout and exact element count. This
retains the parent's record surface without leaking its array components. Override
equals/hashCode with array-content semantics and toString with a bounded
shape/layout/element-count summary; never print the entire tensor. `copyValuesTo` checks
destination bounds and provides cross-module bulk access without allocating a new multi-megabyte
buffer on every accessor call. Test content equality, hashes, defensive copies and bulk-copy
bounds. Validate finite scale, three finite means and three finite positive standard deviations.

Decode using `javax.imageio` with the JDK's built-in PNG/JPEG reader providers selected through
`IIORegistry.getDefaultInstance()` and ImageReaderSpi format/capability checks; require
`spi.getClass().getModule() == ImageIO.class.getModule()` (java.desktop). Do not reference
non-exported `com.sun.imageio` classes or vendor-name strings. Reject a missing built-in
provider rather than falling back to an installed plugin. Use `MemoryCacheImageInputStream` for
both path and stream adapters, dispose readers and close internal wrappers without closing the
caller's stream. Do not change global ImageIO cache settings. Unsupported/corrupt input throws a
descriptive `IOException`. `ImageDecodeLimits` requires positive fields; default to 16,777,216
pixels and maximum dimension 16,384. Both no-options overloads delegate to those defaults. Read
metadata and use checked size arithmetic before decoding/allocating pixels; explicit limits
overloads permit larger inputs. The Javadoc must state that these conservative defaults reject
24 MP/48 MP phone photos and that callers can use the explicit limits overloads to accept them
with adequate memory. Model-specific aggregate pixel/tile/token budgets remain separate.

Support 8-bit RGB, grayscale, RGBA, grayscale-with-alpha and palette PNG with tRNS; fixtures
verify conversion through the pinned raw-PIL loading path, including whether palette rebuilding
retains or discards tRNS. Reject 16-bit PNG and CMYK/YCCK JPEG with format-specific messages
before conversion. Accept EXIF-rotated JPEG but leave orientation unapplied, matching the
raw-PIL path; include a non-symmetric fixture proving this policy. ImageDecoder Javadoc and the
vision README must explicitly state that EXIF orientation and ICC/gAMA color management are
ignored and that callers must rotate images themselves when needed, including phone photos.
Exposing parsed EXIF orientation is deferred; this milestone promises no orientation accessor.
Extract decoded samples from Raster without ColorModel/getRGB color conversion; embedded
ICC/gAMA information is ignored, matching the reference path without ImageCms. Include Display
P3/ICC and gAMA fixtures to detect reader-side conversion; verify actual JDK reader behavior in
WP1. If a reader already transforms samples, use its raw-raster path and explicit format channel
conversion (for example JPEG YCbCr to RGB), not ICC conversion. Failure to obtain matching
samples is an unresolved acceptance gate, not permission to reject commonplace profiled images
or silently use sRGB conversion. PNG decode/color output targets exact bytes. JPEG JDK/IJG
versus Pillow/libjpeg-turbo output requires measured maximum/mean channel error on a committed
RGB JPEG corpus; no assumed exactness or invented tolerance. Publish an explicit measured JPEG
bound in reference findings before WP2 acceptance, then measure its effect through the complete
resize chain independently. Run headless on Ubuntu and macOS and record reader/JDK versions.

Keep model-specific `SmolVlmImageProcessor` and its validated config in vision. It parses
`preprocessor_config.json`, returns immutable tiles/tensors plus valid-pixel masks and
grid/order metadata. Use `implementation libs.jackson.databind`, matching tokenizer; include it
in published POM checks. There is no native dependency. Reject unrecognized operational options.
Tiles and their masks retain original image association; zero-padding slots never become
features for a real image. Compose common resize/normalize and split/crop primitives rather than
putting checkpoint naming in those primitives. Freeze the processor result signatures after WP1
settles padding, tile ordering and variable resolution.

### Image oracle and exact tolerance

Add
`tools/image-oracle/{install.sh,runner.py,requirements.in,requirements.lock,provenance.json}`
with CPython 3.12, hash-locked Pillow/Transformers/NumPy and explicit fixture source hashes. Add
`verifyImageOracle`, `verifyImageOracleFixtures`, `generateImageOracleFixtures`; only the last
rewrites expected outputs. Normal Java tests read committed fixtures, with SHA verification in
root `check`; executable oracle checks are explicit and run in configured pure-Java CI.

Fixtures include identity, up/downsample, odd dimensions, portrait/landscape, split boundaries,
multiple tiles/global thumbnail, gradients, checkerboards, edge impulses, grayscale, alpha PNG,
palette/tRNS and grayscale-alpha PNG, explicit rejection fixtures for 16-bit PNG and CMYK/YCCK
JPEG, and committed licensed RGB/EXIF-rotated JPEGs. Separate decode, resize, normalized tensor
and full-processor assertions so downstream tolerance cannot hide wrong tile counts or ordering.

Target bit-exact uint8 output for every single resize, including Pillow's three-lobe
windowed-sinc LANCZOS. Port coefficient support/normalization, fixed-point precision,
intermediate pass rounding and clip8 behavior from the pinned Pillow source; record source
hash/license. Preserve the full copyright, permission and disclaimer text of the pinned [Pillow
MIT-CMU license](https://github.com/python-pillow/Pillow/blob/12.3.0/LICENSE) and any additional
notices on copied source in `jmlx-vision/NOTICE`, alongside the project's MIT license. Choose
vision-only packaging: the root convention already adds LICENSE/NOTICE to the main jar, but does
not configure sourcesJar or javadocJar. In WP2, add explicit `from(...)` blocks in
`jmlx-vision/build.gradle` for both ancillary jar tasks, placing the root MIT LICENSE and module
NOTICE under META-INF. Do not extend the root convention or change the six existing modules'
ancillary artifacts. Add vision-specific release-smoke checks for LICENSE and the full required
Pillow notice content in all three vision jars; retain other modules' runtime checks. Keep the
applicable Pillow copyright, permission and disclaimer notice as a header comment in each ported
source file as well, so extracted source files retain it independently of META-INF. Verify those
headers against the pinned source notices in WP2. Reject unsupported numeric resample values
rather than mapping them to a nearby filter. Java2D rendering hints alone do not establish
equivalence.

The target processor chains uint8 resizes: longest edge 2048, encoder-compatible multiples of
512, then a global-image resize from the second-stage output inside split_image. Rescale and
normalize only after that chain. Add stage-by-stage outputs and full-chain tile/global outputs
for all three kernels. Negative LANCZOS lobes mean single-stage error bounds cannot be reused as
full-chain bounds. Record maximum/mean error at each stage and at the final processor output.
Exact single-stage/full-chain bytes are the initial acceptance target. If exactness is not
achieved, acceptance stays pending until a reviewed amendment records separate measured numeric
single-stage, full-chain and JPEG-to-full-chain bounds on both hosts; no implicit one-level
fallback. Once a final uint8 bound E is approved, normalized channel tolerance is `E *
abs(scale) / std[channel] + 1e-6`; E is the measured complete-chain bound, not a stage bound.
Dimensions/grid/masks/order and identity/copy remain exact. Model-math goldens use identical
hash-pinned oracle tensors; Java-preprocessor end-to-end cases get independently recorded
bounds.

## 5. Additive request, metadata and chat contracts (WP3)

Add `GenerationRequest.withImages(List<RgbImage>)` and `images()`. Reject null list/elements;
use an immutable list of immutable images, empty by default. Copies retain tokenizer, prompt,
config, cancellation and cache policy. Test caller-list/pixel mutation and both copy-method
orders.

Add public `InputModality { TEXT, IMAGE }` and `default Set<InputModality>
ModelMetadata.inputModalities()` returning immutable `{TEXT}`. Vision metadata returns `{TEXT,
IMAGE}`. Reject images in every existing decoder and T5 model before native work, naming model
type. Encoder/classifier APIs have no GenerationRequest input. Scheduler admission rejects
nonempty images synchronously; its existing non-DecoderModel start check rejects the composing
VLM and names the model type and “not supported by the batch scheduler”.

Allow string content or a list of `{type: text, text: String}` / `{type: image}` maps through
`HfTokenizer.renderChat`. Validate list elements and known types, reject video/audio and
malformed text; preserve ordering and the template's structured values. Image descriptors are
placeholders; no URL/path is fetched and pixels come only from `withImages`, in encounter order.
Document this on chat/request APIs. Preserve string behavior and render options. Bundle the
pinned processor chat template into metadata loading when it is not in tokenizer_config, without
letting a global fallback alter text tokenizers. Test through `GenerationRequest.chat`, not only
direct rendering.

Extend HF `--chat` generation to use the pinned processor. Commit rendered unexpanded text/IDs
separately from processor-expanded IDs for one/two/adjacent images, system/multi-turn content,
interleaved text, and text-only content. Encoding fixtures belong in tokenizer-oracle; processor
rendering/expansion references belong in HF/image tools. Add any required Jinja feature with its
own regression and CHANGELOG entry. Qwen3-VL execution tests wait for 7.3b; generic structured
content must preserve its wrapper-shaped template output without assuming SmolVLM syntax.

Delivered via PR #42 (merge `0dac174`, 2026-10-08): `GenerationRequest.withImages`/`images()`,
`InputModality`, model and scheduler image rejection, structured `renderChat` content, the pinned
chat template in metadata loading, and the HF `--chat` golden (`tools/hf-reference/goldens/
chat-smolvlm.json`) with unexpanded and expanded IDs committed separately.

## 6. Pure expansion and generation accounting (WP4)

Implement a package-private pure `SmolVlmPromptExpander` in models. Inputs are unexpanded IDs,
validated special IDs and per-image processor grids. Output owns expanded IDs and ordered image
feature positions/ranges. It never receives native tensors. The validated special IDs also carry
the newline-run encodings: the reference expands at the string level and re-tokenizes the result
(reference findings §6), and the only non-special characters it inserts are the row newlines — a
lone `\n` after every row except the last and the merged `\n\n` run before the global block, both
ordinary BPE tokens resolved from the tokenizer at model load (`[198]` and `[1116]` for the pinned
SmolVLM-256M tokenizer). Reject newline IDs that collide with the image, fake, global or video
token or the row/column marker block: a run is a plain-text encoding and a colliding ID would
insert a token that no feature destination owns. The plan/placement/token records own their
arrays; accessors return defensive copies and internal paths read the fields directly. The
records compare by value (arrays elementwise) and render the arrays in toString.

Count each `<image>` as one unit. `<image><image>` is valid for two supplied images. Reject
count mismatches, any processor-only fake/row-column/global marker and unsupported video with
distinct messages. Unsupported video means the tokenizer's `<video>` token when the tokenizer
declares one; the pinned SmolVLM-256M tokenizer has no `<video>` added token, so WP3's
content-part rejection is the effective gate there. Freeze exact markers and tile/global sequence
in WP1 (amended in place with the re-tokenization counts, per the committed golden); feature
positions must exclude wrapper/row/column text. Expand after preprocessing, before native
allocation; use checked token length arithmetic. Validate every projected-feature row has exactly
one destination, in order. The expander anchors the marker block at its base and cannot detect a
block that is contiguous but ordered differently, so WP6 must verify all 36 `<row_r_col_c>` IDs
against the row-major formula when it resolves them from the tokenizer (or store the 36 IDs in an
explicit table).

Use expanded IDs for penalties, positions, context limits and cache budget `expandedLength +
maxNewTokens - 1` when maxNewTokens is positive. Include both unexpanded and expanded lengths in
capacity errors. Bound total images, decoded pixels, tiles and expanded positions before
building GPU arrays. Cancellation is checked before preprocessing, between image preparations,
before prefill and between decode steps.

For `maxNewTokens == 0`, still validate placeholder counts and processor-only/video markers, run
pure geometry planning and expansion, and report the expanded promptPositions. Skip pixel
resize/rescale/normalize, vision tower, embedding/native work and KV allocation; required cache
capacity is zero. Geometry planning must share the processor's rounding/splitting code so counts
cannot drift; it is delivered as pure `SmolVlmImageProcessor.geometry(height, width)` /
`SmolVlmGeometryPlan` in jmlx-vision, which `intermediates()` itself uses. Validate context
length and image/geometry limits even in this path. Test rejection, position reporting and zero
native/tower calls explicitly; the model-level parts of this paragraph (pixel-work skip, zero
tower calls, the cancellation points) are realized in WP6's `SmolVlmModel.generate`, which
composes these pure components, while WP4's acceptance covers their pure side.

Add `int promptPositions` as a GenerationResult component: define it as the effective input
prompt length, expanded VLM length, ordinary decoder/scheduler prompt length, and **encoder
source length for T5**, never decoder start/history or charged cache capacity. Retain four- and
five-argument constructors defaulting to prompt ID count. Require `promptPositions >=
promptTokenIds().size()` for all current results; T5 uses equality. Update DecoderModel,
BatchGenerationScheduler and T5Model construction sites explicitly. This changes record
equality/hash/toString and breaks record deconstruction source patterns; it is a deliberate
pre-publication compatibility amendment, not a strictly additive API change. Recheck publication
status before implementation; if GenerationResult has been published, stop this shape change and
amend the plan to a compatible result-metadata design. `promptTokenIds()`, `tokenIds()` and
aborted partial results always use unexpanded prompt IDs. Inspect `GenerationAbortedException`
and preserve that invariant on every wrapped failure. Do not change GenerationEvent token
indices or count the expanded prompt toward maxNewTokens.

Tests cover exact HF expansion, different tile counts, images between text, malformed tokens,
count mismatch, exact/one-over capacity, zero/one/many generated tokens, EOS/stop and abort IDs.
Generated image-special tokens are ordinary decoder history, not an instruction to re-run
vision.

Delivered via PR #43 (branch `phase7-3a-wp4`, 2026-10-08): the package-private
`SmolVlmPromptExpander` and the `SmolVlmPromptTokens`/`SmolVlmImageGrid`/`SmolVlmImagePlacement`/
`SmolVlmPromptPlan` records in `se.alipsa.jmlx.models`; the shared
`SmolVlmImageProcessor.geometry`/`SmolVlmGeometryPlan` landed in jmlx-vision in the same PR.
`SmolVlmPromptExpanderTest` matches all six golden `expanded_ids` byte-for-byte against the
pinned transformers 4.57.6 reference (grids derived from the shared geometry plan, not the
golden's names) and covers the synthetic unsplit/split sequences, prompt-order placements, every
distinct rejection (count mismatch, fake, row/column, global, video, the 6x6 bound, length
overflow) and the cache budget at exact/one-over/zero. `GenerationResult` is now a six-component
record with `promptPositions` (the four-/five-argument compatibility constructors default to the
prompt ID count; the floor `promptPositions >= promptTokenIds().size()` is enforced;
`DecoderModel`, `T5Model` and `BatchGenerationScheduler` pass the explicit length). Runtime
coverage: `LlamaModelTest` zero/one/many/EOS/stop/abort, `Seq2SeqGenerationTest` T5
source-length equality and aborted prompt IDs, `BatchGenerationSchedulerTest` per-row scheduler
lengths against the direct path. The model-level `maxNewTokens == 0` path (pixel-work skip,
zero tower calls, the cancellation points) remains in WP6's `SmolVlmModel.generate`. Post-review
amendments: the plan's minimum-length check computes the per-image marker minimum in long
arithmetic (an int wrap of `tokensPerTile + 3` would pass the check at extreme values); the
record array accessors return defensive copies; `SmolVlmPromptTokens` rejects newline IDs
colliding with the special tokens or the marker block; the pinned-fixture test verifies all 36
`<row_r_col_c>` IDs against the row-major formula, not just the block's ends. Second review
round: the array-holding records override equals/hashCode/toString with elementwise array
semantics (the generated versions compared and printed the arrays by reference), and
`featureDestinations` reads the placements' internal start arrays through a package-private
raw accessor so the same-package path pays no copy.

## 7. Decoder refactor and model assembly (WP5–WP6)

First build the exact comparison recorder in §9 and capture the base implementation. Factor
embedding lookup from the stack behind an internal embedding-start method, with explicit
sequence width/valid lengths and unchanged cache checks. Invoke EmbeddingHook at the identical
post-lookup/pre-scale point; scheduler Hooks and production token-ID callers retain their
contracts. Preserve graph operation ordering, dtype and synchronization. No public forward-from-
embeddings API is needed for this milestone.

Delivered via PR #44 (branch `phase7-3a-wp5`, 2026-10-09) — this section's WP5 parts, the §9
recorder and the refactor. The opt-in `:jmlx-models:exactBitsRecord` / `exactBitsVerify` tasks
(new `exactBits` source set; compiled by `check` via `checkstyleExactBits`, never run by it)
capture raw float bits and exact greedy IDs for `direct-prefill`, each golden decode step,
2-row left-padded `stepLogits` batch prefill/decode steps (null and a deterministic non-null
hook), direct greedy generation and an end-to-end gated two-request scheduler run, for all eight
Tier-A decoder families plus the six derived 4-bit/group-32 quantized variants the README claims,
in both TF32 modes. Recordings (14 variants × 9 captures per mode) live under
`build/exact-bits/tf32-{1,0}/` with commit, native-pin, device, macOS, precision, spec and
fixture-hash metadata; verify allows only the git commit to differ. The refactor splits
`normalizedHiddenStatesBatch` into the private `decoderEmbeddings(tokenIds, hook)` (lookup, hook
at the identical post-lookup/pre-scale point, sqrt-hidden scale) and the package-private `final
decoderStack(embeddings, caches, validLengths, sequenceWidth)` (left-padded mask, RoPE, blocks,
final norm; no evaluation, no poisoning, callers keep preflight/postflight); the single-row path
reuses `decoderEmbeddings` and keeps its own sliding-window mask. No public or protected member
was added and T5 was not touched (its §9 capture clause stays live for any later shared-generation
extraction). Acceptance: `exactBitsVerify` bit-exact in both modes against the pre-refactor
baseline recorded at `d658309` (a one-bit recording flip fails with the first divergent element
named), and every existing scheduler test green, including `BatchStepEquivalenceTest` in both
modes.

Add `SmolVlmModel implements TextGenerationModel`, composing DecoderModel rather than extending
it. Dispatch confirmed artifact model types before decoder-only ArchitectureMappings parsing.
Use a family-specific descriptor/tensor plan for outer vision/connector state and map the nested
text config into the existing decoder assembler with explicit tensor prefixes. Check tie
aliases, vocabulary/image IDs, dimensions and all unexpected tensors before returning a model.
Convert PyTorch patch-convolution weights `[out,in,kH,kW]` to `[out,kH,kW,in]` by explicit
provenance and shape validation. Preserve float checkpoints initially; document the selected
compute dtype. Use FLOAT32 for initial correctness/Tier-B acceptance, converting FP16/BF16
source weights with existing core facades and accounting for memory expansion.

Build reusable `nn` vision modules from `Conv2d`, Embedding, LayerNorm, bidirectional attention,
FFN/activation and EncoderBlock where compatible. Model packages never import ffi. Match valid
patch masks, variable-resolution position buckets, residual/norm order and configured
activation; verify intermediate patch/tower/connector tensors against CPU references. Pixel
shuffle uses existing reshape/transpose and linear operations with index-coded permutation
tests.

Use one `take` gather over `concatenate([textEmbeddings, projectedFeatures])` along the sequence
axis, indexed by the expander's validated destination map. Batch-one shapes are `[1,T,H]`,
`[1,F,H]`, and output `[1,T,H]`; map text destinations to their original row and image
destinations to `T + featureRow`. Validate bijective feature consumption and integer bounds in
pure Java. Probe ordering, dtype, scope and temporary allocation behavior. This is the dated
2026-10-06 parent amendment replacing masked-scatter-first with existing core compositions.
`mlx_masked_scatter` is deferred unless measured needs justify a separately probed core wrapper,
ownership/error tests and inventory mapping. No ffi calls or opaque mutation in models.

Prefill merges image features once; subsequent steps use ordinary token lookup and KV caches.
Extract shared generation mechanics only where necessary to avoid duplicating sampling/streaming
logic; keep separate original and effective prompts in an internal execution context. T5's
source/start/history semantics remain independent. Test shared-loop changes against both paths.

## 8. Request ownership and cache/memory evidence

A model holds weights only. Adopt parent decision 8's 2026-10-08 lifetime amendment: within
generate, use a per-request generation scope for KV and a prefill child scope for projected
image features. Evaluate/hoist features from shorter vision activation children into the
prefill scope. Evaluate prefill logits and retained KV, then close the prefill scope and
release features and embedding-hook references before cached decode. Decode reads only KV
and surviving request state. Build KV caches in the generation scope. Always close request scopes
on completion, cancellation, listener/tokenizer/native failures and failed prefill.

Use a package-private models observer to count actual vision/connector executions and identify
request state. Assert once per image/tile batch at prefill, zero during cached decode, and fresh
state for a second request. Observer injection is test-only; production uses a no-op.

Add `VisionLanguageMemoryTest` with warm-up, evaluated arrays and `MLXMemory.activeBytes()`:
repeated same/different images, changing tile counts, text-only requests, cancellation before
vision/between steps, and injected post-prefill/listener/tokenizer/native failures return to
baseline within a measured fixed allocator margin. Verify features remain live after temporary
vision activation closure, visual storage is released after prefill, retained KV remains valid
after prefill closure, and request-lived state becomes unusable after request closure.

For batch one, FLOAT32 retained self-KV bytes per position are `2 * decoderLayers * kvHeads *
headDim * 4`. Projected features add `totalImageFeatureRows * textHiddenSize * 4` during prefill,
plus documented mask/metadata overhead; their retained cost during decode is zero. Assert
post-prefill visual release within a measured allocator margin. Use KV bytes and a
measured fixed margin for early/late per-token slope tests. Full(capacity) rejects an
over-budget request before native work and never evicts; only sliding retention requires a
plateau after its window fills. Support the composed text decoder's existing cache policies only
after cached/full equivalence and sliding image-prefill tests pass; if an artifact's positions
forbid a policy, reject it explicitly and record the restriction.

## 9. Numerical, regression and Tier-B gates (WP7)

Create an opt-in Gradle recorder/comparator before refactoring. Capture raw float bits and exact
greedy IDs for prefill and several cached steps of every existing Tier-A decoder family,
including Qwen3 and applicable quantized fixtures, in both TF32 modes. Include direct and
scheduler batch paths with their existing hook tests. If shared generation extraction touches
T5, capture its raw prefill/cached logits and exact greedy IDs in both modes before that
extraction and require the same exact comparison; tolerance coverage alone is insufficient.
Record commit, both native pins, device, macOS, precision, fixture/request hashes and batch
configuration. Compare base/candidate on the same tuple allowing only commit differences. Store
recordings as PR evidence, not portable CI goldens. A tolerance pass cannot replace this
comparison; inability to run the original GPU profile leaves acceptance pending on the required
macOS 26 host.

Extend HF fixtures with a tiny Idefics3-compatible checkpoint: asymmetric/nontrivial weights,
patch/padded-position/mask outputs, connector features, image-replaced embeddings, prefill
logits, at least three cached decode steps and exact greedy IDs. Cases cover one image, two
different tile counts, image/text/image, and text-only input. Compare cached logits with full
uncached forwards over the same expanded history. Pin all
source/checkpoint/output/image/template hashes and verify them through
`verifyHfReferenceGoldens`; regenerate only manually and review diffs.

Strict CPU float32 references use full-float32 suites in float32GoldenTest, absolute `1e-4`;
ordinary test suites assert a separately named, measured TF32 bound. Document default-mode
errors and combined Java-preprocessor end-to-end errors in `req/phase6-golden-precision.md`.
Never loosen strict model-math assertions or regenerate goldens to fit TF32. Compare model math
on identical oracle pixels and preprocessing independently under §4's approved full-chain bound
(or exactness gate).

Extend Tier-B manifest/task dispatch with an image-text-generation task, image hashes/licenses,
structured prompt, unexpanded/expanded ID expectations, maxNewTokens and exact greedy-ID golden.
Downloader retains safe nested paths and SHA checks; committed test images require no runtime
URL. Pin SmolVLM's revision and run the existing TF32-off tierBTest profile. Record numerical
logits at every expected decoder history and top-two gaps. Exact-ID acceptance requires each gap
to exceed `2 * measuredAllowedLogitError + positiveSafetyMargin`; select a stable image/prompt
before acceptance rather than weakening exact IDs. Record peak memory/RSS and download size
before adding a weekly/on-demand CI row. No artifact download belongs in ordinary tests.

Add mandatory vision/model native suites to matching test and float32GoldenTest XML assertions
in macOS CI; use EnabledIfNativeAvailable. Ubuntu checks Java 21 vision bytecode, formatting,
lint, pure preprocessing and tokenizer tests plus image-oracle verification. Keep Python neural
reference regeneration generate-only.

## 10. Work packages and completion checklist

| Package | Deliverable | Depends on | Acceptance |
| --- | --- | --- | --- |
| WP1 | Pinned reference findings and mapping tables | 7.1 | All artifact/processor/architecture questions resolved |
| WP2 | Vision module plus publishing/smoke/CI/AGENTS integration, contracts and image oracle | WP1 | Java 21 Ubuntu checks, exact structure and stated pixel bound |
| WP3 | Images/modality APIs, structured rendering/template loading | WP1–WP2 | Deep immutability, copy-method and real-template regressions |
| WP4 | Pure expansion and effective/original prompt accounting | WP1–WP3 | Exact processor IDs, rejection and capacity tests |
| WP5 | Base bit recorder and embedding-entry refactor | WP1 | Exact comparison in both modes; existing scheduler tests pass |
| WP6 | Tower/connector, checkpoint assembly, multimodal prefill/generation | WP2–WP5 | Intermediate goldens, cache agreement and terminal-path semantics |
| WP7 | Memory, precision, Tier-B and CI gates | WP6 | Stable exact IDs, leak evidence, required native suites executed |
| WP8 | Final consumer evidence and capability documentation | WP2–WP7 | Seven-artifact consumer smoke and compatibility evidence |

These are reviewable PR boundaries, not permission to claim family support before all gates
pass. WP5 may precede WP3/WP4 once its fixture cases and baseline recorder are ready.

WP2 owns settings/build wiring, every hard-coded publication/version list, bytecode/license/
POM-versus-metadata consumer checks and the pure vision smoke assertion. Existing seeded
scheduler smoke remains. WP8 reruns that seven-artifact consumer and completes capability
evidence; it does not postpone integration required to keep earlier PRs green. Release vision
before models, keeping released dependency versions checked out until the chain completes. Run
buildSrc check if its publishing/downloader logic changes. No publication or manual version bump
is in scope.

Update AGENTS.md module/architecture/release sections and vision README in WP2; update each
module's README/CHANGELOG with the PR changing its API. WP8 finalizes
`req/phase6-tier-a-fixtures.md`, `req/phase6-tier-b-artifacts.md`, compatibility matrix and
inference report. Add a Phase 7.3a implementation report separating local evidence, macOS 26 CI
acceptance and deferred capabilities. Link this sub-plan from the parent milestone.

Required final checks include module check tasks, root build/check, spotless/checkstyle,
verifyBytecodeLevel, verifyReleaseScriptsMatch, verifyMlxApiCallSites and inventory freshness,
verifyHfReferenceGoldens, verifyImageOracleDependencyPins, image/tokenizer/MLX oracle
verification under their supported profiles, release-smoke ci, opt-in pinned Tier-B and both
exact bit comparisons. Native skips are not acceptance. Image oracle fixtures must pass on
Ubuntu and macOS. New bindings, if any, additionally require inventory/header coverage checks
without hand-editing generated sources.

Phase 7.3a is complete only when preprocessing, synthetic prefill/cached logits, real-artifact
exact greedy output, unchanged text bits, request/result contracts, cleanup and CI gates all
pass, and the matrix gains an evidence-backed “image+text → text” row. Pending artifact pins or
missing host evidence remain explicitly pending, never replaced by structural smoke alone.
