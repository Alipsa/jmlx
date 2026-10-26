# Changelog

## Unreleased

- Add the Phase 7.3b WP8 Tier-A acceptance suites: `Qwen3VlGoldenTest` (full float32, strict
  `1e-4`) and `Qwen3VlDefaultPrecisionTest` (default TF32 bound `0.0025`) run the pinned tiny
  WP8 Qwen3-VL checkpoint's four plan cases and both forced generated-marker continuations
  against the golden's fresh full uncached CPU float32 reference, including expansion/grids/
  pixel hashes, three-axis positions, tower merged/DeepStack rows and per-step DeepStack
  injection counts. The tiny checkpoint, golden and generator are digest-pinned; the measured
  maxima (`2.3245811e-6` strict, `0.0019590855` TF32) are recorded in
  `req/phase6-golden-precision.md`.

- Verify Qwen3-VL cached/full logits and continuation-axis consistency in both precision modes,
  including generated markers, and evaluated visual release/KV growth on cancellation and failure.
  Process images sequentially with cancellation checks between tower executions. FULL and bounded
  FULL retain all positions; sliding retention remains rejected. Real-artifact acceptance is WP8.

- Add WP7 evaluated vision-language memory and lifetime regressions, including repeated varied
  requests, cancellation, listener/tokenizer/native failures, visual release and derived full-KV
  retention budgets. Require the memory suite in macOS CI and upload native acceptance evidence.
- Add opt-in `image-text-generation` Tier-B dispatch, a pinned SmolVLM-256M checkpoint/image/chat
  manifest, full CPU logits at prefill and three cached histories, stable exact greedy IDs and
  sampled memory evidence. The weekly/on-demand workflow includes the new task; local macOS 27
  evidence passes while macOS 26 acceptance remains pending.

- Compare tied image/text checkpoint head aliases on the device, reading back one scalar;
  numerical equality accepts signed-zero differences and rejects NaNs. Normalize request-tokenizer
  marker validation failures to `IllegalArgumentException` while preserving their causes.
- Remove the shared generation loop's unused step-zero token array/preflight; first-step callers
  own their real prompt validation and cache-failure guards. Document the production embedding
  hook and its position before embedding scaling.

- Add `SmolVlmModel` and `idefics3` dispatch through `TextGenerationModels.load`: strict floating
  checkpoint assembly, FLOAT32 computation, image-marker resolution, tower/connector prefill,
  concatenate/gather feature replacement and cached generation using the shared decoder loop.
  Metadata reports `{TEXT, IMAGE}`. Full/bounded-full retention is supported; sliding VLM
  retention, memory evidence and real-artifact acceptance remain deferred.
- Verify tiny Idefics3 intermediate tensors, merged embeddings, prefill/cached logits, cached/full
  equivalence and exact greedy IDs; cover terminal/failure paths, zero-token geometry-only
  requests and actual visual execution counts. Require the new native suites in CI.
- Fix tower position-table validation to use its learned grid instead of patch-kernel size, and
  use the configured learned grid for rectangular valid-patch positions. Remove the connector's
  unnecessary channel-divisibility restriction and check merge-index overflow.
- Zero-token decoder requests now avoid the shared loop's native generation/sampling scope.

- Add `GenerationRequest.withImages(List<RgbImage>)` / `images()`: requests start image-free and
  carry an immutable, copied list of pixel values. Copy methods (`withImages`, `withCachePolicy`)
  retain the images in either order.
- Add the public `InputModality` enum. `ModelMetadata.inputModalities()` defaults to the immutable
  `{TEXT}` set; every delivered model (decoder and T5) and batch-scheduler admission reject
  non-empty images before any native work, naming the model type.
- `GenerationRequest.chat` accepts structured chat content (`{"type": "text"}` / `{"type": "image"}`
  parts) via `jmlx-tokenizer`'s validated `renderChat`; image parts are positional placeholders
  paired with `withImages` in order of appearance. Text parts, not plain strings, carry text to
  the prompt: under a template that prints only structured parts, plain string content renders
  as an empty turn (see the README).
- Add BERT encoding, sentence-transformers pooling, sequence and token classification.
- Add T5/Flan-T5 generation, relative bias and static cross caches, including `T5LoadOptions`.
- Add defensive result records and encoder-depth metadata.
- Reject T5 in the batch scheduler and both sliding policies before resolving cache defaults.
- Add task-specific MiniLM, BERT SST-2 and Flan-T5 Tier-B checks.
- `GenerationResult` gains `promptPositions`: the effective input prompt length the model
  consumed. It equals `promptTokenIds().size()` for the decoder models, the batch scheduler, and
  T5 (the encoder source length); for vision models it is the expanded prompt length, where each
  `<image>` placeholder stands for a block of marker-delimited image tokens. `promptTokenIds()`
  always retains the unexpanded prompt IDs (as do `tokenIds()` and `GenerationAbortedException`
  partials), so `promptPositions >= promptTokenIds().size()`. The four- and five-argument
  compatibility constructors default `promptPositions` to the prompt ID count.
- Factor the decoder stack's embedding lookup into an internal embedding-input helper plus a
  package-private embedding-start stack entry that takes decoder-ready activations, the explicit
  padded sequence width and the valid lengths. No public or protected API changed; the embedding
  hook still fires at the identical post-lookup/pre-scale point and the refactor is verified
  bit-exact in both TF32 precision modes against the pre-refactor recordings. The upcoming
  SmolVLM work feeds merged text and image embeddings through this entry.
- Add the opt-in `exactBitsRecord` / `exactBitsVerify` Gradle tasks: an exact-bit
  recorder/comparator that captures raw float bits and exact greedy IDs for every Tier-A decoder
  family (plus derived quantized variants) across direct forward, the lazy batched step (null and
  non-null embedding hook), greedy generation and the batch scheduler, in both TF32 precision
  modes. The recordings under `build/exact-bits` are PR evidence, never committed goldens and
  never part of `check`.

Versions are unchanged; implementation does not publish artifacts.
