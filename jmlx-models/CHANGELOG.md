# Changelog

## Unreleased

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
