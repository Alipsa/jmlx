# Changelog

## Unreleased

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

Versions are unchanged; implementation does not publish artifacts.
