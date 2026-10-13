# Changelog

## Unreleased

- Add key-only bidirectional masks, explicit-scale bidirectional/cross attention, pre/post-norm
  encoder blocks and request-owned, non-appending `StaticKVCache`.
- Add scale constructor overloads while retaining existing default scales.
- Add `MLXMemory.clearCache()`, which releases the Metal allocator's pooled device buffers
  back to the OS without affecting live arrays (wraps `mlx_clear_cache`).

Native pins, generated bindings and versions are unchanged.
