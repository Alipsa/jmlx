# Changelog

## [0.5.0] - 2026-11-17

Date denotes the version cut, not publication to Maven Central.

Initial Central release; the entries below cover changes tracked since changelog adoption,
not the full public API.

- Add key-only bidirectional masks, explicit-scale bidirectional/cross attention, pre/post-norm
  encoder blocks and request-owned, non-appending `StaticKVCache`.
- Add scale constructor overloads while retaining existing default scales.
- Add `MLXMemory.clearCache()`, which releases the Metal allocator's pooled device buffers
  back to the OS without affecting live arrays (wraps `mlx_clear_cache`).
