# Changelog

## Unreleased

- Add SentencePiece Precompiled normalization with original UTF-8 offset alignment.
- Add BERT pair encoding with `PairEncodingOptions` and all three `PairTruncationStrategy` values.
- OnlySecond configurations now load. Short single inputs succeed; truncating a missing second
  sequence fails during encoding, matching the locked Rust oracle.
- Pair offsets are input-local; type IDs and masks identify inputs only for supported BERT templates.

The additive API belongs to this module's independent release. Versions are unchanged.
