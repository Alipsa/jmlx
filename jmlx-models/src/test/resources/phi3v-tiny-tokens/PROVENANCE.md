# Synthetic Phi-3.5-Vision tiny token fixture

Hand-written minimal byte-level BPE `tokenizer.json`, not a real Phi-3.5-Vision tokenizer. It
exists only so `Phi3VModel`'s load and generation tests can run against the tiny `phi3_v`
checkpoint (21-token vocabulary, matching the checkpoint's `vocab_size: 21`) without the
1.85 MB real tokenizer fixture in `../phi3v-tokens/` (which pairs with the pinned artifact and the
WP3 golden, and whose 32,045 tokens exceed the tiny checkpoint's vocabulary bound).

The 21 tokens are exactly: `<` (0), `|` (1), `>` (2), the digits `1`-`9` (3-11), the
byte-level-escaped space `Ġ` (12)
and the letters `a b c d e o s t` (13-20). That covers the per-image marker spellings
`<|image_N|>` for `N` in 1-9 (the markers are ordinary text, never added tokens — see
`../phi3v-tokens/PROVENANCE.md`) plus a few short test words composed of the eight letters
(e.g. `bad`, `best`, `toast`, `code`). The pre-tokenizer, byte-level post-processor and decoder
mirror the sibling `qwen3vl-tokens` fixture byte for byte. There are no added tokens: the
fixture's `vocabSize()` is 21, equal to the tiny checkpoint's `vocab_size`.
