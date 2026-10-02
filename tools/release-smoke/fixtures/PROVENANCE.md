# Release-smoke tokenizer: provenance

`tokenizer.json` and `tokenizer_config.json` are **hand-written, not Hugging Face-generated**. They
are produced deterministically by `build-tokenizer.py` in this directory (rerunning it must leave the
committed files unchanged).

Why it exists: it pairs with the committed synthetic Mistral checkpoint at
`tools/hf-reference/goldens/checkpoints/mistral/` (`vocab_size` 128, `bos_token_id` 1,
`eos_token_id` 2, `pad_token_id` 0). The tokenizer under `jmlx-tokenizer/src/test/resources/families/
mistral/` is not a pair for it: it defines only ids 0-24 with BOS/EOS 21/22, and the decoder returns
`""` for ids above the largest known id, so sampling over 128 logits would silently drop ids 25-127
and stream empty deltas while an exact golden still passed.

Shape: a ByteLevel BPE with ids 0-2 as the special tokens `<pad>`, `<s>`, `</s>`, ids 3-98 the
byte-level stand-ins for printable ASCII and newline, and ids 99-127 common-bigram merges. Every id
0-127 is defined, so every id the model can emit decodes to text, apart from the specials. The chat
template is a minimal `[role] content` renderer that prepends `<s>`.

The smoke asserts that the tokenizer's id range equals the checkpoint's `vocab_size` and that its
BOS/EOS equal the checkpoint's.
