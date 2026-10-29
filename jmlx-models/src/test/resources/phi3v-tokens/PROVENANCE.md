# Real Phi-3.5-Vision token fixture

The artifact's own `tokenizer.json` and `tokenizer_config.json` (the `LlamaTokenizerFast`
configuration plus the full BPE model), byte-identical to the pinned
`microsoft/Phi-3.5-vision-instruct` revision `12b77fb40b63a2c73c68243d3f767aab688a1b2a`
(entry `wp3_token_fixture` in `tools/hf-reference/provenance.json`; the same digests appear in
`tools/hf-reference/goldens/phi3v-wp1-probe.json`'s `pins` block):

| file                     | bytes     | SHA-256                                                            |
|--------------------------|-----------:|--------------------------------------------------------------------|
| `tokenizer.json`         | 1,851,360 | `493f8ad437be806d5ccdc57e91479bdbb0c656448c1b89ff70f7653530d78518` |
| `tokenizer_config.json`  | 9,519     | `581d4654c018fbcef2504b57a805dcad37126038137b5d25d22b5a02abc1ef29` |

Unlike the synthetic `qwen3vl-tokens` fixture (a hand-written minimal BPE tokenizer), this one
is the real artifact file: the WP3 golden
(`tools/hf-reference/goldens/phi3v-wp3-prompt.json`) was generated through it, so committing it
keeps `Phi3VPromptExpanderTest` a pure, network-free byte-exact replay of the reference text
path (findings §7.1) instead of a re-tokenization approximation.

## Marker note (hex-verified)

The per-image markers are `<` + `0x7c` + `image_i` + `0x7c` + `>` — e.g. marker 1 is the UTF-8
bytes `3c 7c 69 6d 61 67 65 5f 31 7c 3e` — and are **not** added tokens: they tokenize as
ordinary text and never survive to the expanded IDs (they are replaced by the sentinel rows of
`-i`). The literal single-word added token `<` + `0x7c` + `image` + `0x7c` + `>` (bytes
`3c 7c 69 6d 61 67 65 7c 3e`) **is** an added token at id 32044 (golden `tokenizer_facts`);
`Phi3VPromptTokens` deliberately neither treats it as a marker nor rejects it. Terminal displays
cannot be trusted for these strings; compare bytes, never glyphs.

The 13-character `endoftext` literal (special token id 32000, the stop token) is likewise
present in `added_tokens`; its content is written by ords in the findings (§7.3) for the same
transport reason.
