# Phase 6.3 family tokenizer bundles

These are deliberately small synthetic BPE vocabularies derived from the existing
`tools/tokenizer-oracle/fixtures/metaspace-bpe.tokenizer.json` component fixture. Their IDs and
vocabularies are not real model vocabularies and cannot be paired with production checkpoints.
The Mistral and Mixtral chat template is copied from the existing
`jmlx-jinja/src/test/resources/model-templates/mistral-7b-instruct-v0.3.jinja` corpus fixture.
The Gemma and Phi-3 templates are small family-shaped templates: Gemma uses
`<start_of_turn>` and the `assistant` → `model` role mapping; Phi-3 uses its role and end markers.
These two templates are synthetic and do not claim byte-for-byte production-template compatibility.
The Qwen3 bundle embeds the **real** Qwen3 chat template from
`Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` (SHA-256
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`, also committed to the
`jmlx-jinja` model-template corpus as `qwen3-0.6b.jinja`); its added tokens cover the template's
special markers, and the `tokenizer_config.json` keeps every live scalar field except the
vocabulary-keyed `added_tokens_decoder`/`additional_special_tokens` (the live empty-string
`pad_token` is nulled, since it is not in the synthetic vocabulary).

The `smolvlm` bundle is different: it is the **real**, unmodified tokenizer bundle of
`HuggingFaceTB/SmolVLM-256M-Instruct` pinned at revision
`7e3e67edbbed1bf9888184d9df282b700a323964` (SHA-256 per file in
`req/plans/phase7-3a-reference-findings.md`; the model weights themselves are not committed). It
exists for the Phase 7.3a vision work: its chat template iterates structured content parts and
renders image parts to an `<image>` placeholder, and the bundle's legacy processor file
`chat_template.json` carries the same template string that `tokenizer_config.json` already holds
(the loader prefers the config). `tools/hf-reference/generate.py --chat` renders its eight
conversation cases (text-only, single/two/adjacent images, system and multi-turn) through
`AutoProcessor.apply_chat_template`, and `tools/tokenizer-oracle` pins the encoding of the exact
rendered texts in the `smolvlm.*` fixtures.

`tools/hf-reference/generate.py --chat` loads each committed bundle with pinned Transformers,
renders the family's conversation cases (three standard conversations for the earlier families;
the enable_thinking-on/off matrix plus a tool-call conversation for Qwen3, each recorded with the
per-case template variables it was rendered with) through `AutoTokenizer.apply_chat_template`, and
tokenizes the rendered text without adding special tokens. `tools/tokenizer-oracle` separately pins
plain encode behavior for the same tokenizer JSON files. Production artifact compatibility remains
a Tier-B task.
