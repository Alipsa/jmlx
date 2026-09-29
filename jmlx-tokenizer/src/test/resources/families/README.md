# Phase 6.3 family tokenizer bundles

These are deliberately small synthetic BPE vocabularies derived from the existing
`tools/tokenizer-oracle/fixtures/metaspace-bpe.tokenizer.json` component fixture. Their IDs and
vocabularies are not real model vocabularies and cannot be paired with production checkpoints.
The Mistral and Mixtral chat template is copied from the existing
`jmlx-jinja/src/test/resources/model-templates/mistral-7b-instruct-v0.3.jinja` corpus fixture.
The Gemma and Phi-3 templates are small family-shaped templates: Gemma uses
`<start_of_turn>` and the `assistant` → `model` role mapping; Phi-3 uses its role and end markers.
These two templates are synthetic and do not claim byte-for-byte production-template compatibility.

`tools/hf-reference/generate.py --chat` loads each committed bundle with pinned Transformers,
renders six conversation cases through `AutoTokenizer.apply_chat_template`, and tokenizes the
rendered text without adding special tokens. `tools/tokenizer-oracle` separately pins plain encode
behavior for the same tokenizer JSON files. Production artifact compatibility remains a Tier-B task.
