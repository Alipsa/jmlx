# Phase 7.2 tokenizer fixture attribution

These derived inputs are Apache-2.0; see `LICENSE-Apache-2.0.txt` and immutable model-card
license evidence in `tools/tier-b/*.json`.

| Input | Upstream | Revision | Original bytes | Derived bytes |
| --- | --- | --- | ---: | ---: |
| phase72-precompiled.tokenizer.json | google/flan-t5-small | 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab | 2424064 | 1376750 |
| phase72-wordpiece.tokenizer.json | yoshitomo-matsubara/bert-base-uncased-sst2 | ce4cfd087e0c988beae0699f77a610bf7916742c | 466132 | 466133 |

Flan-T5 is Google's instruction-tuned T5 model. The SST-2 artifact is maintained by Yoshitomo
Matsubara and derives from Google's BERT. Attribution is retained in downloaded model cards.
`derive-phase72.py` compacts JSON and removes Flan added-token metadata, padding and truncation.
It preserves the real charsmap byte-for-byte; source/charsmap hashes are in `provenance.json`.
OnlySecond derives from WordPiece by changing its truncation strategy. These are test inputs,
with no model weights; modifications do not imply upstream endorsement.
