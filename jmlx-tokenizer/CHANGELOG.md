# Changelog

## Unreleased

- `renderChat` accepts structured message content: an ordered list of `{"type": "text",
  "text": ...}` and `{"type": "image"}` parts. Parts keep their order, image parts render the
  template's image marker as a placeholder (pixels never come from the messages), and any other
  part shape is rejected. Plain-string content keeps its existing behavior.
- A legacy processor file `chat_template.json` (`{"chat_template": "..."}`) now fills the
  `default` template slot when `tokenizer_config.json`, a root `chat_template.jinja`, and the
  additional-templates directory all leave it unset. Existing default sources still take
  precedence, and a bundle with no template source still has no templates.
- Add the real, unmodified SmolVLM-256M-Instruct tokenizer bundle (pinned revision, all file
  SHA-256s in `req/plans/phase7-3a-reference-findings.md`) under the test `families/` resources,
  with its pinned chat golden and `tools/tokenizer-oracle` encoding fixtures.
- Add SentencePiece Precompiled normalization with original UTF-8 offset alignment.
- Add BERT pair encoding with `PairEncodingOptions` and all three `PairTruncationStrategy` values.
- OnlySecond configurations now load. Short single inputs succeed; truncating a missing second
  sequence fails during encoding, matching the locked Rust oracle.
- Pair offsets are input-local; type IDs and masks identify inputs only for supported BERT templates.

The additive API belongs to this module's independent release. Versions are unchanged.
