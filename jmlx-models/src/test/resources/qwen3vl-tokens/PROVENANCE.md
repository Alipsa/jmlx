# Synthetic Qwen3-VL token fixture

Hand-written minimal BPE `tokenizer.json`, not a real Qwen3-VL tokenizer. It exists only so
`Qwen3VlPromptTokens.resolve` can be exercised against a bare `tokenizer.json` without the
gitignored Tier-B artifact: its four added tokens carry exactly the same sentinel spellings as the
pinned Qwen3-VL-2B-Instruct artifact's IDs 151655/151656/151652/151653 (see
req/plans/phase7-3b-reference-findings.md section 6.1) but at small local IDs, so the resolve
match, config-mismatch and missing-token paths are all testable without downloading anything.
An earlier draft of this fixture used placeholder marker strings that did not match the pinned
artifact at all; that draft was itself a product of a mangled terminal display, and every such
string is gone from this fixture. The sentinels below are hex-verified against the pinned
artifact, and both the `model.vocab` map and the `added_tokens` array carry the identical bytes
for each one.

Each sentinel is `<` + `0x7c` + name + `0x7c` + `>`. The hex columns (UTF-8 bytes) are the
authoritative re-verification record: terminal displays cannot be trusted for these strings, so
compare bytes, never glyphs.

| ID | role         | name           | sentinel UTF-8 hex                               |
|----|--------------|----------------|--------------------------------------------------|
| 10 | image        | `image_pad`    | `3c 7c 69 6d 61 67 65 5f 70 61 64 7c 3e`          |
| 11 | video        | `video_pad`    | `3c 7c 76 69 64 65 6f 5f 70 61 64 7c 3e`          |
| 12 | vision start | `vision_start` | `3c 7c 76 69 73 69 6f 6e 5f 73 74 61 72 74 7c 3e` |
| 13 | vision end   | `vision_end`   | `3c 7c 76 69 73 69 6f 6e 5f 65 6e 64 7c 3e`       |

The base BPE vocabulary (IDs 0-9, 14-20) holds the 17 distinct bytes the sentinels are composed
of (`<`, `0x7c`, `>`, plus the lowercase letters and underscore of the four names); `merges` is
empty because added tokens never pass through BPE.
