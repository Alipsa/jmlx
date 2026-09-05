# Phase 6.2 tokenizer component evidence

This matrix records the exact offline evidence behind Phase 6.2. The reference is `tokenizers`
0.23.2's Python binding over the Hugging Face Rust runtime; hashes and supported oracle platforms
are pinned in `tools/tokenizer-oracle/provenance.json`. Fixture generation opens only committed local
paths with Hub access disabled.

| Family | Fixture evidence | Supported boundary |
| --- | --- | --- |
| ByteLevel BPE | Existing Qwen/Llama fixtures and pre-6.2 goldens, including a supplementary scalar split across token IDs | NFC/null normalization, ByteLevel prefix-space/regex/trim-offset behavior, BPE merges, ByteLevel decoding |
| Metaspace BPE | `metaspace-bpe.tokenizer.json`: known pieces, unknown symbols, UTF-8 byte fallback | Deterministic dropout-free BPE; declared unknown/fuse/fallback/prefix/suffix/ignore-merges fields |
| Metaspace Unigram | `unigram.tokenizer.json`: competing scored pieces, unknown symbols, UTF-8 byte fallback | Trie-bounded Viterbi and declared `unk_id`; no SentencePiece `Precompiled` charsmap |
| Bert/WordPiece | `wordpiece.tokenizer.json`: case/accent normalization, CJK, punctuation, continuations, unknown, post-processing, truncation and padding | Single-sequence WordPiece/Bert pipeline and fixed padding |
| Added tokens | `added-token.tokenizer.json`: normalized, single-word, left/right whitespace consumption | Longest match then declaration order around normalized and raw input |

The Python binding reports offsets in Unicode code-point positions for these Python strings. The
oracle runner converts each boundary by UTF-8 encoding the preceding substring. Java results are
therefore compared in the independently defined form `[startByte,endByte)` into the original input;
special and padding tokens use `(0,0)`.

Recognized components and field values are accepted only when they affect behavior as implemented.
Unknown types, nonzero BPE dropout, pair/overflow truncation, padding multiples, malformed fallback
tokens, and `Precompiled` normalization fail with their JSON path/field. WordLevel, pair encoding,
arbitrary plugin components, and real Mistral/Gemma/Phi/Mixtral artifact claims remain deferred.
