#!/usr/bin/env python3
"""Builds the release-smoke tokenizer fixture (tokenizer.json + tokenizer_config.json).

Hand-written, NOT Hugging Face-generated: see PROVENANCE.md. Deterministic; rerunning must leave the
committed files unchanged.
"""
import json
import pathlib

VOCAB_SIZE = 128  # must equal the paired checkpoint's config.json vocab_size
SPECIALS = [(0, "<pad>"), (1, "<s>"), (2, "</s>")]  # BOS 1 / EOS 2 / pad 0, as the checkpoint


def byte_level_map():
    """GPT-2's printable stand-ins for raw bytes (so a space is U+0120, a newline U+010A)."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping, extra = {}, 0
    for b in range(256):
        if b in keep:
            mapping[b] = chr(b)
        else:
            mapping[b] = chr(256 + extra)
            extra += 1
    return mapping


def main():
    bl = byte_level_map()
    pieces = [bl[b] for b in range(0x20, 0x7F)] + [bl[0x0A]]  # printable ASCII + newline (96)
    # Common bigrams (merge results) fill the remaining ids; both halves are single pieces.
    bigrams = [
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
        "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
        "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
    ]
    vocab, merges = {}, []
    next_id = len(SPECIALS)
    for piece in pieces:
        vocab[piece] = next_id
        next_id += 1
    for pair in bigrams:
        if next_id >= VOCAB_SIZE:
            break
        vocab[pair] = next_id
        merges.append(f"{pair[0]} {pair[1]}")
        next_id += 1
    assert next_id == VOCAB_SIZE, next_id
    assert sorted(vocab.values()) == list(range(len(SPECIALS), VOCAB_SIZE))

    byte_level = {"add_prefix_space": False, "trim_offsets": False, "use_regex": True}
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": c,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i, c in SPECIALS
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", **byte_level},
        "post_processor": {"type": "ByteLevel", **byte_level},
        "decoder": {"type": "ByteLevel", **byte_level},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    template = (
        "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}[assistant] {% endif %}"
    )
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "chat_template": template,
    }
    here = pathlib.Path(__file__).parent
    (here / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n")
    (here / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
