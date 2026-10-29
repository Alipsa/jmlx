#!/usr/bin/env python3
"""Phase 7.3c WP3 reference golden: Phi-3.5-Vision prompt expansion.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0,
Pillow 12.3.0, NumPy 2.5.3, Python 3.12), CPU, deterministic. It is never
invoked by Gradle; it writes one golden to tools/hf-reference/goldens/
whose SHA-256 is pinned in tools/hf-reference/provenance.json.

The golden is the executable evidence behind req/plans/phase7-3c-plan.md
section 3 (WP3) and req/plans/phase7-3c-reference-findings.md section 7:

  * the artifact's chat-template rendering through the real
    LlamaTokenizerFast.apply_chat_template(tokenize=False) per case and per
    add_generation_prompt variant (the rendered strings pin the jmlx-jinja
    render of the same template, WP3's second case);
  * the bundled trust-remote-code processor's own
    Phi3VProcessor._convert_images_texts_to_inputs (re.split on
    <|image_\\d+|>, per-chunk fast-tokenizer encodings with the per-encoding
    BOS, marker i expanded to num_img_tokens[i-1] sentinel rows of -i),
    cross-checked against an independent string-level re-derivation;
  * the per-image sentinel destination ranges in the expanded IDs;
  * the tokenizer-structure facts of findings section 7.2 (BPE model flags,
    normalizer, decoder, post-processor, no pre-tokenizer, the literal
    <|image|> added token at 32044) re-asserted from the pinned bytes.

The marker strings are built from character codes (chr(0x3C) + chr(0x7C) +
name + chr(0x7C) + chr(0x3E)) so this source never depends on a literal that
message transport mangles; the per-chunk BOS facts of findings section 7.1
(WP1 golden p3) are re-asserted from the pinned tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import phi3v_probe as probe

TRANSFORMERS_PIN = "4.57.6"
PATTERN = r"<\|image_\d+\|>"


def sentinel(name: str) -> str:
    """Build a <|name|>-shaped marker from character codes."""
    return chr(0x3C) + chr(0x7C) + name + chr(0x7C) + chr(0x3E)


def marker(i: int) -> str:
    return sentinel(f"image_{i}")


def sha256_file(path: Path) -> str:
    return probe.sha256_bytes(path.read_bytes())


def check(field: str, actual, expected, note: str = "") -> None:
    if actual != expected:
        suffix = f" ({note})" if note else ""
        sys.exit(f"wp3 check failed: {field}: {actual!r} != {expected!r}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    parser.add_argument(
        "--dir",
        type=Path,
        default=repo / "jmlx-models" / "src" / "test" / "resources" / "phi3v-tokens",
        help="directory holding the artifact's tokenizer.json and tokenizer_config.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "goldens" / "phi3v-wp3-prompt.json",
        help="golden output path",
    )
    args = parser.parse_args()
    directory = args.dir.resolve()

    import transformers

    check("transformers_version", transformers.__version__, TRANSFORMERS_PIN,
          "run under the pinned hash-locked venv")
    if (sys.version_info[:2]) != (3, 12):
        sys.exit(f"wp3 requires CPython 3.12, got {sys.version_info[:2]}")

    # The two tokenizer files must be byte-identical to the pinned artifact
    # (WP0 inventory); the bundled processor must be the pinned file, which
    # is committed under the image-oracle fixtures.
    wp1 = json.loads((here / "goldens" / "phi3v-wp1-probe.json").read_text())
    for name in ("tokenizer.json", "tokenizer_config.json"):
        check(name, sha256_file(directory / name), wp1["pins"][name]["sha256"],
              "tokenizer fixture drifted from the pinned artifact")
    proc_fixture = (repo / "tools" / "image-oracle" / "fixtures" / "phi3v"
                    / "processing_phi3_v.py")
    check("processing_phi3_v.py", sha256_file(proc_fixture),
          wp1["pins"]["processing_phi3_v.py"]["sha256"],
          "bundled processor fixture drifted from the pinned artifact")

    # Stage the three files for the dynamic-module import.
    staged = Path(tempfile.mkdtemp(prefix="phi3v-wp3-"))
    try:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (staged / name).write_bytes((directory / name).read_bytes())
        (staged / "processing_phi3_v.py").write_bytes(proc_fixture.read_bytes())

        from transformers import LlamaTokenizerFast
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        proc_cls = get_class_from_dynamic_module("processing_phi3_v.Phi3VProcessor",
                                                 str(staged))
        tokenizer = LlamaTokenizerFast.from_pretrained(str(staged))

        class _ImageProcessorStub:
            # Phi3VProcessor.__init__ reads only this attribute; the text
            # path under test consumes the caller-supplied num_img_tokens.
            num_img_tokens = None

        processor = proc_cls(image_processor=_ImageProcessorStub(),
                             tokenizer=tokenizer)

        # Structure facts (findings 7.2): the shape jmlx-tokenizer must
        # cover, re-asserted from the pinned bytes.
        tj = json.loads((directory / "tokenizer.json").read_text())
        tc = json.loads((directory / "tokenizer_config.json").read_text())
        model = tj["model"]
        check("model_type", model["type"], "BPE")
        check("byte_fallback", model["byte_fallback"], True)
        check("fuse_unk", model["fuse_unk"], True)
        check("pre_tokenizer", tj["pre_tokenizer"], None)
        check("normalizer", [n["type"] for n in tj["normalizer"]["normalizers"]],
              ["Prepend", "Replace"])
        check("decoder_types", [d["type"] for d in tj["decoder"]["decoders"]],
              ["Replace", "ByteFallback", "Fuse", "Strip"])
        check("post_processor_type", tj["post_processor"]["type"],
              "TemplateProcessing")
        check("add_bos_token", tc["add_bos_token"], True)
        check("add_eos_token", tc["add_eos_token"], False)
        check("tokenizer_class", tc["tokenizer_class"], "LlamaTokenizer")
        check("bos_id", tokenizer.bos_token_id, 1)
        check("eos_id", tokenizer.eos_token_id, 32000)
        check("literal_image_id", tokenizer.convert_tokens_to_ids(sentinel("image")),
              32044)
        added = {int(t["id"]): t for t in tj["added_tokens"]}
        check("literal_image_content", added[32044]["content"], sentinel("image"))
        check("vocab_size", tokenizer.vocab_size, len(model["vocab"]))
        check("chat_template_on_tokenizer", tokenizer.chat_template,
              tc["chat_template"])

        # Per-chunk BOS facts (findings 7.1, WP1 golden p3 re-asserted):
        # every chunk encodes with its own leading BOS, the empty chunk to
        # exactly the BOS.
        check("chunk_empty", tokenizer("").input_ids, [1])
        check("chunk_hello", tokenizer("hello").input_ids, [1, 22172])
        check("chunk_space_hello", tokenizer(" hello").input_ids,
              [1, 29871, 22172])

        def reference_expand(text: str, num_img_tokens: list[int]) -> dict:
            # The bundled assert reads len(pixel_values) as the image count,
            # so the dummy must have one entry per image.
            out = processor._convert_images_texts_to_inputs(
                images={"pixel_values": [0] * len(num_img_tokens),
                        "image_sizes": [[672, 672]] * len(num_img_tokens),
                        "num_img_tokens": num_img_tokens},
                texts=text)
            ids = out["input_ids"][0].tolist()
            mask = [int(x) for x in out["attention_mask"][0]]
            check(f"attention_mask_len_{len(ids)}", mask, [1] * len(ids),
                  "the reference mask is all ones (sentinels are > -1000000)")
            # Per-image destination ranges: the maximal run of -i starting at
            # the i-th marker's expansion, in prompt order.
            destinations = []
            pos = 0
            for i in range(1, len(num_img_tokens) + 1):
                start = ids.index(-i, pos)
                count = 0
                while ids[start + count] == -i:
                    count += 1
                check(f"destination_{i}", count, num_img_tokens[i - 1])
                destinations.append({"start": start, "count": count})
                pos = start + count
            return {"expanded_ids": ids, "image_destinations": destinations}

        def rederive(text: str, num_img_tokens: list[int]) -> dict:
            """Independent string-level re-derivation: the reference's
            re.split + per-chunk tokenizer + separator-insertion algorithm
            (findings 7.1), re-coded without the processor."""
            chunks = re.split(PATTERN, text)
            tags = re.findall(PATTERN, text)
            image_ids = [int(s.split("|")[1].split("_")[-1]) for s in tags]
            unique = sorted(set(image_ids))
            check("rederive_unique", unique,
                  list(range(1, len(num_img_tokens) + 1)),
                  "marker indices must be exactly 1..n as a set")
            check("rederive_count", len(unique), len(num_img_tokens))
            seps = [[-iid] * num_img_tokens[iid - 1] for iid in image_ids]
            if len(chunks) > len(seps):
                seps.append([])
            ids: list[int] = []
            chunk_ids: list[list[int]] = []
            for chunk, sep in zip(chunks, seps):
                encoded = tokenizer(chunk).input_ids
                chunk_ids.append(encoded)
                ids.extend(encoded)
                ids.extend(sep)
            destinations = []
            pos = 0
            for i in range(1, len(num_img_tokens) + 1):
                start = ids.index(-i, pos)
                destinations.append({"start": start,
                                     "count": num_img_tokens[i - 1]})
                pos = start + num_img_tokens[i - 1]
            return {"expanded_ids": ids, "image_destinations": destinations,
                    "chunks": chunks, "chunk_ids": chunk_ids}

        def case_record(name: str, messages: list[dict],
                        num_img_tokens: list[int]) -> dict:
            record = {"name": name, "messages": messages,
                      "num_img_tokens": num_img_tokens, "variants": {}}
            for gen_prompt in (True, False):
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=gen_prompt)
                ref = reference_expand(rendered, num_img_tokens)
                own = rederive(rendered, num_img_tokens)
                check(f"{name}_genprompt_{gen_prompt}_expanded",
                      own["expanded_ids"], ref["expanded_ids"],
                      "string-level re-derivation disagrees with the bundled processor")
                check(f"{name}_genprompt_{gen_prompt}_destinations",
                      own["image_destinations"], ref["image_destinations"])
                record["variants"][
                    f"add_generation_prompt_{str(gen_prompt).lower()}"] = {
                    "rendered": rendered,
                    "rendered_sha256": hashlib.sha256(
                        rendered.encode("utf-8")).hexdigest(),
                    "chunks": own["chunks"],
                    "chunk_ids": own["chunk_ids"],
                    "expanded_ids": ref["expanded_ids"],
                    "image_destinations": ref["image_destinations"],
                }
            return record

        cases = [
            case_record(
                "text_only",
                [{"role": "user",
                  "content": "What is the capital of France?"}],
                []),
            case_record(
                "single_image",
                [{"role": "user",
                  "content": marker(1) + "\nWhat is shown in this image?"}],
                [757]),
            case_record(
                "two_images_text_between",
                [{"role": "user",
                  "content": (marker(1) + "\nDescribe this image in one sentence. "
                              "Now the second one: " + marker(2) + "\n"
                              "What color is it?")}],
                [757, 457]),
            case_record(
                "trailing_marker",
                [{"role": "user",
                  "content": "Describe the image." + marker(1)}],
                [757]),
            case_record(
                "adjacent_markers",
                [{"role": "user",
                  "content": marker(1) + marker(2) + "\nWhat do both show?"}],
                [757, 457]),
        ]

        golden = {
            "schema": "jmlx-phase7-3c-wp3-phi3v-prompt-v1",
            "generated_by": "tools/hf-reference/phi3v_wp3_prompt.py",
            "environment": probe.env_info(),
            "artifact": {
                "org": "microsoft",
                "repo": "Phi-3.5-vision-instruct",
                "revision": probe.REVISION,
                "file_sha256": {
                    name: wp1["pins"][name]["sha256"]
                    for name in ("tokenizer.json", "tokenizer_config.json",
                                 "processing_phi3_v.py")
                },
            },
            "tokenizer_facts": {
                "model": {
                    "type": model["type"],
                    "byte_fallback": model["byte_fallback"],
                    "fuse_unk": model["fuse_unk"],
                    "vocab_size": len(model["vocab"]),
                    "merges_count": len(model["merges"]),
                },
                "added_tokens_count": len(tj["added_tokens"]),
                "vocab_size": tokenizer.vocab_size,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "literal_image_token_id": 32044,
                "add_bos_token": tc["add_bos_token"],
                "add_eos_token": tc["add_eos_token"],
                "normalizer": tj["normalizer"],
                "pre_tokenizer": tj["pre_tokenizer"],
                "decoder": tj["decoder"],
                "post_processor": tj["post_processor"],
                "chat_template": tc["chat_template"],
            },
            "cases": cases,
        }
        args.out.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    finally:
        import shutil

        shutil.rmtree(staged, ignore_errors=True)


if __name__ == "__main__":
    main()
