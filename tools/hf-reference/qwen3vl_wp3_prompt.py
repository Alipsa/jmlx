#!/usr/bin/env python3
"""Phase 7.3b WP3 reference golden: Qwen3-VL-2B-Instruct prompt expansion and
three-axis RoPE positions.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0,
Pillow 12.3.0, NumPy 2.5.3, Python 3.12), CPU, single-threaded, deterministic.
It is never invoked by Gradle; it writes one golden to
tools/hf-reference/goldens/ whose SHA-256 is pinned in
tools/hf-reference/provenance.json.

The golden is the executable evidence behind req/plans/phase7-3b-plan.md
section 5 (WP3) and req/plans/phase7-3b-reference-findings.md section 6.8:

  * chat-template rendering through the artifact's processor-owned template
    (add_generation_prompt False and True per case), covering text-only,
    single/multiple images, adjacent images, text between images and a
    system/multi-turn conversation;
  * unexpanded prompt IDs from the real Qwen2TokenizerFast;
  * processor expansion from the real Qwen3VLProcessor.__call__ (each image's
    grid_thw.prod() // merge_size**2 tokens replace that image's single image
    sentinel -- the chat template renders each image as the vision start
    sentinel, one image sentinel and the vision end sentinel -- in prompt
    order), cross-checked against an independent string-level re-expansion
    that re-tokenizes;
  * three-axis position ids and the mrope delta from the real
    Qwen3VLModel.get_rope_index, called against the real config with no
    weight loading (the method reads only config fields), cross-checked
    against an independent multi-image re-derivation;
  * the WP1 probe invariants (256x256 -> length 82 / delta -56, 64x32 ->
    length 90 / delta -60) re-asserted from the pinned WP1 golden.

The artifact's four vision token IDs are read from the outer config.json and
cross-checked against every tokenization surface: tokenizer_config.json's
added_tokens_decoder (id -> content), tokenizer.json's added_tokens, the fast
tokenizer's convert_ids_to_tokens and decode, and the processor's four
image/video/vision-start/vision-end token attributes. The sentinel strings
are built from character codes (the source never contains a literal with
them) and are verified against the WP1 golden's rendered single-image
prompt, which must contain exactly one image sentinel wrapped by the vision
start and end sentinels -- three tokens when tokenized.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import qwen3vl_probe as probe

def _sentinel(name: str) -> str:
    """Build the artifact's vision sentinel: marker, bare token name, marker.

    The marker characters are built from character codes so the source file
    never depends on a literal containing them.
    """
    return chr(0x3C) + chr(0x7C) + name + chr(0x7C) + chr(0x3E)


IMAGE_TOKEN = _sentinel("image_pad")
CONTRACT_CONTENTS = {
    "image": _sentinel("image_pad"),
    "video": _sentinel("video_pad"),
    "vision_start": _sentinel("vision_start"),
    "vision_end": _sentinel("vision_end"),
}
CONFIG_FIELDS = {
    "image": "image_token_id",
    "video": "video_token_id",
    "vision_start": "vision_start_token_id",
    "vision_end": "vision_end_token_id",
}


def rederive_rope_all(
    input_ids: list[int], grids: list[list[int]], merge: int, image_token_id: int
) -> dict:
    """Independent re-derivation of the 4.57.6 get_rope_index image path.

    Encodes the exact algorithm for N images in prompt order: text runs get
    all three axes equal (arange + st_idx); an image's llm grid is
    (t, h/merge, w/merge) and its t * lh * lw tokens get T = base + f for
    every token of frame f, while H = base + h_idx (rows of llm_w) and
    W = base + w_idx restart per frame, with base = st_idx + text_len -- the
    reference's (t, h, w) index stack flattened t-major (see the arange/expand
    construction at modular_qwen3_vl.py:939-943, shared with the video path);
    st advances past the whole run (t * lh * lw); every block starts at the
    previous block's maximum plus one; the delta is the largest coordinate
    plus one minus the sequence length.
    """
    positions: list[list[int]] = [[], [], []]
    st = 0
    for t, h, w in grids:
        lt, lh, lw = t, h // merge, w // merge
        ed = input_ids.index(image_token_id, st)
        text_len = ed - st
        st_idx = max(max(axis) for axis in positions) + 1 if positions[0] else 0
        for i in range(text_len):
            for axis in positions:
                axis.append(st_idx + i)
        base = st_idx + text_len
        for frame in range(lt):
            for hi in range(lh):
                for wi in range(lw):
                    positions[0].append(base + frame)
                    positions[1].append(base + hi)
                    positions[2].append(base + wi)
        st = ed + lt * lh * lw
    if st < len(input_ids):
        st_idx = max(max(axis) for axis in positions) + 1 if positions[0] else 0
        for i in range(len(input_ids) - st):
            for axis in positions:
                axis.append(st_idx + i)
    if not positions[0]:
        return {"position_ids": [[], [], []], "rope_delta": 0}
    max_pos = max(max(a) for a in positions)
    return {"position_ids": positions, "rope_delta": max_pos + 1 - len(input_ids)}


def expand_text(rendered: str, merged_counts: list[int]) -> str:
    """The reference's per-image single-replacement loop, in prompt order:
    replace the first remaining image sentinel with n copies, per image. The
    search offset skips past each inserted run, so the copies themselves are
    never mistaken for the next original placeholder."""
    result = rendered
    offset = 0
    for count in merged_counts:
        start = result.index(IMAGE_TOKEN, offset)
        result = result[:start] + IMAGE_TOKEN * count + result[start + len(IMAGE_TOKEN):]
        offset = start + count * len(IMAGE_TOKEN)
    return result


def image_token_runs(ids: list[int], image_token_id: int) -> list[int]:
    runs: list[int] = []
    i = 0
    while i < len(ids):
        if ids[i] == image_token_id:
            j = i
            while j < len(ids) and ids[j] == image_token_id:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return runs


def build_images(workdir: Path) -> dict:
    paths = probe.build_test_images(workdir)
    from PIL import Image

    px = bytearray()
    for y in range(64):
        for x in range(128):
            px += bytes((x, 255 - y, (x ^ y) & 0xFF))
    path = workdir / "rgb-128x64.png"
    Image.frombytes("RGB", (128, 64), px).save(path, format="PNG")
    return {name: paths[name] for name in ("rgb_256x256", "rgb_64x32")} | {"rgb_128x64": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--dir", type=Path, default=here.parent.parent / ".tier-b" / "qwen3-vl-2b-instruct")
    parser.add_argument("--out", type=Path, default=here / "goldens" / "qwen3-vl-wp3-prompt.json")
    args = parser.parse_args()
    directory: Path = args.dir
    if not (directory / "config.json").is_file():
        sys.exit(f"no config.json under {directory}; pass --dir to the pinned snapshot")

    environment = probe.gate_versions()
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # The WP1 probe golden is already SHA-pinned in provenance.json, so it is
    # a trustworthy source for the artifact digests and the probe invariants
    # this script re-asserts instead of re-typing them.
    wp1 = json.loads((here / "goldens" / "qwen3-vl-2b-wp1-probe.json").read_text())
    file_digests = {f["path"]: f["sha256"] for f in wp1["artifact"]["files"]}
    for name, expected in sorted(file_digests.items()):
        probe.check_facts(
            "artifact", name, probe.sha256_file(directory / name), expected,
            "staged artifact file drifted from the pinned WP1 golden")

    config = json.loads((directory / "config.json").read_text())
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())

    tokens = {role: config[field] for role, field in CONFIG_FIELDS.items()}
    # Decode-direction cross-checks: every tokenization surface must agree
    # with the outer config on all four roles (WP3 findings 6.8).
    decoder = {int(k): v["content"] for k, v in tokenizer_config["added_tokens_decoder"].items()}
    for role, content in CONTRACT_CONTENTS.items():
        probe.check_facts("tokens", f"{role}_content", decoder.get(tokens[role]), content)
    added = {int(t["id"]): t["content"] for t in
             json.loads((directory / "tokenizer.json").read_text())["added_tokens"]}
    for role, content in CONTRACT_CONTENTS.items():
        probe.check_facts("tokens", f"{role}_added_tokens", added.get(tokens[role]), content)

    from PIL import Image
    from transformers import Qwen2TokenizerFast
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    workdir = Path(tempfile.mkdtemp(prefix="qwen3vl-wp3-"))
    try:
        images = build_images(workdir)

        preprocessor = json.loads((directory / "preprocessor_config.json").read_text())
        # Explicit min/max pixels: the 4.57.6 slow-processor bug (findings 4.2)
        # ignores the config's size dict, so the probe and this script pass
        # the artifact's declared values explicitly.
        img_proc = Qwen2VLImageProcessor.from_pretrained(
            str(directory),
            min_pixels=int(preprocessor["size"]["shortest_edge"]),
            max_pixels=int(preprocessor["size"]["longest_edge"]))
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(directory))
        video_proc = Qwen3VLVideoProcessor.from_pretrained(str(directory))

        class Wp3Processor(Qwen3VLProcessor):
            def check_argument_for_proper_class(self, name, arg):
                if name == "video_processor":
                    return Qwen3VLVideoProcessor
                return super().check_argument_for_proper_class(name, arg)

        processor = Wp3Processor(image_processor=img_proc, tokenizer=tokenizer,
                                 video_processor=video_proc,
                                 chat_template=tokenizer_config["chat_template"])
        for role, attr in (("image", "image_token"), ("video", "video_token"),
                           ("vision_start", "vision_start_token"),
                           ("vision_end", "vision_end_token")):
            probe.check_facts("tokens", f"{role}_processor_attr",
                              getattr(processor, attr), CONTRACT_CONTENTS[role])
        # Encode direction through the fast tokenizer: the outer-config ids
        # must convert back to the exact sentinel strings.
        for role, content in CONTRACT_CONTENTS.items():
            probe.check_facts("tokens", f"{role}_convert_ids",
                              tokenizer.convert_ids_to_tokens(tokens[role]), content)
            probe.check_facts("tokens", f"{role}_decode",
                              tokenizer.decode([tokens[role]]), content)

        # WP1 rendered-placeholder consistency (hex-list evidence only): the
        # WP1 golden's rendered single-image prompt must contain exactly one
        # image sentinel wrapped by the vision start and end sentinels, its
        # bytes must equal the script's expected image sentinel, and the
        # wrapped span must tokenize to the three vision ids from the outer
        # config.
        wp1_rendered = wp1["processor"]["prompt_template"]["rendered"]
        wrapped = (CONTRACT_CONTENTS["vision_start"] + IMAGE_TOKEN
                   + CONTRACT_CONTENTS["vision_end"])
        probe.check_facts("sentinel", "wp1_wrapped_count",
                          wp1_rendered.count(wrapped), 1,
                          "WP1 rendered prompt does not wrap exactly one image sentinel")
        start_len = len(CONTRACT_CONTENTS["vision_start"])
        offset = wp1_rendered.index(wrapped)
        inner = wp1_rendered[offset + start_len:offset + start_len + len(IMAGE_TOKEN)]
        expected_hex = [hex(ord(c)) for c in IMAGE_TOKEN]
        probe.check_facts("sentinel", "wp1_placeholder_hex",
                          [hex(ord(c)) for c in inner], expected_hex,
                          "WP1 rendered placeholder != expected image sentinel")
        probe.check_facts("sentinel", "wrapped_tokens",
                          list(tokenizer.encode(wrapped, add_special_tokens=False)),
                          [tokens["vision_start"], tokens["image"], tokens["vision_end"]])
        # Display-mangling-safe evidence for the token contract: hex lists,
        # never the sentinel strings themselves.
        for role in ("image", "video", "vision_start", "vision_end"):
            print(f"{role} sentinel hex: {[hex(ord(c)) for c in CONTRACT_CONTENTS[role]]}")

        # get_rope_index reads only config fields, so a namespace carrying the
        # real config exercises the actual method without loading 4 GB of
        # weights.
        rope_config = Qwen3VLConfig.from_pretrained(str(directory))
        for role, field in CONFIG_FIELDS.items():
            probe.check_facts("config", f"{role}_id", getattr(rope_config, field), tokens[role])
        rope_self = SimpleNamespace(config=rope_config)

        def run_variant(name: str, messages: list[dict], image_names: list[str],
                        add_generation_prompt: bool) -> dict:
            rendered = processor.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=False)
            unexpanded = list(tokenizer.encode(rendered, add_special_tokens=False))
            if image_names:
                pil_images = []
                for image_name in image_names:
                    image = Image.open(images[image_name])
                    try:
                        pil_images.append(image.convert("RGB"))
                    finally:
                        image.close()
                batch = processor(text=[rendered], images=pil_images)
            else:
                batch = processor(text=[rendered])
            expanded = [int(v) for v in batch["input_ids"][0]]
            mask = [int(v) for v in batch["attention_mask"][0]]
            probe.check_facts(name, "attention_mask", mask, [1] * len(expanded),
                              "padding in the golden")
            grid_raw = batch.get("image_grid_thw")
            grid_thw = (None if grid_raw is None
                        else [list(int(v) for v in row) for row in grid_raw])
            probe.check_facts(name, "grid_count",
                              None if grid_thw is None else len(grid_thw),
                              None if not image_names else len(image_names))
            grids = grid_thw or []
            merged_counts = [(t * h * w) // img_proc.merge_size**2 for (t, h, w) in grids]

            # Each placeholder becomes one contiguous run of its image's merged
            # count, in prompt order.
            probe.check_facts(name, "image_runs",
                              image_token_runs(expanded, tokens["image"]), merged_counts,
                              "image token runs != merged counts in prompt order")
            # The reference's string-level expansion re-tokenizes to the same IDs.
            re_expanded = list(tokenizer.encode(expand_text(rendered, merged_counts),
                                                add_special_tokens=False))
            probe.check_facts(name, "reexpansion_match", re_expanded, expanded,
                              "string-level re-expansion != processor expansion")

            geometries = []
            for image_name, (t, h, w) in zip(image_names, grids):
                width_in, height_in = Image.open(images[image_name]).size
                probe.check_facts(name, f"grid_{image_name}_h_div", h % img_proc.merge_size, 0)
                probe.check_facts(name, f"grid_{image_name}_w_div", w % img_proc.merge_size, 0)
                geometries.append({
                    "input_width": width_in,
                    "input_height": height_in,
                    "resized_width": w * img_proc.patch_size,
                    "resized_height": h * img_proc.patch_size,
                    "grid_thw": [t, h, w],
                    "raw_patch_rows": h * w,
                    "merged_row_count": (t * h * w) // img_proc.merge_size**2,
                    "patch_stride": 3 * img_proc.temporal_patch_size
                                    * img_proc.patch_size * img_proc.patch_size,
                })

            ids_tensor = torch.as_tensor([expanded], dtype=torch.long)
            grid_tensor = (None if grid_thw is None
                           else torch.as_tensor(grid_thw, dtype=torch.long))
            attention_mask = torch.as_tensor([[1] * len(expanded)], dtype=torch.long)
            position_ids, rope_deltas = Qwen3VLModel.get_rope_index(
                rope_self, ids_tensor, grid_tensor, None, attention_mask=attention_mask)
            actual = [[int(v) for v in row] for row in position_ids[:, 0].tolist()]
            delta = int(rope_deltas[0][0].item())
            # The golden contains only t = 1 (single-frame) image grids, so
            # rederive_rope_all's multi-frame (t > 1) coordinates are verified
            # by logic against the pinned reference's get_rope_index, not by
            # golden coverage.
            expected = rederive_rope_all(expanded, grids, img_proc.merge_size, tokens["image"])
            probe.check_facts(name, "position_ids", actual, expected["position_ids"],
                              "get_rope_index != independent re-derivation")
            probe.check_facts(name, "rope_delta", delta, expected["rope_delta"])

            return {
                "rendered": rendered,
                "rendered_sha256": probe.sha256_bytes(rendered.encode("utf-8")),
                "unexpanded_ids": unexpanded,
                "images": geometries,
                "expanded_ids": expanded,
                "positions": {"t": actual[0], "h": actual[1], "w": actual[2]},
                "delta": delta,
            }

        def image(name: str) -> dict:
            return {"type": "image"}

        def text(value: str) -> dict:
            return {"type": "text", "text": value}

        case_specs = [
            ("text_only",
             [{"role": "user", "content": "Hello, world."}],
             []),
            ("single_image_256x256",
             [{"role": "user", "content": [
                 image("rgb_256x256"),
                 text("Describe the image in one short sentence.")]}],
             ["rgb_256x256"]),
            ("single_image_64x32",
             [{"role": "user", "content": [
                 image("rgb_64x32"),
                 text("Describe the image in one short sentence.")]}],
             ["rgb_64x32"]),
            ("image_after_text",
             [{"role": "user", "content": [
                 text("Look at this picture and name it."),
                 image("rgb_256x256")]}],
             ["rgb_256x256"]),
            ("text_between_images",
             [{"role": "user", "content": [
                 text("Compare:"),
                 image("rgb_256x256"),
                 text(" with this one:"),
                 image("rgb_64x32"),
                 text(" and tell me which is wider.")]}],
             ["rgb_256x256", "rgb_64x32"]),
            ("two_images_adjacent",
             [{"role": "user", "content": [
                 text("Two pictures:"),
                 image("rgb_256x256"),
                 image("rgb_128x64"),
                 text(" done.")]}],
             ["rgb_256x256", "rgb_128x64"]),
            ("three_images",
             [{"role": "user", "content": [
                 image("rgb_256x256"),
                 image("rgb_64x32"),
                 image("rgb_128x64"),
                 text("Describe all three.")]}],
             ["rgb_256x256", "rgb_64x32", "rgb_128x64"]),
            ("system_multi_turn",
             [{"role": "system", "content": "You are a concise vision assistant."},
              {"role": "user", "content": [
                  image("rgb_64x32"),
                  text("What is in this picture?")]},
              {"role": "assistant", "content": "A small rectangle."},
              {"role": "user", "content": "And its orientation?"}],
             ["rgb_64x32"]),
        ]

        cases = []
        for case_name, messages, image_names in case_specs:
            entry = {
                "name": case_name,
                "messages": messages,
                "variants": {
                    flag_name: run_variant(
                        f"{case_name}/{flag_name}", messages, image_names,
                        add_generation_prompt=add_generation_prompt)
                    for flag_name, add_generation_prompt in (
                        ("add_generation_prompt_false", False),
                        ("add_generation_prompt_true", True))
                },
            }
            cases.append(entry)

        # WP1 probe invariants, re-asserted from the pinned WP1 golden rather
        # than re-typed: the 256x256 case must reproduce its rendered prompt,
        # expanded length and delta, and the 64x32 case its length and delta.
        by_name = {c["name"]: c for c in cases}
        wp1_prompt_sha = wp1["processor"]["prompt_template"]["sha256"]
        wp1_256 = wp1["model"]["cases"]["rgb_256x256"]
        wp1_64 = wp1["model"]["cases"]["rgb_64x32"]
        v = by_name["single_image_256x256"]["variants"]["add_generation_prompt_true"]
        probe.check_facts("wp1", "256x256_rendered_sha", v["rendered_sha256"], wp1_prompt_sha)
        probe.check_facts("wp1", "256x256_expanded_length", len(v["expanded_ids"]),
                          wp1["processor"]["cases"]["rgb_256x256"]["prompt_length_after_expansion"])
        starts = [i for i, x in enumerate(v["expanded_ids"]) if x == tokens["image"]]
        probe.check_facts("wp1", "256x256_image_span", [starts[0], starts[-1]],
                          wp1_256["sequence"]["image_token_span"])
        probe.check_facts("wp1", "256x256_delta", v["delta"], wp1_256["rope"]["delta"])
        v2 = by_name["single_image_64x32"]["variants"]["add_generation_prompt_true"]
        starts2 = [i for i, x in enumerate(v2["expanded_ids"]) if x == tokens["image"]]
        probe.check_facts("wp1", "64x32_image_span", [starts2[0], starts2[-1]],
                          wp1_64["sequence"]["image_token_span"])
        probe.check_facts("wp1", "64x32_expanded_length", len(v2["expanded_ids"]),
                          wp1["processor"]["cases"]["rgb_64x32"]["prompt_length_after_expansion"])
        probe.check_facts("wp1", "64x32_delta", v2["delta"], wp1_64["rope"]["delta"])

        result = {
            "schema": "jmlx-phase7-3b-wp3-qwen3vl-prompt-v1",
            "generated_by": "tools/hf-reference/qwen3vl_wp3_prompt.py",
            "environment": environment,
            "artifact": {
                "org": probe.ORG,
                "repo": probe.REPO,
                "revision": probe.REVISION,
                "file_sha256": {k: probe.sha256_file(directory / k)
                                for k in sorted(file_digests)},
            },
            "tokens": tokens,
            "merge_size": img_proc.merge_size,
            "patch_size": img_proc.patch_size,
            "temporal_patch_size": img_proc.temporal_patch_size,
            "cases": cases,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
