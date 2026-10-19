#!/usr/bin/env python3
"""Phase 7.3b WP3 opt-in cross-check: multi-frame (t > 1) image RoPE.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0,
Pillow 12.3.0, NumPy 2.5.3, Python 3.12), CPU, single-threaded,
deterministic. It is never invoked by Gradle and writes no golden: it
cross-checks qwen3vl_wp3_prompt.rederive_rope_all -- the independent
re-derivation of the 4.57.6 Qwen3VLModel.get_rope_index image path --
against the real pinned reference method on synthetic image grids with
t = 1, 2, 3 and 5 plus a two-image prompt with a text run between the
images. The pinned WP3 golden is t = 1 only, so its cases never exercise
the per-frame T coordinate (T = base + frame index while H/W restart per
frame); this check covers that path, and the text run between the two
images of the last case exercises per-block starts as well.

The synthetic prompts carry the pinned snapshot's real four vision token
IDs (read from its config.json) plus arbitrary non-colliding text IDs:
get_rope_index reads only config fields and token IDs, so no tokenizer or
images are needed. Each image block has the real chat-template shape --
vision start, the image sentinel run, vision end -- because the reference
counts images as vision-start tokens whose next token is the image
sentinel, and the sentinel run has exactly t * (h/merge) * (w/merge)
tokens, as a real expansion produces. Every case requires exact integer
equality of all three position axes and the mrope delta; one line is
printed per case and the exit code is non-zero if any case mismatches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import qwen3vl_probe as probe
from qwen3vl_wp3_prompt import CONFIG_FIELDS, rederive_rope_all

AXIS_NAMES = ("t", "h", "w")

# (name, segments): a segment is ("text", n) -- n text tokens -- or
# ("image", t, h, w) -- one vision-wrapped image block with grid (t, h, w).
CASES: list[tuple[str, list[tuple]]] = [
    ("single_image_t1", [("text", 3), ("image", 1, 4, 4), ("text", 2)]),
    ("single_image_t2", [("text", 3), ("image", 2, 4, 4), ("text", 2)]),
    ("single_image_t3", [("text", 3), ("image", 3, 4, 6), ("text", 2)]),
    ("single_image_t5", [("text", 3), ("image", 5, 4, 4), ("text", 2)]),
    ("two_images_text_between",
     [("text", 3), ("image", 2, 4, 4), ("text", 4), ("image", 3, 6, 4), ("text", 2)]),
]


def build_ids(segments: list[tuple], tokens: dict, merge: int) -> tuple[list[int], list[list[int]]]:
    """Expand the segments into a synthetic prompt plus its image grids.

    Text token IDs are fresh integers (1000 and up) that can never collide
    with the four vision IDs; the image block is the vision start sentinel,
    the run of t * (h/merge) * (w/merge) image sentinels and the vision end
    sentinel.
    """
    ids: list[int] = []
    grids: list[list[int]] = []
    next_text = 1000
    for segment in segments:
        if segment[0] == "text":
            for _ in range(segment[1]):
                ids.append(next_text)
                next_text += 1
        else:
            _, t, h, w = segment
            grids.append([t, h, w])
            ids.append(tokens["vision_start"])
            ids.extend([tokens["image"]] * (t * (h // merge) * (w // merge)))
            ids.append(tokens["vision_end"])
    return ids, grids


def first_difference(expected: list[int], actual: list[int]) -> str | None:
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            return f"first difference at index {i}: expected {e}, got {a}"
    if len(expected) != len(actual):
        return f"length {len(actual)} != expected {len(expected)}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--dir", type=Path,
                        default=here.parent.parent / ".tier-b" / "qwen3-vl-2b-instruct")
    args = parser.parse_args()
    directory: Path = args.dir
    if not (directory / "config.json").is_file():
        sys.exit(f"no config.json under {directory}; pass --dir to the pinned snapshot")

    probe.gate_versions()
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    config = json.loads((directory / "config.json").read_text())
    tokens = {role: int(config[field]) for role, field in CONFIG_FIELDS.items()}
    merge = int(config["vision_config"]["spatial_merge_size"])

    # get_rope_index reads only config fields, so a namespace carrying the
    # real config exercises the actual method without loading weights (the
    # same construction as qwen3vl_wp3_prompt.main).
    rope_config = Qwen3VLConfig.from_pretrained(str(directory))
    for role, field in CONFIG_FIELDS.items():
        probe.check_facts("config", f"{role}_id", int(getattr(rope_config, field)),
                          tokens[role], "pinned snapshot config drifted from config.json")
    probe.check_facts("config", "spatial_merge_size",
                      int(rope_config.vision_config.spatial_merge_size), merge)
    rope_self = SimpleNamespace(config=rope_config)

    failures = 0
    for name, segments in CASES:
        ids, grids = build_ids(segments, tokens, merge)
        ids_tensor = torch.as_tensor([ids], dtype=torch.long)
        grid_tensor = torch.as_tensor(grids, dtype=torch.long)
        attention_mask = torch.as_tensor([[1] * len(ids)], dtype=torch.long)
        position_ids, rope_deltas = Qwen3VLModel.get_rope_index(
            rope_self, ids_tensor, grid_tensor, None, attention_mask=attention_mask)
        actual = [[int(v) for v in row] for row in position_ids[:, 0].tolist()]
        delta = int(rope_deltas[0][0].item())
        expected = rederive_rope_all(ids, grids, merge, tokens["image"])

        diffs = []
        for axis_name, axis, expected_axis in zip(AXIS_NAMES, actual, expected["position_ids"]):
            difference = first_difference(expected_axis, axis)
            if difference is not None:
                diffs.append(f"axis {axis_name}: {difference}")
        if delta != expected["rope_delta"]:
            diffs.append(f"rope_delta {delta} != expected {expected['rope_delta']}")
        if diffs:
            failures += 1
            print(f"{name}: grids={grids} length={len(ids)} MISMATCH {'; '.join(diffs)}")
        else:
            print(f"{name}: grids={grids} length={len(ids)} delta={delta} OK")

    if failures:
        sys.exit(f"{failures}/{len(CASES)} cases mismatched get_rope_index")
    print(f"all {len(CASES)} cases match get_rope_index exactly")


if __name__ == "__main__":
    main()
