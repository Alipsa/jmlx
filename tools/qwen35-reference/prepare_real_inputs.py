#!/usr/bin/env python3
"""Prepare digest-pinned slow-PIL inputs and compare the declared fast backend."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLImageProcessorPil, Qwen2VLImageProcessor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, local_files_only=True)
    # Explicit cap: 4096 raw rows * (16*16) pixels; keep the artifact's minimum.
    config = dict(patch_size=16, temporal_patch_size=2, merge_size=2,
                  image_mean=[.5]*3, image_std=[.5]*3,
                  size={"shortest_edge": 65536, "longest_edge": 1048576})
    slow = Qwen2VLImageProcessorPil(**config)
    fast = Qwen2VLImageProcessor(**config)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        slow_result = slow(images=[image], return_tensors="np")
        fast_result = fast(images=[image], return_tensors="np")
    pixels = slow_result["pixel_values"]
    assert pixels.shape[0] <= 4096
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pixel_path = args.output.with_suffix(".patches.npy")
    np.save(pixel_path, pixels, allow_pickle=False)
    cases = []
    for image in (False, True):
        content = "<|vision_start|><|image_pad|><|vision_end|>Describe this image briefly." if image else "What is two plus two? Answer briefly."
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if image:
            rows = int(np.prod(slow_result["image_grid_thw"][0]) // 4)
            rendered = rendered.replace("<|image_pad|>", "<|image_pad|>" * rows)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        case = dict(name="image" if image else "text", rendered_prompt=rendered,
                    input_ids=[ids], forced_decode_id=tokenizer.encode("The", add_special_tokens=False)[0])
        if image:
            case.update(pixel_values_file=pixel_path.name,
                        pixel_values_sha256=digest(pixel_path),
                        image_grid_thw=slow_result["image_grid_thw"].tolist())
        cases.append(case)
    assert np.array_equal(slow_result["image_grid_thw"], fast_result["image_grid_thw"])
    record = dict(kind="WP0 shared reference inputs", cases=cases,
                  image_sha256=digest(args.image), config=config,
                  slow_class=type(slow).__name__, fast_class=type(fast).__name__,
                  slow_source_sha256=digest(inspect.getfile(type(slow))),
                  fast_source_sha256=digest(inspect.getfile(type(fast))),
                  prepare_source_sha256=digest(__file__),
                  fast_slow_pixel_max_abs=float(np.max(np.abs(pixels-fast_result["pixel_values"]))),
                  fast_slow_pixel_equal=bool(np.array_equal(pixels, fast_result["pixel_values"])))
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
