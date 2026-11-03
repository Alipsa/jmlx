#!/usr/bin/env python3
"""Unmodified HF slow-PIL processor, vision tower and image/text hybrid-stack goldens."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2VLImageProcessorPil, Qwen3VLProcessor, Qwen3VLVideoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from probe_hf import tiny_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    torch.manual_seed(20261106)
    variant = root/"fixtures/hf-grid48"
    if variant.exists():
        raise ValueError("Preserve the existing grid48 fixture before intentional regeneration")
    config = tiny_config()
    config.vision_config.num_position_embeddings = 2304
    Qwen3_5ForConditionalGeneration(config).eval().bfloat16().float().save_pretrained(variant)
    for filename in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"):
        shutil.copyfile(root/"fixtures/hf"/filename, variant/filename)
    image_dir = root/"fixtures/images"
    image_dir.mkdir(exist_ok=True)
    images = []
    for width, height in ((32, 32), (73, 45), (250, 130)):
        y, x = np.mgrid[:height, :width]
        pixels = np.stack(((7*x+11*y) % 256, (13*x+5*y) % 256, (19*x+3*y) % 256), -1).astype(np.uint8)
        image = Image.fromarray(pixels)
        path = image_dir/f"rgb-{width}x{height}.png"
        image.save(path)
        images.append((path, image))
    arrays, records = {}, []
    slow = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2,
                                  image_mean=[.5]*3, image_std=[.5]*3,
                                  size={"shortest_edge": 1024, "longest_edge": 1048576})
    processor = Qwen3VLProcessor(image_processor=slow,
        tokenizer=AutoTokenizer.from_pretrained(root/"fixtures/hf", local_files_only=True),
        video_processor=Qwen3VLVideoProcessor())
    with torch.no_grad():
        for fixture in ("hf", "hf-grid48"):
            directory = root/"fixtures"/fixture
            model = Qwen3_5ForConditionalGeneration.from_pretrained(directory,
                dtype=torch.float32, attn_implementation="eager").eval()
            vision = model.model.visual
            for name, selected in (("small", [0]), ("resize", [1]), ("large", [2]), ("multi", [0, 1])):
                stem = fixture+"."+name
                chosen = [images[i][1] for i in selected]
                text = "<bos>"+"<|vision_start|><|image_pad|><|vision_end|>A"*len(chosen)+"B"
                original_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                encoded = processor(text=[text], images=chosen, return_tensors="pt", add_special_tokens=False)
                ids, pixels, grids = encoded.input_ids, encoded.pixel_values, encoded.image_grid_thw
                types = (ids == 120).to(torch.int32)
                positions, delta = model.model.get_rope_index(ids, mm_token_type_ids=types, image_grid_thw=grids)
                arrays[stem+".ids"] = ids.to(torch.int32)
                arrays[stem+".positions"] = positions.to(torch.int32)
                arrays[stem+".delta"] = delta.to(torch.int32)
                offset = 0
                image_records = []
                for i, (index, grid) in enumerate(zip(selected, grids)):
                    rows = int(grid.prod())
                    one_pixels = pixels[offset:offset+rows]
                    one_grid = grid[None]
                    offset += rows
                    p = stem+f".image{i}"
                    indices, weights = hf.get_vision_interpolation_indices_and_weights(one_grid,
                        num_grid_per_side=vision.num_grid_per_side, mode=vision.interpolation_mode,
                        align_corners=vision.interpolation_align_corners, spatial_merge_size=2, kwargs={})
                    arrays[p+".position"] = (vision.pos_embed(indices)*weights[:, :, None]).sum(1)
                    arrays[p+".patch"] = vision.patch_embed(one_pixels)
                    captured = []
                    handle = vision.blocks[0].register_forward_hook(lambda module, args, output: captured.append(output.clone()))
                    visual = vision(one_pixels, one_grid)
                    handle.remove()
                    arrays[p+".block0"] = captured[0]
                    arrays[p+".merged"] = visual.pooler_output
                    arrays[p+".pixels"] = one_pixels
                    arrays[p+".grid"] = one_grid.to(torch.int32)
                    image_records.append(dict(file=str(images[index][0].relative_to(root)),
                                              sha256=digest(images[index][0])))
                output = model(input_ids=ids, pixel_values=pixels, image_grid_thw=grids,
                               mm_token_type_ids=types, use_cache=True)
                arrays[stem+".logits"] = output.logits
                cache = output.past_key_values
                for step, token in enumerate((6, 7)):
                    current = torch.tensor([[token]], dtype=torch.int32)
                    output = model(input_ids=current, past_key_values=cache, use_cache=True)
                    arrays[stem+f".decode{step}.logits"] = output.logits
                    ids = torch.cat((ids, current), 1)
                    fresh = model(input_ids=ids, pixel_values=pixels, image_grid_thw=grids,
                                  mm_token_type_ids=(ids == 120).to(torch.int32), use_cache=False).logits[:, -1:]
                    arrays[stem+f".decode{step}.fresh_logits"] = fresh
                    assert (fresh-output.logits).abs().max().item() < 1e-4
                prefix_ids = encoded.input_ids
                output = model(input_ids=prefix_ids, pixel_values=pixels, image_grid_thw=grids,
                               mm_token_type_ids=types, use_cache=True)
                cache = output.past_key_values
                greedy, margins = [], []
                for step in range(3):
                    last = output.logits[:, -1]
                    arrays[stem+f".greedy{step}.logits"] = last
                    greedy.append(int(last.argmax()))
                    top = last.sort().values
                    margins.append(float(top[0, -1]-top[0, -2]))
                    if step < 2:
                        output = model(input_ids=torch.tensor([[greedy[-1]]], dtype=torch.int32),
                                       past_key_values=cache, use_cache=True)
                assert min(margins) > 2e-4+1e-5
                records.append(dict(name=stem, fixture=fixture, images=image_records,
                    original_ids=original_ids,
                    greedy_ids=greedy, margins=margins))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().clone() for k, v in arrays.items()}, args.output,
              metadata={"reference": "unmodified HF CPU FLOAT32 slow PIL; no DeepStack"})
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        source_sha256=digest(__file__), hf_source_sha256=digest(hf.__file__),
        safetensors_sha256=digest(args.output), seed=20261106, epsilon=1e-4,
        raw_row_budget=4096, effective_max_pixels=1048576, cases=records), indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
