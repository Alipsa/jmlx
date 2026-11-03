#!/usr/bin/env python3
"""Record actual mlx-vlm processor serialization and slow-PIL pixel comparison."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    file = root/"fixtures/vlm-float/processor_config.json"
    config = json.loads(file.read_text())["image_processor"]
    processor = Qwen3VLImageProcessor(**config)
    golden_file = root/"goldens/model-vision.safetensors"
    golden = load_file(golden_file)
    records = []
    for case, image in (("small", "rgb-32x32.png"), ("resize", "rgb-73x45.png"), ("large", "rgb-250x130.png")):
        path = root/"fixtures/images"/image
        output = processor(images=[Image.open(path).convert("RGB")])
        expected = golden["hf."+case+".image0.pixels"]
        assert np.array_equal(output["image_grid_thw"], golden["hf."+case+".image0.grid"])
        error = float(np.max(np.abs(output["pixel_values"]-expected)))
        assert error < 1e-6
        records.append(dict(image=image, image_sha256=digest(path), pixel_max_abs=error))
    args.output.write_text(json.dumps(dict(
        kind="Observed mlx-vlm 0.6.8 saved processor alias; selected slow-PIL contract unchanged",
        source_sha256=digest(__file__), processor_source_sha256=digest(inspect.getfile(Qwen3VLImageProcessor)),
        config_sha256=digest(file), golden_sha256=digest(golden_file),
        raw_row_budget=4096, effective_max_pixels=1048576, cases=records), indent=2)+"\n")


if __name__ == "__main__":
    main()
