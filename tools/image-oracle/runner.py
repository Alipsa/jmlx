#!/usr/bin/env python3
"""Runner for the jmlx image oracle.

References the pinned slow PIL processor (transformers 4.57.6 `Idefics3ImageProcessor`
over Pillow 12.3.0 / numpy 2.5.3) for every committed fixture and either regenerates
the canonical `*.expected.json` files (`--generate-all`) or verifies the committed ones
(`--verify-all`). It loads only committed local files and verifies every fixture source
against `provenance.json` before use.
"""
import argparse
import base64
import difflib
import hashlib
import json
import math
import os
from pathlib import Path


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_fixture(root: Path, relative: str, provenance: dict) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise SystemExit(f"image fixture must be a committed local file: {path}")
    expected = provenance["fixtureSources"].get(relative)
    if expected is None:
        raise SystemExit(f"image fixture has no provenance digest: {relative}")
    if source_digest(path) != expected:
        raise SystemExit(f"image fixture digest differs from provenance: {relative}")
    return path


def uint8_base64(arr) -> str:
    import numpy as np

    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def float32_be_base64(arr) -> str:
    import numpy as np

    # IEEE-754 float32 bits, big-endian: the committed expected files are byte-stable
    # across host endiannes and decode in Java with (b0<<24)|(b1<<16)|(b2<<8)|b3.
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype=np.float32).astype(">f4").tobytes()
    ).decode("ascii")


def reference_decode(path: Path):
    """The pinned raw-PIL reference decode: open -> load(), keep original mode, then the
    processor's convert_to_rgb math: RGBA/LA composite over white via its alpha_composite step,
    every other mode a plain convert("RGB"). A palette tRNS chunk is dropped before the convert:
    the pinned processor rebuilds the palette from getpalette() (RGB only), so its
    info["transparency"] never survives - and raw PIL's PngImageFile.convert would apply the
    tRNS, so the key must be removed first. No exif_transpose, no ICC/gAMA management."""
    from PIL import Image

    image = Image.open(path)
    image.load()
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        if image.mode == "P" and "transparency" in image.info:
            del image.info["transparency"]
        rgb = image.convert("RGB")
    import numpy as np

    return np.asarray(rgb, dtype=np.uint8)


def sample_regions(arr) -> list:
    """Five deterministic 64x64 diagnostic patches (corners + center) of a large image."""
    import numpy as np

    height, width = int(arr.shape[0]), int(arr.shape[1])
    size = min(64, height, width)
    regions = [
        (0, 0),
        (width - size, 0),
        (0, height - size),
        (width - size, height - size),
        ((width - size) // 2, (height - size) // 2),
    ]
    samples = []
    for x0, y0 in regions:
        patch = np.ascontiguousarray(arr[y0 : y0 + size, x0 : x0 + size], dtype=np.uint8)
        samples.append(
            {
                "pixelsBase64": base64.b64encode(patch.tobytes()).decode("ascii"),
                "x1": x0 + size,
                "x0": x0,
                "y1": y0 + size,
                "y0": y0,
            }
        )
    return samples


def run_decode(input_path: Path, provenance: dict) -> dict:
    import numpy as np

    root = input_path.parent.resolve()
    cases = []
    for case in json.loads(input_path.read_text(encoding="utf-8"))["cases"]:
        fixture = verify_fixture(root, case["image"], provenance)
        if case.get("expectedRejected"):
            cases.append({"expectedRejected": True, "name": case["name"]})
            continue
        arr = reference_decode(fixture)
        record = {"height": int(arr.shape[0]), "name": case["name"], "width": int(arr.shape[1])}
        if case.get("shaOnly"):
            # Multi-megapixel limit fixtures: the full decode is pinned by SHA-256 (a match
            # is a byte-exact full-image comparison) with small diagnostic patches recorded.
            flat = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
            record["pixelSha256"] = hashlib.sha256(flat).hexdigest()
            record["samples"] = sample_regions(arr)
        else:
            record["pixelsBase64"] = uint8_base64(arr)
        if case.get("defaultLimitsRejected"):
            record["defaultLimitsRejected"] = True
        cases.append(record)
    return {"cases": cases}


def run_resize(input_path: Path, provenance: dict) -> dict:
    import numpy as np
    from PIL import Image

    kernels = {
        1: Image.Resampling.LANCZOS,
        2: Image.Resampling.BILINEAR,
        3: Image.Resampling.BICUBIC,
    }
    root = input_path.parent.resolve()
    cases = []
    for case in json.loads(input_path.read_text(encoding="utf-8"))["cases"]:
        fixture = verify_fixture(root, case["image"], provenance)
        image = Image.open(fixture)
        image.load()
        if image.mode != "RGB":
            raise SystemExit(f"resize fixture must be opaque RGB: {case['image']}")
        source = np.asarray(image, dtype=np.uint8)
        resized = np.asarray(
            image.resize((case["width"], case["height"]), resample=kernels[case["resample"]]),
            dtype=np.uint8,
        )
        assert resized.shape == (case["height"], case["width"], 3)
        cases.append(
            {
                "height": case["height"],
                "name": case["name"],
                "pixelsBase64": uint8_base64(resized),
                "resample": case["resample"],
                "width": case["width"],
                "inputShape": [int(source.shape[0]), int(source.shape[1])],
            }
        )
    return {"cases": cases}


def run_chain(input_path: Path, provenance: dict) -> dict:
    import numpy as np
    from PIL import Image
    from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor

    specification = json.loads(input_path.read_text(encoding="utf-8"))
    root = input_path.parent.resolve()
    config = json.loads(verify_fixture(root, "preprocessor_config.json", provenance).read_text("utf-8"))
    size = int(specification["configOverrides"]["size"])
    tile = int(specification["configOverrides"]["maxImageSize"])

    def build(resample: int, do_splitting: bool) -> Idefics3ImageProcessor:
        cfg = {k: v for k, v in config.items() if k not in ("image_processor_type", "processor_class")}
        cfg["size"] = {"longest_edge": size}
        cfg["max_image_size"] = {"longest_edge": tile}
        cfg["resample"] = resample
        cfg["do_image_splitting"] = do_splitting
        return Idefics3ImageProcessor(**cfg)

    cases = []
    for case in specification["cases"]:
        fixture = verify_fixture(root, case["image"], provenance)
        resample = case["resample"]
        do_splitting = case["doImageSplitting"]
        proc = build(resample, do_splitting)
        image = Image.open(fixture)
        image.load()
        if image.mode != "RGB":
            raise SystemExit(f"chain fixture must be opaque RGB: {case['image']}")
        source = np.asarray(image, dtype=np.uint8)

        stage1 = np.asarray(
            proc.resize(source, size={"longest_edge": size}, resample=resample), dtype=np.uint8
        )
        stage2 = None
        if do_splitting:
            stage2 = np.asarray(
                proc.resize_for_vision_encoder(stage1, tile, resample=resample), dtype=np.uint8
            )
        full = proc.preprocess(image, return_tensors=None, return_row_col_info=True)
        pixel_values = full["pixel_values"][0]
        masks = full["pixel_attention_mask"][0]
        # rows/cols are nested one level per batch dimension: [[value]] for a single image.
        rows = full["rows"][0][0]
        cols = full["cols"][0][0]

        if do_splitting:
            frames, split_rows, split_cols = proc.split_image(
                stage2, {"longest_edge": tile}, resample=resample
            )
            frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
            if (split_rows, split_cols) != (rows, cols):
                raise SystemExit(
                    f"split metadata disagrees between split_image and preprocess: {case['name']}"
                )
            height, width = stage2.shape[:2]
            regions = []
            if height > tile or width > tile:
                optimal_h = math.ceil(height / rows)
                optimal_w = math.ceil(width / cols)
                for row_index in range(rows):
                    for col_index in range(cols):
                        start_x = col_index * optimal_w
                        start_y = row_index * optimal_h
                        regions.append(
                            [
                                start_x,
                                start_y,
                                min(start_x + optimal_w, width),
                                min(start_y + optimal_h, height),
                            ]
                        )
        else:
            frames = [
                np.asarray(
                    proc.resize(stage1, {"height": tile, "width": tile}, resample=resample),
                    dtype=np.uint8,
                )
            ]
            rows = cols = 0
            regions = None

        if not (len(frames) == len(pixel_values) == len(masks)):
            raise SystemExit(f"frame/tile/mask count mismatch: {case['name']}")
        tile_records = []
        mask_records = []
        for index, frame in enumerate(frames):
            frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
            record = {"height": frame_height, "width": frame_width}
            if do_splitting and index < len(frames) - 1:
                x0, y0, x1, y1 = regions[index]
                if not np.array_equal(frame, stage2[y0:y1, x0:x1]):
                    raise SystemExit(
                        f"crop tile {index} is not a byte slice of stage2: {case['name']}"
                    )
                record["crop"] = [x0, y0, x1, y1]
            else:
                record["global"] = True
                record["pixelsBase64"] = uint8_base64(frame)
            tile_value = pixel_values[index]
            if tuple(tile_value.shape) != (3, frame_height, frame_width):
                raise SystemExit(
                    f"tile {index} shape {tile_value.shape} does not match frame: {case['name']}"
                )
            # The padded, full-run tile must equal the per-frame method chain exactly.
            chained = proc.normalize(
                proc.rescale(frame, config["rescale_factor"]),
                mean=config["image_mean"],
                std=config["image_std"],
            )
            if not np.array_equal(
                np.asarray(chained, dtype=np.float32).transpose(2, 0, 1),
                np.asarray(tile_value, dtype=np.float32),
            ):
                raise SystemExit(f"per-frame chain disagrees with preprocess output: {case['name']}")
            record["normalizedBase64"] = float32_be_base64(np.asarray(tile_value, dtype=np.float32).transpose(1, 2, 0))
            mask = np.asarray(masks[index])
            if tuple(mask.shape) != (frame_height, frame_width) or not np.all(mask == 1):
                raise SystemExit(
                    f"tile {index} mask is not an all-ones {frame_height}x{frame_width}: {case['name']}"
                )
            mask_records.append({"allOnes": True, "height": frame_height, "width": frame_width})
            tile_records.append(record)

        record = {
            "cols": cols,
            "doImageSplitting": do_splitting,
            "masks": mask_records,
            "name": case["name"],
            "resample": resample,
            "rows": rows,
            "stage1": {
                "height": int(stage1.shape[0]),
                "pixelsBase64": uint8_base64(stage1),
                "width": int(stage1.shape[1]),
            },
            "tiles": tile_records,
        }
        if stage2 is not None:
            record["stage2"] = {
                "height": int(stage2.shape[0]),
                "pixelsBase64": uint8_base64(stage2),
                "width": int(stage2.shape[1]),
            }
        cases.append(record)
    return {
        "cases": cases,
        "config": {
            "doNormalize": config["do_normalize"],
            "doPad": config["do_pad"],
            "doResize": config["do_resize"],
            "doRescale": config["do_rescale"],
            "imageMean": config["image_mean"],
            "imageStd": config["image_std"],
            "maxImageSize": tile,
            "rescaleFactor": config["rescale_factor"],
            "size": size,
        },
    }


def stale_message(path: Path, expected: str, actual: str) -> str:
    try:
        expected_value = json.loads(expected)
        actual_value = json.loads(actual)
        expected_lines = (
            json.dumps(expected_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).splitlines(keepends=True)
        actual_lines = (
            json.dumps(actual_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).splitlines(keepends=True)
    except json.JSONDecodeError:
        expected_lines = expected.splitlines(keepends=True)
        actual_lines = actual.splitlines(keepends=True)
    difference = "".join(
        difflib.unified_diff(
            expected_lines, actual_lines, fromfile=str(path), tofile="oracle output"
        )
    )
    if not difference:
        difference = (
            "documents are structurally identical; the committed file is not in "
            "canonical form (key order, separators, indentation, or trailing newline)\n"
        )
    return f"image oracle fixture is stale: {path}\n{difference}"


RUNNERS = {
    "decode.input.json": run_decode,
    "resize.input.json": run_resize,
    "chain.input.json": run_chain,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate-all", action="store_true")
    action.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    inputs = sorted(args.fixtures_dir.glob("*.input.json"))
    if not inputs:
        raise SystemExit("no image oracle input fixtures found")
    unknown = sorted(path.name for path in inputs if path.name not in RUNNERS)
    if unknown:
        raise SystemExit(f"unknown image oracle input fixtures: {unknown}")
    expected_names = {path.name for path in args.fixtures_dir.glob("*.expected.json")}
    declared_names = {
        path.name.replace(".input.json", ".expected.json") for path in inputs
    }
    orphaned = sorted(expected_names - declared_names)
    if orphaned:
        raise SystemExit(f"orphaned image oracle fixtures: {orphaned}")
    for input_path in inputs:
        expected_path = input_path.with_name(
            input_path.name.replace(".input.json", ".expected.json")
        )
        actual = canonical(RUNNERS[input_path.name](input_path, provenance))
        if args.generate_all:
            expected_path.write_text(actual, encoding="utf-8")
            print(f"Image oracle fixture generated: {expected_path}")
        else:
            if not expected_path.is_file():
                raise SystemExit(f"image oracle fixture is missing: {expected_path}")
            expected = expected_path.read_text(encoding="utf-8")
            if actual != expected:
                raise SystemExit(stale_message(expected_path, expected, actual))
            print(f"Image oracle fixture verified: {expected_path}")


if __name__ == "__main__":
    main()
