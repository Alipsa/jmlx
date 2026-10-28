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


def run_exif_orientation(input_path: Path, provenance: dict) -> dict:
    """Per case: the stored (unrotated) RGB decode and the Reference's
    `ImageOps.exif_transpose` result, plus the effective orientation `getexif().get(274, 0)`
    reports *after* the XMP fallback. A present Orientation tag suppresses the fallback even
    when its value never matches the transpose table (2-8), so such a fixture records the
    raw stored value, not the XMP's."""
    import numpy as np
    from PIL import Image, ImageOps

    root = input_path.parent.resolve()
    cases = []
    for case in json.loads(input_path.read_text(encoding="utf-8"))["cases"]:
        fixture = verify_fixture(root, case["image"], provenance)
        image = Image.open(fixture)
        image.load()
        orientation = int(image.getexif().get(274, 0))
        stored = np.asarray(image.convert("RGB"), dtype=np.uint8)
        transposed = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        cases.append(
            {
                "name": case["name"],
                "orientation": orientation,
                "stored": {
                    "height": int(stored.shape[0]),
                    "pixelsBase64": uint8_base64(stored),
                    "width": int(stored.shape[1]),
                },
                "transposed": {
                    "height": int(transposed.shape[0]),
                    "pixelsBase64": uint8_base64(transposed),
                    "width": int(transposed.shape[1]),
                },
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


# The pinned Qwen3-VL artifact's preprocessing constants (findings §4.2-§4.3): the slow
# PIL processor's size-dict bug is compensated with explicit min/max pixels, and
# max_pixels is the jmlx planned-effective bound (decision (b)), not the artifact's
# declared 16777216. The raw-patch-row budget is the jmlx decision-(c) hard bound.
QWEN3VL_MIN_PIXELS = 65536
QWEN3VL_MAX_PIXELS = 1048576
QWEN3VL_RAW_ROW_BUDGET = 4096
QWEN3VL_PATCH_STRIDE = 1536  # 3 channels * 2 temporal slots * 16 * 16 patch pixels
QWEN3VL_PATCH_CASES = ("model-rgb-256x256", "model-rgb-64x32")


def qwen3vl_reconstruct_hwc(raw, grid_h: int, grid_w: int):
    """Invert the reference's merge-block patch flatten (findings §4.6) back to the resized
    HWC uint8 image: rows are C-order over (merged_h, merged_w, merge_h, merge_w) and each
    row is C-order over (channel, temporal, patch_row, patch_col), so the [rows, 1536]
    buffer is [merged_h, merged_w, 2, 2, 3, 2, 16, 16]. The one image frame is duplicated
    across both temporal slots (checked by the caller), so slot 0 recovers every pixel
    losslessly."""
    import numpy as np

    buf = raw.reshape(grid_h // 2, grid_w // 2, 2, 2, 3, 2, 16, 16)
    hwc = buf[:, :, :, :, :, 0, :, :].transpose(0, 2, 5, 1, 3, 6, 4)
    return np.ascontiguousarray(hwc, dtype=np.uint8).reshape(grid_h * 16, grid_w * 16, 3)


def run_qwen3vl(input_path: Path, provenance: dict) -> dict:
    """Qwen3-VL preprocessing oracle (Phase 7.3b WP2): the documented jmlx reference
    (findings §4.1-§4.3) is the slow PIL processor built over the pinned
    preprocessor_config.json with the 4.57.6 size-dict bug compensated by explicit
    options: Qwen2VLImageProcessor.from_pretrained(qwen3vl, min_pixels=65536,
    max_pixels=1048576), the planned-effective max per decision (b). doResize:false
    cases run through a second instance built with the construct-time do_resize=False
    flag.

    Every case feeds the processor the fixture's absolute path string, never a PIL
    image object: in the 4.57.6 slow processor exif_transpose runs only on the
    string-input load path, while PIL-image inputs skip it entirely, and the
    file->processor pipeline is what the jmlx Qwen decode policy
    (ExifOrientationPolicy.APPLY + AlphaPolicy.DROP_ALPHA) mirrors and what the WP1
    exif golden measured. Parent-probed 2026-10-18 that alpha and plain cases are
    input-mode-insensitive while the rotated fixtures differ. The processor's own
    convert_to_rgb drops alpha (never composites) and never applies palette tRNS;
    the per-image source dims and the storedDecode/whiteComposite controls are read
    from a separate Image.open(fixture), keeping them on the stored (untransformed)
    pixels. Per image, both the full pass (rescaled
    and normalized float32 patch buffer) and the pre-rescale pass (do_rescale=False,
    do_normalize=False, uint8 patch buffer) are pinned; the uint8 buffer is inverted
    back to the resized HWC image (qwen3vl_reconstruct_hwc) so resize divergence stays
    separable from normalization divergence. The two model fixtures' full-pass float32
    bytes are additionally materialized as the committed processed-patch bins under
    qwen3vl/patches/ (written in --generate-all mode, digest-verified otherwise;
    --verify-all never writes)."""
    import sys

    import numpy as np
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    generate = "--generate-all" in sys.argv
    specification = json.loads(input_path.read_text(encoding="utf-8"))
    root = input_path.parent.resolve()
    # The pinned config the processor resolves its options from; digest-checked here so a
    # tampered copy fails before from_pretrained reads it.
    verify_fixture(root, "qwen3vl/preprocessor_config.json", provenance)

    processors = {}

    def processor_for(do_resize: bool) -> Qwen2VLImageProcessor:
        if do_resize not in processors:
            kwargs = {"min_pixels": QWEN3VL_MIN_PIXELS, "max_pixels": QWEN3VL_MAX_PIXELS}
            if not do_resize:
                kwargs["do_resize"] = False
            processors[do_resize] = Qwen2VLImageProcessor.from_pretrained(
                str(root / "qwen3vl"), **kwargs
            )
        return processors[do_resize]

    proc = processor_for(True)
    if (int(proc.min_pixels), int(proc.max_pixels)) != (
        QWEN3VL_MIN_PIXELS,
        QWEN3VL_MAX_PIXELS,
    ):
        raise SystemExit(f"qwen3vl processor options drifted: {proc.min_pixels}/{proc.max_pixels}")
    if int(proc.resample) != 3 or not proc.do_convert_rgb:
        raise SystemExit("qwen3vl processor options drifted: resample or do_convert_rgb")
    if list(proc.image_mean) != [0.5, 0.5, 0.5] or list(proc.image_std) != [0.5, 0.5, 0.5]:
        raise SystemExit("qwen3vl processor options drifted: image_mean/image_std")

    patch_blobs = {}
    patch_grids = {}
    cases = []
    for case in specification["cases"]:
        if case.get("expectedRejected"):
            fixture = verify_fixture(root, case["image"], provenance)
            image = Image.open(fixture)
            image.load()
            source = [image.width, image.height]
            if case["rejectKind"] == "aspect":
                # The reference's smart_resize aspect guard must reject: the evidence
                # pins that the rejection is the reference's own, not only jmlx's.
                try:
                    processor_for(True)(images=[str(fixture)], return_tensors=None)
                except ValueError:
                    pass
                else:
                    raise SystemExit(f"qwen3vl aspect guard did not reject: {case['name']}")
                cases.append(
                    {
                        "expectedRejected": True,
                        "name": case["name"],
                        "rejectKind": "aspect",
                        "source": source,
                    }
                )
            elif case["rejectKind"] == "raw-row-budget":
                # jmlx decision-(c) budget: the reference would process this image
                # fine, so it is deliberately not run; the stored dims must be
                # patch-size divisible or the fixture would hit the no-resize
                # divisibility check instead.
                if source[0] % 16 != 0 or source[1] % 16 != 0:
                    raise SystemExit(
                        f"qwen3vl raw-row-budget fixture must be divisible by 16: {case['image']}"
                    )
                cases.append(
                    {
                        "doResize": False,
                        "expectedRejected": True,
                        "name": case["name"],
                        "rawPatchRows": (source[1] // 16) * (source[0] // 16),
                        "rawRowBudget": QWEN3VL_RAW_ROW_BUDGET,
                        "rejectKind": "raw-row-budget",
                        "source": source,
                    }
                )
            else:
                raise SystemExit(f"unknown qwen3vl rejectKind: {case['rejectKind']}")
            continue

        do_resize = case.get("doResize", True)
        image_rels = [case["image"]] if "image" in case else list(case["images"])
        image_records = []
        multi_pixel_values = []
        multi_grids = []
        for rel in image_rels:
            fixture = verify_fixture(root, rel, provenance)
            image = Image.open(fixture)
            image.load()
            source = [image.width, image.height]
            p = processor_for(do_resize)
            # Path string, not PIL object: the 4.57.6 slow processor applies
            # exif_transpose only on the string-input load path (PIL inputs skip
            # it), and this file->processor pipeline is what the jmlx Qwen decode
            # policy (APPLY + DROP_ALPHA) mirrors and what the WP1 exif golden
            # measured; alpha/plain cases are input-mode-insensitive while the
            # rotated fixtures differ (parent-probed 2026-10-18).
            full = p(images=[str(fixture)], return_tensors=None)
            pv = np.ascontiguousarray(full["pixel_values"], dtype=np.float32)
            grid = [int(v) for v in np.asarray(full["image_grid_thw"][0]).tolist()]
            if grid[0] != 1:
                raise SystemExit(
                    f"qwen3vl grid_t != 1 for a still image: {case['name']} {rel}"
                )
            grid_h, grid_w = grid[1], grid[2]
            expected_rows = grid_h * grid_w
            if tuple(pv.shape) != (expected_rows, QWEN3VL_PATCH_STRIDE):
                raise SystemExit(
                    f"qwen3vl pixel_values shape {pv.shape} != "
                    f"[{expected_rows}, {QWEN3VL_PATCH_STRIDE}]: {rel}"
                )
            multi_pixel_values.append(pv)
            multi_grids.append(grid)

            raw_result = p(
                images=[str(fixture)], do_rescale=False, do_normalize=False, return_tensors=None
            )
            raw = raw_result["pixel_values"]
            if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8:
                raise SystemExit(
                    f"qwen3vl pre-rescale pass is not a uint8 array: {rel} "
                    f"(type={type(raw).__name__}, dtype={getattr(raw, 'dtype', None)})"
                )
            raw = np.ascontiguousarray(raw, dtype=np.uint8)
            raw_grid = [int(v) for v in np.asarray(raw_result["image_grid_thw"][0]).tolist()]
            if raw_grid != grid or tuple(raw.shape) != (expected_rows, QWEN3VL_PATCH_STRIDE):
                raise SystemExit(
                    f"qwen3vl pre-rescale pass disagrees with the full pass: {rel}"
                )
            # The one image frame fills both temporal slots identically (a per-channel
            # column-band check, the t axis sits inside each channel's 512 columns); a
            # mismatch would mean the reference's duplication changed, breaking the
            # HWC inversion below.
            for c in range(3):
                if not np.array_equal(
                    raw[:, c * 512:c * 512 + 256], raw[:, c * 512 + 256:c * 512 + 512]
                ):
                    raise SystemExit(f"qwen3vl temporal slots differ: {rel}")
            hwc = qwen3vl_reconstruct_hwc(raw, grid_h, grid_w)

            sample_rows = sorted({0, expected_rows // 2, expected_rows - 1})
            record = {
                "gridThw": [1, grid_h, grid_w],
                "image": rel,
                "mergedRowCount": expected_rows // 4,
                "pixelSha256": hashlib.sha256(pv.tobytes()).hexdigest(),
                "rawPatchRows": expected_rows,
                "resized": [grid_w * 16, grid_h * 16],
                "samples": [
                    {"row": i, "valuesBase64": float32_be_base64(pv[i])}
                    for i in sample_rows
                ],
                "source": source,
                "uint8Sha256": hashlib.sha256(hwc.tobytes()).hexdigest(),
            }
            if case["name"].startswith("alpha-"):
                # The unchanged SmolVLM composite-over-white decode: the control a
                # drop-alpha implementation must visibly differ from.
                white = reference_decode(fixture)
                record["whiteComposite"] = {
                    "height": int(white.shape[0]),
                    "pixelsBase64": uint8_base64(white),
                    "width": int(white.shape[1]),
                }
            if case["name"].startswith("rotated-"):
                # The raw stored (untransformed) pixels, read straight from the
                # opened image before any transpose. The path-input pipeline
                # applies exif_transpose to exactly these (orient6: 90 CW,
                # orient2: mirrored left-right), so the processor output must
                # differ from this control; an exif-ignoring implementation
                # would produce exactly the stored-pixel pipeline instead.
                stored = np.asarray(image.convert("RGB"), dtype=np.uint8)
                record["storedDecode"] = {
                    "height": int(stored.shape[0]),
                    "pixelsBase64": uint8_base64(stored),
                    "width": int(stored.shape[1]),
                }
            image_records.append(record)
            if case["name"] in QWEN3VL_PATCH_CASES:
                patch_blobs[case["name"]] = pv.tobytes()
                patch_grids[case["name"]] = grid
        if len(image_rels) > 1:
            # The combined call must reproduce the per-image results in order: a
            # reference version that padded or reordered images would change this
            # batch layout without changing any single-image output.
            both = processor_for(do_resize)(
                images=[str(root / r) for r in image_rels], return_tensors=None
            )
            combined_grids = [
                [int(v) for v in row] for row in np.asarray(both["image_grid_thw"]).tolist()
            ]
            if not np.array_equal(
                np.asarray(both["pixel_values"], dtype=np.float32),
                np.concatenate(multi_pixel_values),
            ) or combined_grids != multi_grids:
                raise SystemExit(
                    f"qwen3vl multi-image call disagrees with per-image concat: {case['name']}"
                )
        cases.append({"images": image_records, "name": case["name"]})

    patches_dir = root / "qwen3vl" / "patches"
    manifest_entries = {}
    for name in QWEN3VL_PATCH_CASES:
        if name not in patch_blobs:
            raise SystemExit(f"qwen3vl patch case missing from input: {name}")
        data = patch_blobs[name]
        grid = patch_grids[name]
        digest = hashlib.sha256(data).hexdigest()
        bin_path = patches_dir / f"{name}.pixel-values.f32"
        if generate:
            patches_dir.mkdir(parents=True, exist_ok=True)
            bin_path.write_bytes(data)
        elif not bin_path.is_file() or source_digest(bin_path) != digest:
            raise SystemExit(f"qwen3vl patch bin missing or stale: {bin_path}")
        manifest_entries[f"{name}.pixel-values.f32"] = {
            "gridThw": [1, grid[1], grid[2]],
            "rows": grid[1] * grid[2],
            "sha256": digest,
            "sourceImage": f"qwen3vl/{name}.png",
            "sourceImageSha256": provenance["fixtureSources"][f"qwen3vl/{name}.png"],
            "stride": QWEN3VL_PATCH_STRIDE,
        }
    manifest = {
        "layout": (
            "little-endian IEEE-754 float32, C order, [rawPatchRows, 1536] (patch stride "
            "= 3 * temporal_patch_size * patch_size * patch_size)"
        ),
        "patches": manifest_entries,
        "processor": (
            "Qwen2VLImageProcessor.from_pretrained(qwen3vl, min_pixels=65536, "
            "max_pixels=1048576)"
        ),
        "schema": "jmlx-phase7-3b-wp2-qwen3vl-processed-patches-v1",
    }
    manifest_path = patches_dir / "manifest.json"
    if generate:
        manifest_path.write_text(canonical(manifest), encoding="utf-8")
    else:
        if not manifest_path.is_file():
            raise SystemExit(f"qwen3vl patch manifest is missing: {manifest_path}")
        committed = manifest_path.read_text(encoding="utf-8")
        if canonical(json.loads(committed)) != committed:
            raise SystemExit(
                f"qwen3vl patch manifest is not in canonical form: {manifest_path}"
            )
        if json.loads(committed) != manifest:
            raise SystemExit(
                f"qwen3vl patch manifest disagrees with recomputed patches: {manifest_path}"
            )

    return {
        "cases": cases,
        "config": {
            "class": "Qwen2VLImageProcessor",
            "construction": (
                "Qwen2VLImageProcessor.from_pretrained(qwen3vl, min_pixels=65536, "
                "max_pixels=1048576)"
            ),
            "imageMean": list(proc.image_mean),
            "imageStd": list(proc.image_std),
            "maxPixels": QWEN3VL_MAX_PIXELS,
            "mergeSize": 2,
            "minPixels": QWEN3VL_MIN_PIXELS,
            "patchSize": 16,
            "rawRowBudget": QWEN3VL_RAW_ROW_BUDGET,
            "resample": int(proc.resample),
            "rescaleFactor": proc.rescale_factor,
            "temporalPatchSize": 2,
        },
    }


PHI3V_PATCH_CASES = ("square-336x336", "wide-672x336", "tall-336x672", "trunc-300x301")


def run_phi3v(input_path: Path, provenance: dict) -> dict:
    """Phi-3.5-Vision preprocessing oracle (Phase 7.3c WP2): the hash-pinned bundled
    trust-remote-code processor under fixtures/phi3v/ (findings §6), loaded exactly as
    the WP1 probe loads it (get_class_from_dynamic_module +
    Phi3VImageProcessor.from_pretrained) and run with return_tensors="pt" on PIL images
    generated with the probe's seeding, so the pixel_values sha256s cross-check the
    probe golden (tools/hf-reference/goldens/phi3v-wp1-probe.json).

    Each case feeds the processor a list of PIL images (the reference's preprocess path
    never consults EXIF, so input mode cannot change the pixels) and records, per image:
    the source dimensions, the padded image_sizes entry [height, width], the crop grid,
    the slot count and the per-image token count, the full-row sha256 and deterministic
    sample rows. The four single-image cases' full float32 rows are additionally
    materialized as the committed processed-patch bins under phi3v/patches/ (written in
    --generate-all mode, digest-verified otherwise; --verify-all never writes)."""
    import sys

    import numpy as np
    from PIL import Image
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    generate = "--generate-all" in sys.argv
    specification = json.loads(input_path.read_text(encoding="utf-8"))
    root = input_path.parent.resolve()
    # The pinned bundled sources; digest-checked here so a tampered copy fails before
    # import (the runner is a Python entry point of the image oracle, not Gradle).
    for rel in (
        "phi3v/preprocessor_config.json",
        "phi3v/processing_phi3_v.py",
        "phi3v/configuration_phi3_v.py",
        "phi3v/modeling_phi3_v.py",
        "phi3v/sample_inference.py",
    ):
        verify_fixture(root, rel, provenance)

    proc_path = str(root / "phi3v")
    proc_cls = get_class_from_dynamic_module("processing_phi3_v.Phi3VProcessor", proc_path)
    proc_mod = sys.modules[proc_cls.__module__]
    image_processor = proc_mod.Phi3VImageProcessor.from_pretrained(proc_path)
    if int(image_processor.num_crops) != 4:
        raise SystemExit(
            f"phi3v processor options drifted: num_crops={image_processor.num_crops}")
    if list(image_processor.image_mean) != [0.48145466, 0.4578275, 0.40821073]:
        raise SystemExit("phi3v processor options drifted: image_mean")
    if list(image_processor.image_std) != [0.26862954, 0.26130258, 0.27577711]:
        raise SystemExit("phi3v processor options drifted: image_std")
    if not image_processor.do_convert_rgb:
        raise SystemExit("phi3v processor options drifted: do_convert_rgb")

    patch_blobs = {}
    cases = []
    for case in specification["cases"]:
        image_rels = [case["image"]] if "image" in case else list(case["images"])
        pil_images = []
        for rel in image_rels:
            fixture = verify_fixture(root, rel, provenance)
            image = Image.open(fixture)
            image.load()
            if image.mode != "RGB":
                raise SystemExit(f"phi3v fixture must be an RGB image: {rel} mode={image.mode}")
            pil_images.append(image)
        out = image_processor(images=pil_images, return_tensors="pt")
        pv = np.ascontiguousarray(out["pixel_values"], dtype=np.float32)
        image_sizes = [[int(x) for x in row] for row in out["image_sizes"]]
        num_img_tokens = [int(x) for x in out["num_img_tokens"]]
        if pv.ndim != 5 or pv.shape[0] != len(image_rels) or tuple(pv.shape[2:]) != (3, 336, 336):
            raise SystemExit(f"phi3v pixel_values shape {pv.shape} unexpected: {case['name']}")
        if len(image_sizes) != len(image_rels) or len(num_img_tokens) != len(image_rels):
            raise SystemExit(f"phi3v per-image metadata count mismatch: {case['name']}")
        image_records = []
        for i, rel in enumerate(image_rels):
            fixture = (root / rel).resolve()
            image = Image.open(fixture)
            image.load()
            row = pv[i]
            slots = int(row.shape[0])
            if slots != 5:
                raise SystemExit(f"phi3v slot count {slots} != 5 (num_crops=4): {rel}")
            padded_height, padded_width = image_sizes[i]
            grid_h = padded_height // 336
            grid_w = padded_width // 336
            tokens = num_img_tokens[i]
            expected_tokens = (grid_h * grid_w + 1) * 144 + 1 + (grid_h + 1) * 12
            if tokens != expected_tokens:
                raise SystemExit(
                    f"phi3v token count {tokens} != {expected_tokens} for {rel}")
            sample_rows = sorted({0, slots // 2, slots - 1})
            record = {
                "cropBlocks": grid_h * grid_w,
                "gridH": grid_h,
                "gridW": grid_w,
                "image": rel,
                "imageSize": image_sizes[i],
                "numImgTokens": tokens,
                "pixelSha256": hashlib.sha256(row.tobytes()).hexdigest(),
                "samples": [
                    {"row": s, "valuesBase64": float32_be_base64(row[s])}
                    for s in sample_rows
                ],
                "slotCount": slots,
                "source": [image.width, image.height],
            }
            image_records.append(record)
            if len(image_rels) == 1 and case["name"] in PHI3V_PATCH_CASES:
                patch_blobs[case["name"]] = row.tobytes()
        if len(image_rels) > 1:
            # The combined call must reproduce the per-image rows in order: a reference
            # version that padded or reordered images would change this batch layout
            # without changing any single-image output.
            per_image = np.concatenate(
                [
                    np.ascontiguousarray(
                        image_processor(images=[p], return_tensors="pt")["pixel_values"],
                        dtype=np.float32,
                    )
                    for p in pil_images
                ],
                axis=0,
            )
            if not np.array_equal(pv, per_image):
                raise SystemExit(
                    f"phi3v multi-image call disagrees with per-image concat: {case['name']}")
        cases.append({"images": image_records, "name": case["name"]})

    patches_dir = root / "phi3v" / "patches"
    manifest_entries = {}
    for name in PHI3V_PATCH_CASES:
        if name not in patch_blobs:
            raise SystemExit(f"phi3v patch case missing from input: {name}")
        data = patch_blobs[name]
        digest = hashlib.sha256(data).hexdigest()
        bin_path = patches_dir / f"{name}.pixel-values.f32"
        if generate:
            patches_dir.mkdir(parents=True, exist_ok=True)
            bin_path.write_bytes(data)
        elif not bin_path.is_file() or source_digest(bin_path) != digest:
            raise SystemExit(f"phi3v patch bin missing or stale: {bin_path}")
        manifest_entries[f"{name}.pixel-values.f32"] = {
            "rows": 5,
            "sha256": digest,
            "slotCount": 5,
            "sourceImage": f"phi3v/{name}.png",
            "sourceImageSha256": provenance["fixtureSources"][f"phi3v/{name}.png"],
            "stride": 3 * 336 * 336,
        }
    manifest = {
        "layout": (
            "little-endian IEEE-754 float32, C order, [slotCount, 3, 336, 336] "
            "(slot-major: global bicubic image first, then the h-major 336x336 crops, "
            "zero-padded slots)"
        ),
        "patches": manifest_entries,
        "processor": (
            "Phi3VImageProcessor.from_pretrained(phi3v) (bundled trust-remote-code, "
            "num_crops=4)"
        ),
        "schema": "jmlx-phase7-3c-wp2-phi3v-processed-patches-v1",
    }
    manifest_path = patches_dir / "manifest.json"
    if generate:
        manifest_path.write_text(canonical(manifest), encoding="utf-8")
    else:
        if not manifest_path.is_file():
            raise SystemExit(f"phi3v patch manifest is missing: {manifest_path}")
        committed = manifest_path.read_text(encoding="utf-8")
        if canonical(json.loads(committed)) != committed:
            raise SystemExit(f"phi3v patch manifest is not in canonical form: {manifest_path}")
        if json.loads(committed) != manifest:
            raise SystemExit(
                f"phi3v patch manifest disagrees with recomputed patches: {manifest_path}")

    return {
        "cases": cases,
        "config": {
            "class": "Phi3VImageProcessor",
            "construction": "Phi3VImageProcessor.from_pretrained(phi3v)",
            "imageMean": list(image_processor.image_mean),
            "imageStd": list(image_processor.image_std),
            "numCrops": int(image_processor.num_crops),
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
    "exif-orientation.input.json": run_exif_orientation,
    "resize.input.json": run_resize,
    "chain.input.json": run_chain,
    "qwen3vl.input.json": run_qwen3vl,
    "phi3v.input.json": run_phi3v,
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
