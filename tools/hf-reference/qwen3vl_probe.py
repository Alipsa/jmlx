#!/usr/bin/env python3
"""Phase 7.3b WP1 reference probe for Qwen3-VL-2B-Instruct.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0,
Pillow 12.3.0, NumPy 2.5.3, Python 3.12), CPU float32, single-threaded,
deterministic. It is never invoked by Gradle; it writes one probe JSON to
tools/hf-reference/goldens/ whose SHA-256 is pinned in
tools/hf-reference/provenance.json.

The probe is the executable evidence behind req/plans/phase7-3b-reference-
findings.md:

  * artifact inventory: file SHA-256s, full safetensors tensor inventory,
    Hugging Face Hub commit/download metadata for the pinned revision;
  * resolved configuration: vision/text config keys, preprocessor options,
    tokenizer/chat-template facts, generation config;
  * slow-processor pipeline: explicit slow Qwen2VLImageProcessor plus a
    Qwen3VLProcessor(video_processor=None) construction probe, smart_resize
    geometry re-derivation, alpha/color/EXIF path evidence;
  * raw model stepping: prefill plus cached decode with three-axis
    position ids, rope deltas (re-derived independently), intermediate
    shapes, DeepStack features, and greedy ids.

No generation processors, no sampling: only forward() stepping with
argmax selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
import tempfile
import time
from pathlib import Path

ORG = "Qwen"
REPO = "Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"

IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652


def gate_versions() -> dict:
    import PIL
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import __version__ as hf_hub_version
    from safetensors import __version__ as safetensors_version

    if (
        transformers.__version__ != "4.57.6"
        or PIL.__version__ != "12.3.0"
        or np.__version__ != "2.5.3"
        or torch.__version__ not in ("2.6.0", "2.6.0+cpu")
        or platform.python_version_tuple()[:2] != ("3", "12")
    ):
        sys.exit(
            "qwen3vl_probe requires transformers 4.57.6, Pillow 12.3.0, NumPy 2.5.3, "
            f"torch 2.6.0 and CPython 3.12, got transformers {transformers.__version__}, "
            f"Pillow {PIL.__version__}, NumPy {np.__version__}, torch {torch.__version__}, "
            f"Python {platform.python_version()}"
        )
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "safetensors": safetensors_version,
        "huggingface_hub": hf_hub_version,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(t) -> dict:
    """Full SHA-256 of the float32 buffer plus deterministic subsamples."""
    import torch

    if not torch.is_tensor(t):
        t = torch.from_numpy(t)
    t = t.detach().cpu().contiguous()
    data = t.numpy().tobytes()
    flat = t.reshape(-1)
    n = flat.numel()
    indices = [i * (n - 1) // 31 for i in range(min(32, n))]
    spaced = [flat[i].item() for i in indices]
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "first16": [flat[i].item() for i in range(min(16, n))],
        "spaced32": spaced,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def py(obj):
    """Make numpy/torch values JSON-serializable."""
    if isinstance(obj, dict):
        return {k: py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    import torch

    if isinstance(obj, torch.Tensor):
        return py(obj.tolist())
    if hasattr(obj, "item"):
        return py(obj.item())
    return str(obj)


def build_test_images(workdir: Path) -> dict:
    from PIL import Image

    paths = {}

    def save(name, image, **kwargs):
        path = workdir / name
        image.save(path, **kwargs)
        return path

    # 256x256: exactly at min_pixels, no smart_resize branch taken.
    px = bytes()
    for y in range(256):
        for x in range(256):
            px += bytes((x, y, (x + y) // 2))
    paths["rgb_256x256"] = save(
        "rgb-256x256.png", Image.frombytes("RGB", (256, 256), px), format="PNG"
    )

    # 64x32: below min_pixels, exercises the upscale branch (192x384).
    px = bytes()
    for y in range(32):
        for x in range(64):
            px += bytes(((x * 255) // 63, (y * 255) // 31, (x * 3 + y * 5) % 256))
    paths["rgb_64x32"] = save(
        "rgb-64x32.png", Image.frombytes("RGB", (64, 32), px), format="PNG"
    )

    # RGBA: constant opaque-ish RGB, per-pixel varying alpha (includes 0).
    px = bytes()
    for y in range(8):
        for x in range(8):
            px += bytes((200, 100, 50, (x * 31 + y * 31) % 256))
    paths["rgba_8x8"] = save(
        "rgba-8x8.png", Image.frombytes("RGBA", (8, 8), px), format="PNG"
    )

    # LA: varying gray and alpha.
    px = bytes()
    for y in range(8):
        for x in range(8):
            px += bytes(((x * 31 + y * 31) % 256, (x * 17 + y * 23) % 256))
    paths["la_8x8"] = save("la-8x8.png", Image.frombytes("LA", (8, 8), px), format="PNG")

    # 4-bit palette PNG with a tRNS chunk: colors 0-3 partially/fully
    # transparent, colors 4-15 opaque.
    palette = []
    for i in range(16):
        palette.extend((i * 17) % 256 for _ in range(3))
    img = Image.new("P", (16, 16))
    img.putpalette(palette)
    img.putdata([(x * 2 + y * 3) % 16 for y in range(16) for x in range(16)])
    paths["palette_4bit_trns"] = save(
        "palette-4bit-trns.png", img, format="PNG", transparency=bytes([0, 128, 200, 255])
    )

    # 24x12 JPEG with EXIF orientation 6 (correct orientation is a 90 CW turn).
    px = bytes()
    for y in range(12):
        for x in range(24):
            px += bytes((min(255, x * 10), min(255, y * 20), 64))
    img = Image.frombytes("RGB", (24, 12), px)
    tiff = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    ifd = struct.pack("<H", 1) + struct.pack("<HHII", 0x0112, 3, 1, 6) + struct.pack("<I", 0)
    exif = b"Exif\x00\x00" + tiff + ifd
    paths["exif_orientation6"] = save(
        "exif-orientation6.jpg", img, format="JPEG", quality=95, exif=exif
    )
    return paths


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    """Exact re-implementation of the pinned slow processor's smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_token_span(ids: list[int]) -> tuple[int, int]:
    starts = [i for i, v in enumerate(ids) if v == IMAGE_TOKEN_ID]
    return starts[0], starts[-1]


def rederive_rope(input_ids: list[int], grid: list[int]) -> dict:
    """Independent re-derivation of get_rope_index for one image, one batch row.

    Encodes the exact 4.57.6 algorithm: text tokens get all three axes equal;
    image tokens get T = base, H = base + h_idx, W = base + w_idx with base =
    st_idx + text_len, where the llm grid is (t, h/merge, w/merge); st then
    advances by ed + t*lh*lw and the following text resumes at max+1.
    """
    t, h, w = grid
    lh, lw = h // 2, w // 2
    ed_image = input_ids.index(IMAGE_TOKEN_ID)
    text_len = ed_image
    st_idx = 0
    positions = [[st_idx + i for i in range(text_len)] for _ in range(3)]
    base = st_idx + text_len
    for hi in range(lh):
        for wi in range(lw):
            positions[0].append(base + 0)
            positions[1].append(base + hi)
            positions[2].append(base + wi)
    st = ed_image + lh * lw
    if st < len(input_ids):
        st_idx = max(max(axis) for axis in positions) + 1 if positions[0] else 0
        positions[0].extend(st_idx + i for i in range(len(input_ids) - st))
        positions[1].extend(st_idx + i for i in range(len(input_ids) - st))
        positions[2].extend(st_idx + i for i in range(len(input_ids) - st))
    max_pos = max(max(a) for a in positions)
    return {
        "position_ids": positions,
        "rope_delta": max_pos + 1 - len(input_ids),
        "expected_delta_formula": "max(llm_pos) + 1 - seq_len",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parent.parent.parent / ".tier-b" / "qwen3-vl-2b-instruct"
    parser.add_argument("--dir", type=Path, default=default_dir)
    default_out = Path(__file__).resolve().parent / "goldens" / "qwen3-vl-2b-wp1-probe.json"
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--skip-model", action="store_true",
                        help="run inventory + processor only (no model loading)")
    args = parser.parse_args()

    directory: Path = args.dir
    if not (directory / "config.json").is_file():
        sys.exit(f"no config.json under {directory}; pass --dir to the pinned snapshot")

    environment = gate_versions()
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    result: dict = {
        "schema": "jmlx-phase7-3b-wp1-qwen3vl-probe-v1",
        "generated_by": "tools/hf-reference/qwen3vl_probe.py",
        "environment": environment,
        "artifact": {
            "org": ORG,
            "repo": REPO,
            "revision": REVISION,
        },
    }

    # ---------------------------------------------------------------- inventory
    import huggingface_hub
    from safetensors import safe_open

    files = []
    total_bytes = 0
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": path.name, "bytes": size, "sha256": sha256_file(path)})

    hub = huggingface_hub.HfApi().model_info(f"{ORG}/{REPO}", revision=REVISION)
    siblings = {
        s.rfilename: {"bytes": s.size, "lfs_sha256": (s.lfs or {}).get("sha256")}
        for s in hub.siblings
    }
    for entry in files:
        hub_entry = siblings.get(entry["path"])
        if hub_entry is not None:
            entry["hub_bytes"] = hub_entry["bytes"]
            entry["hub_lfs_sha256"] = hub_entry["lfs_sha256"]
            # LFS files carry their content SHA-256 in hub metadata; verify it
            # directly. Non-LFS files may report no size; compare when present.
            if hub_entry["lfs_sha256"] is not None and entry["sha256"] != hub_entry["lfs_sha256"]:
                sys.exit(f"hub LFS SHA-256 mismatch for {entry['path']}")
            if hub_entry["bytes"] is not None and entry["bytes"] != hub_entry["bytes"]:
                sys.exit(f"hub size mismatch for {entry['path']}")
    result["artifact"]["hub"] = {
        "commit_sha": hub.sha,
        "last_modified": str(hub.lastModified),
        "downloads": hub.downloads,
        "likes": hub.likes,
        "tags": sorted(hub.tags),
    }
    result["artifact"]["files"] = files
    result["artifact"]["total_bytes"] = total_bytes

    stensors = []
    with safe_open(directory / "model.safetensors", framework="pt") as st:
        for key in st.keys():
            slice_ = st.get_slice(key)
            shape = list(slice_.get_shape())
            itemsize = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2,
                        "I8": 1, "U8": 1, "F64": 8}[slice_.get_dtype()]
            stensors.append({"key": key, "dtype": slice_.get_dtype(),
                             "shape": shape, "bytes": math.prod(shape) * itemsize})
    stensors.sort(key=lambda e: e["key"])
    by_prefix: dict[str, int] = {}
    for entry in stensors:
        prefix = entry["key"].split(".")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
    result["artifact"]["tensors"] = {
        "count": len(stensors),
        "by_prefix": by_prefix,
        "storage_dtypes": sorted({e["dtype"] for e in stensors}),
        "entries": stensors,
    }

    # ------------------------------------------------------------------ configs
    config = json.loads((directory / "config.json").read_text())
    preprocessor = json.loads((directory / "preprocessor_config.json").read_text())
    video_preprocessor = json.loads((directory / "video_preprocessor_config.json").read_text())
    generation = json.loads((directory / "generation_config.json").read_text())
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
    chat_template_file = json.loads((directory / "chat_template.json").read_text())
    result["configs"] = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "tied_weights_keys": config.get("tied_weights_keys"),
        "vision_config": config.get("vision_config"),
        "text_config": config.get("text_config"),
        "image_token_id": config.get("image_token_id"),
        "video_token_id": config.get("video_token_id"),
        "vision_start_token_id": config.get("vision_start_token_id"),
        "vision_end_token_id": config.get("vision_end_token_id"),
        "preprocessor_config": preprocessor,
        "video_preprocessor_config": video_preprocessor,
        "generation_config": generation,
        "tokenizer": {
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "add_bos_token": tokenizer_config.get("add_bos_token"),
            "model_max_length": tokenizer_config.get("model_max_length"),
            "bos_token": tokenizer_config.get("bos_token"),
            "eos_token": tokenizer_config.get("eos_token"),
            "pad_token": tokenizer_config.get("pad_token"),
            "unk_token": tokenizer_config.get("unk_token"),
            "chat_template_sha256": sha256_bytes(
                tokenizer_config["chat_template"].encode("utf-8")
            ),
            "chat_template_length": len(tokenizer_config["chat_template"]),
            "chat_template_file_matches": chat_template_file.get("chat_template")
            == tokenizer_config.get("chat_template"),
        },
    }

    # --------------------------------------------------------------- processor
    from PIL import Image
    from transformers import Qwen2TokenizerFast
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    workdir = Path(tempfile.mkdtemp(prefix="qwen3vl-probe-"))
    images = build_test_images(workdir)

    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    # A 4.57.6 slow-processor bug changes the effective geometry: the
    # __init__'s else branch replaces ANY size (including a valid
    # shortest_edge/longest_edge dict) with the defaults 3136/1003520, so
    # from_pretrained alone resolves min/max pixels to the defaults, not to
    # the artifact's declared 65536/16777216. The probe records the
    # with-config-only resolution as evidence, then compensates with explicit
    # min_pixels/max_pixels so the pipeline matches the artifact-declared
    # semantics (the fast processor the artifact names honors its config).
    img_proc_config_only = Qwen2VLImageProcessor.from_pretrained(str(directory))
    declared_min = int(preprocessor["size"]["shortest_edge"])
    declared_max = int(preprocessor["size"]["longest_edge"])
    size_bug = {
        "declared_min_pixels": declared_min,
        "declared_max_pixels": declared_max,
        "resolved_with_config_only": {
            "min_pixels": img_proc_config_only.min_pixels,
            "max_pixels": img_proc_config_only.max_pixels,
        },
        "config_size_ignored": img_proc_config_only.min_pixels != declared_min,
        "note": "transformers 4.57.6 slow Qwen2VLImageProcessor.__init__ "
        "overwrites a valid size dict with the 3136/1003520 defaults; the "
        "probe passes min_pixels/max_pixels explicitly to match the "
        "artifact-declared semantics",
    }
    img_proc = Qwen2VLImageProcessor.from_pretrained(
        str(directory), min_pixels=declared_min, max_pixels=declared_max
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(directory))
    video_proc = Qwen3VLVideoProcessor.from_pretrained(str(directory))

    # Probe the plan's video_processor=None construction first and record the
    # outcome: 4.57.6's ProcessorMixin hard-rejects None for a non-optional
    # attribute.
    try:
        Qwen3VLProcessor(image_processor=img_proc, tokenizer=tokenizer,
                         video_processor=None)
        none_probe = {"constructed": True, "error": None}
    except Exception as exc:  # noqa: BLE001 - the probe records the failure
        none_probe = {"constructed": False, "error": f"{type(exc).__name__}: {exc}"}

    # A second finding: without torchvision, BaseVideoProcessor resolves to a
    # dummy class, so even the concrete Qwen3VLVideoProcessor fails the
    # mixin's class check. The probe subclasses the processor and satisfies
    # that one check against the concrete class; image-only behavior is
    # unaffected because the image path never touches the video processor.
    class ProbeProcessor(Qwen3VLProcessor):
        def check_argument_for_proper_class(self, name, arg):
            if name == "video_processor":
                return Qwen3VLVideoProcessor
            return super().check_argument_for_proper_class(name, arg)

    processor = ProbeProcessor(image_processor=img_proc, tokenizer=tokenizer,
                               video_processor=video_proc,
                               chat_template=tokenizer_config["chat_template"])

    def pixel_summary(batch, key: str = "pixel_values") -> dict:
        return tensor_summary(batch[key])

    factor = img_proc.patch_size * img_proc.merge_size
    result["processor"] = {
        "declared_image_processor_class": preprocessor.get("image_processor_type"),
        "construction": {
            "image_processor": type(img_proc).__name__,
            "image_processor_module": type(img_proc).__module__,
            "tokenizer": type(tokenizer).__name__,
            "processor": type(processor).__name__,
            "video_processor": type(video_proc).__name__,
            "video_processor_module": type(video_proc).__module__,
            "video_processor_constructed_without_torchvision": True,
            "video_processor_none_probe": none_probe,
            "slow_vs_declared": "deviation: artifact declares "
            f"{preprocessor.get('image_processor_type')}; probe uses the slow "
            "Python Qwen2VLImageProcessor explicitly (plan decision 2026-10-08)",
            "class_check_note": "without torchvision, 4.57.6 resolves "
            "BaseVideoProcessor to a dummy class, so the mixin class check "
            "rejects every video processor; the probe overrides "
            "check_argument_for_proper_class for video_processor only "
            "(image-only behavior is unaffected)",
        },
        "slow_size_bug": size_bug,
        "video_processor_options": {
            "size": video_proc.size,
            "do_rescale": video_proc.do_rescale,
            "rescale_factor": video_proc.rescale_factor,
            "do_normalize": video_proc.do_normalize,
            "image_mean": video_proc.image_mean,
            "image_std": video_proc.image_std,
            "do_convert_rgb": video_proc.do_convert_rgb,
            "patch_size": video_proc.patch_size,
            "temporal_patch_size": video_proc.temporal_patch_size,
            "merge_size": video_proc.merge_size,
            "do_sample_frames": video_proc.do_sample_frames,
        },
        "resolved_options": {
            "size": img_proc.size,
            "min_pixels": img_proc.min_pixels,
            "max_pixels": img_proc.max_pixels,
            "do_resize": img_proc.do_resize,
            "resample": repr(img_proc.resample),
            "do_rescale": img_proc.do_rescale,
            "rescale_factor": img_proc.rescale_factor,
            "do_normalize": img_proc.do_normalize,
            "image_mean": img_proc.image_mean,
            "image_std": img_proc.image_std,
            "do_convert_rgb": img_proc.do_convert_rgb,
            "patch_size": img_proc.patch_size,
            "temporal_patch_size": img_proc.temporal_patch_size,
            "merge_size": img_proc.merge_size,
        },
        "tokenizer_ids": {
            "image_token_id": processor.image_token_id,
            "video_token_id": processor.video_token_id,
            "vision_start_token_id": processor.vision_start_token_id,
            "vision_end_token_id": processor.vision_end_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "geometry": {},
        "color": {},
        "cases": {},
    }

    def run_case(name: str, image_path: Path, as_file: bool = False,
                 prompt_text: str | None = None) -> dict:
        from PIL import Image as _Image

        source = _Image.open(image_path)
        size = source.size
        batch = img_proc(images=source) if not as_file else img_proc(images=str(image_path))
        grid = [int(v) for v in torch.as_tensor(batch["image_grid_thw"][0]).tolist()]
        h_bar, w_bar = smart_resize(size[1], size[0], factor,
                                    img_proc.min_pixels, img_proc.max_pixels)
        expected_grid = [1, h_bar // img_proc.patch_size, w_bar // img_proc.patch_size]
        n_merged_expected = (grid[0] * grid[1] * grid[2]) // img_proc.merge_size**2
        entry = {
            "source_wh": [size[0], size[1]],
            "expected_resized_hw": [h_bar, w_bar],
            "grid": grid,
            "grid_matches_rederived_smart_resize": grid == expected_grid,
            "merged_token_count_expected": n_merged_expected,
            "slow_processor_reports_num_image_tokens": "num_image_tokens" in batch.keys(),
            "pixel_values": pixel_summary(batch),
        }
        if prompt_text is not None:
            entry["prompt"] = {"text": prompt_text,
                               "sha256": sha256_bytes(prompt_text.encode("utf-8"))}
        return entry

    # Geometry + patch evidence for the two model images.
    prompt_messages = [
        {"role": "user",
         "content": [
             {"type": "image"},
             {"type": "text", "text": "Describe the image in one short sentence."},
         ]}
    ]
    rendered = processor.apply_chat_template(prompt_messages, add_generation_prompt=True,
                                             tokenize=False)
    result["processor"]["prompt_template"] = {
        "rendered": rendered,
        "sha256": sha256_bytes(rendered.encode("utf-8")),
    }
    for name in ("rgb_256x256", "rgb_64x32"):
        result["processor"]["cases"][name] = run_case(name, images[name])
        result["processor"]["cases"][name]["prompt_length_after_expansion"] = None  # set below

    # Color / alpha / EXIF path evidence.
    def color_case(original_input, variants: dict) -> dict:
        import numpy as np

        original = img_proc(images=original_input)
        summary = pixel_summary(original)
        per_variant = {}
        matches = {}
        for label, variant in variants.items():
            out = img_proc(images=variant)
            per_variant[label] = pixel_summary(out)
            matches[label] = bool(np.array_equal(original["pixel_values"],
                                                 out["pixel_values"]))
        return {"pixel_values": summary, "variants": per_variant, "matches": matches}

    def white_composite(image: Image.Image) -> Image.Image:
        white = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "RGBA":
            white.paste(image, mask=image.getchannel("A"))
        else:
            white.paste(image.convert("L").convert("RGB"), mask=image.getchannel("A"))
        return white

    rgba = Image.open(images["rgba_8x8"])
    result["processor"]["color"]["rgba_8x8"] = color_case(
        Image.open(images["rgba_8x8"]),
        {"drop_alpha": rgba.convert("RGB"), "composite_over_white": white_composite(rgba)})
    la = Image.open(images["la_8x8"])
    result["processor"]["color"]["la_8x8"] = color_case(
        Image.open(images["la_8x8"]),
        {"drop_alpha": la.convert("RGB"), "composite_over_white": white_composite(la)})
    palette = Image.open(images["palette_4bit_trns"])
    result["processor"]["color"]["palette_4bit_trns"] = {
        "tRNS_from_file": list(palette.info.get("transparency", [])),
        **color_case(Image.open(images["palette_4bit_trns"]),
                     {"drop_trns_convert_rgb": palette.convert("RGB")})
    }
    raw_exif = Image.open(images["exif_orientation6"])
    orientation = raw_exif.getexif().get(0x0112)
    result["processor"]["color"]["exif_orientation6"] = {
        "exif_orientation_tag": orientation,
        "source_wh": list(raw_exif.size),
        **color_case(
            str(images["exif_orientation6"]),
            {"no_transpose": raw_exif,
             "transpose_90cw": raw_exif.transpose(Image.Transpose.ROTATE_270)},
        ),
        "note": "original input was the file path, so the reference load_image "
        "path (exif_transpose + convert_to_rgb) is exercised",
    }

    # -------------------------------------------------------------------- model
    model_info: dict = {}
    if not args.skip_model:
        from transformers import Qwen3VLForConditionalGeneration

        torch.manual_seed(0)
        started = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(directory), dtype=torch.float32, attn_implementation="eager"
        )
        model.eval()
        load_seconds = time.perf_counter() - started

        llm = model.model
        visual = llm.visual
        text = llm.language_model
        embed = text.embed_tokens

        def params_of(module) -> int:
            return sum(p.numel() for p in module.parameters())

        model_info["loading"] = {
            "storage_dtype": "bfloat16 (safetensors)",
            "loaded_dtype": "float32",
            "attn_implementation": "eager",
            "load_seconds": round(load_seconds, 3),
        }
        model_info["tied_embedding"] = {
            "lm_head_is_embed_tokens": model.lm_head.weight.data_ptr()
            == embed.weight.data_ptr(),
            "tied_weights_keys": config.get("tied_weights_keys"),
        }
        model_info["parameter_counts"] = {
            "visual": params_of(visual),
            "language_model": params_of(text),
            "deepstack_merger_list": params_of(visual.deepstack_merger_list),
        }
        model_info["vision_shapes"] = {
            "pos_embed": list(visual.pos_embed.weight.shape),
            "num_grid_per_side": visual.num_grid_per_side,
            "patch_embed_weight": list(visual.patch_embed.proj.weight.shape),
            "patch_embed_bias": list(visual.patch_embed.proj.bias.shape),
            "deepstack_visual_indexes": list(visual.deepstack_visual_indexes),
            "rotary_head_dim": config["vision_config"]["hidden_size"]
            // config["vision_config"]["num_heads"] // 2,
        }
        layer0 = text.layers[0].self_attn
        model_info["text_shapes"] = {
            "embed_tokens": list(embed.weight.shape),
            "num_layers": len(text.layers),
            "layer0_self_attn_attrs": sorted(vars(layer0).keys()),
            "layer0_q_proj": list(layer0.q_proj.weight.shape),
            "layer0_k_proj": list(layer0.k_proj.weight.shape),
            "layer0_v_proj": list(layer0.v_proj.weight.shape),
            "layer0_o_proj": list(layer0.o_proj.weight.shape),
            "layer0_q_norm": list(layer0.q_norm.weight.shape) if hasattr(layer0, "q_norm") else None,
            "layer0_k_norm": list(layer0.k_norm.weight.shape) if hasattr(layer0, "k_norm") else None,
            "final_norm": list(text.norm.weight.shape),
        }

        capture: dict[str, list[list[int]]] = {}

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(value):
                    capture.setdefault(name, []).append(list(value.shape))

            return hook

        visual.patch_embed.register_forward_hook(make_hook("patch_embed"))
        visual.blocks[0].register_forward_hook(make_hook("block0"))
        visual.blocks[-1].register_forward_hook(make_hook("block_last"))
        visual.merger.register_forward_hook(make_hook("merger"))
        for i, merger in enumerate(visual.deepstack_merger_list):
            merger.register_forward_hook(make_hook(f"deepstack_merger{i}"))

        with torch.no_grad():
            case_results = {}
            for name in ("rgb_256x256", "rgb_64x32"):
                capture.clear()
                started = time.perf_counter()
                image = Image.open(images[name])
                batch = processor(images=[image.convert("RGB")], text=[rendered])
                input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
                attention_mask = torch.as_tensor(batch["attention_mask"], dtype=torch.long)
                pixel_values = torch.as_tensor(batch["pixel_values"], dtype=torch.float32)
                image_grid_thw = torch.as_tensor(batch["image_grid_thw"], dtype=torch.long)
                ids_row = [int(v) for v in input_ids[0].tolist()]
                span_start, span_end = image_token_span(ids_row)
                grid = [int(v) for v in image_grid_thw[0].tolist()]
                n_merged = (grid[0] * grid[1] * grid[2]) // img_proc.merge_size**2
                result["processor"]["cases"][name]["prompt_length_after_expansion"] = len(ids_row)
                assert span_end - span_start + 1 == n_merged, "image span != merged count"

                expected = rederive_rope(ids_row, grid)
                position_ids, rope_deltas = llm.get_rope_index(
                    input_ids, image_grid_thw, None, attention_mask=attention_mask
                )
                actual = [[int(v) for v in row] for row in position_ids[:, 0].tolist()]
                delta = int(rope_deltas[0][0].item())
                assert actual == expected["position_ids"], "position_ids mismatch"
                assert delta == expected["rope_delta"], "rope delta mismatch"

                pos_samples = {}
                for label, idx in (
                    ("first_token", 0),
                    ("last_text_before_image", span_start - 1),
                    ("first_image_token", span_start),
                    ("last_image_token", span_end),
                    ("last_token", len(ids_row) - 1),
                ):
                    if 0 <= idx < len(ids_row):
                        pos_samples[label] = [int(position_ids[a, 0, idx].item())
                                              for a in range(3)]

                visual_forward = {
                    "pixel_values": tensor_summary(pixel_values),
                    "pos_embed_interpolated": list(
                        visual.fast_pos_embed_interpolate(image_grid_thw).shape
                    ),
                    "rot_pos_emb": list(visual.rot_pos_emb(image_grid_thw).shape),
                }

                model.model.rope_deltas = None
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw)
                assert torch.equal(model.model.rope_deltas, rope_deltas), \
                    "model-cached rope_deltas mismatch"
                last_logits = out.logits[0, -1].float().contiguous()
                first_step = {
                    "prefill_sha256": sha256_bytes(last_logits.numpy().tobytes()),
                    "prefill_argmax": int(last_logits.argmax().item()),
                    "prefill_top20": [
                        [int(i), float(v)]
                        for i, v in zip(*torch.topk(last_logits, 20))
                    ],
                    "logits_shape": list(out.logits.shape),
                    "shapes": {k: v[0] for k, v in capture.items()},
                }
                visual_forward["patch_embed_out"] = capture["patch_embed"][0]
                visual_forward["block0_out"] = capture["block0"][0]
                visual_forward["block_last_out"] = capture["block_last"][0]
                visual_forward["merger_out"] = capture["merger"][0]
                visual_forward["deepstack_merger_outs"] = [
                    capture[f"deepstack_merger{i}"][0]
                    for i in range(len(visual.deepstack_merger_list))
                ]

                # Determinism: a fresh prefill must reproduce the logits bit-exactly.
                model.model.rope_deltas = None
                out_repeat = model(input_ids=input_ids, attention_mask=attention_mask,
                                   pixel_values=pixel_values,
                                   image_grid_thw=image_grid_thw)
                repeat_logits = out_repeat.logits[0, -1].float().contiguous()
                determinism_match = sha256_bytes(repeat_logits.numpy().tobytes()) \
                    == first_step["prefill_sha256"]

                past = out.past_key_values
                current = torch.tensor([[first_step["prefill_argmax"]]], dtype=torch.long)
                greedy = [first_step["prefill_argmax"]]
                decode_steps = []
                for step in range(4):
                    cache_position = len(ids_row) + step
                    out_step = model(input_ids=current, past_key_values=past,
                                     cache_position=torch.tensor([cache_position]))
                    past = out_step.past_key_values
                    step_logits = out_step.logits[0, -1].float().contiguous()
                    nxt = int(step_logits.argmax().item())
                    greedy.append(nxt)
                    decode_steps.append({
                        "step": step,
                        "cache_position": cache_position,
                        "kv_length": int(past.get_seq_length()),
                        "expected_decode_positions_all_axes":
                            cache_position + delta,
                        "logits_sha256": sha256_bytes(step_logits.numpy().tobytes()),
                        "argmax": nxt,
                        "top5": [[int(i), float(v)]
                                 for i, v in zip(*torch.topk(step_logits, 5))],
                    })
                case_results[name] = {
                    "sequence": {
                        "length": len(ids_row),
                        "image_token_span": [span_start, span_end],
                        "image_token_count": span_end - span_start + 1,
                        "merged_token_count_expected": n_merged,
                    },
                    "grid": grid,
                    "rope": {
                        "delta": delta,
                        "rederived_match": True,
                        "delta_formula": "max(llm_pos)+1-seq_len = "
                        f"max(llm_grid_h, llm_grid_w) - merged_count = "
                        f"{max(grid[1] // 2, grid[2] // 2)} - {n_merged}",
                        "position_samples": pos_samples,
                        "position_ids_full": actual,
                    },
                    "visual_forward": visual_forward,
                    "prefill": first_step,
                    "determinism": {"repeat_prefill_sha256_match": determinism_match},
                    "decode": decode_steps,
                    "greedy_ids": greedy,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            model_info["cases"] = case_results
        result["model"] = model_info
    else:
        result["model"] = {"skipped": True}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(py(result), indent=2) + "\n")
    print(f"wrote {args.out}")
    for name, case in result["model"].get("cases", {}).items():
        print(f"{name}: grid={case['grid']} delta={case['rope']['delta']} "
              f"greedy={case['greedy_ids']}")


if __name__ == "__main__":
    main()
