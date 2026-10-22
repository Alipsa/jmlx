#!/usr/bin/env python3
"""Phase 7.3b WP5 tiny Qwen3-VL vision checkpoint + golden generator.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0+cpu,
safetensors 0.6.2, Pillow 12.3.0, NumPy 2.5.3, Python 3.12), CPU float32,
single-threaded, deterministic (seed 20261021). Every LayerNorm affine
(blocks.*.norm1/norm2, merger.norm, deepstack_merger_list.*.norm) is
perturbed away from its PyTorch default (weight 1, bias 0) before the
checkpoint is saved -- weight ~ N(1.0, 0.1), bias ~ N(0, 0.05), drawn from a
dedicated seed-20261021 generator -- so the 16 norms are mutually distinct
and the strict golden discriminates the tower's norm-to-key wiring (a
norm1/norm2 swap, a weight/bias swap, the wrong norm on the merger versus a
DeepStack merger, or a checkpoint-key typo would all change the captures).
It is never invoked by Gradle. It writes:

  * a tiny synthetic vision checkpoint at
    tools/hf-reference/goldens/checkpoints/qwen3vl-vision/
    (config.json, model.safetensors with bf16 tensors keyed with the real
    artifact's `model.visual.` prefix, and four deterministic PNGs under
    images/);
  * the golden tools/hf-reference/goldens/qwen3vl-vision-wp5.json capturing
    pure-math bit probes (linspace double bits, vision inv_freq/cos/sin
    float32 bits, the first-fully-fractional-patch bilinear
    position-embedding sample) plus one packed processor call and one packed
    Qwen3VLVisionModel forward over all four images. The full-packed
    merger/DeepStack arrays ([36, 256] each, image order) are stored exactly
    once in the top-level "packed" section; each case stores only its
    per-image captures plus merged_row_offset/merged_rows slice pointers.

Every written file's SHA-256 is printed at the end and pinned in
tools/hf-reference/provenance.json `files` (keys relative to goldens/).

PNG determinism: the pixel arrays are fixed affine functions of (x, y) (no
RNG) and the PNGs are written through Pillow's fixed encoder path
(Image.frombytes("RGB", ...).save(path, format="PNG")); Pillow emits no
timestamp or per-run metadata for PNG, so identical pixels yield identical
bytes on a pinned Pillow.

Deviation record (transformers 4.57.6): there is no Qwen3VLImageProcessor
class or module in 4.57.6; the reference preprocessing uses the slow PIL
Qwen2VLImageProcessor from transformers.models.qwen2_vl (the same class the
WP1 probe and the jmlx-vision Qwen3Vl* port target), constructed with
explicit min_pixels/max_pixels because the 4.57.6 __init__ else branch
overwrites any (or no) size dict with the 3136/1003520 defaults (findings
§4.2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import struct
import sys
from pathlib import Path

SEED = 20261021

# (name, width, height, per-channel affine coefficients (a, b, k) with
# channel = (a*x + b*y + k) % 256 for x = column, y = row). Coefficients are
# distinct per image and per channel, every y coefficient keeps all rows
# distinct, and every channel provably spans the full 0-255 range for these
# dimensions (the script re-verifies both properties and exits loudly if a
# future dimension change breaks them).
IMAGE_SPECS = [
    ("square", 128, 128, ((1, 2, 0), (2, 1, 0), (3, 5, 17))),
    ("wide", 96, 64, ((2, 3, 5), (1, 7, 13), (3, 11, 29))),
    ("tall", 64, 96, ((5, 2, 3), (2, 3, 7), (7, 5, 11))),
    ("narrow", 64, 128, ((2, 3, 21), (1, 7, 31), (3, 5, 41))),
]

# Expected processor grids per image, [t, h, w]: all dimensions are multiples
# of 32 and every area is within [min_pixels, max_pixels], so smart_resize is
# a no-op and the grid is exactly image dims / patch_size 16.
EXPECTED_GRIDS = {
    "square": [1, 8, 8],
    "wide": [1, 4, 6],
    "tall": [1, 6, 4],
    "narrow": [1, 8, 4],
}

# Patch-row offsets of the per-image slices inside the packed [144, 256]
# intermediate tensors, in packed order square, wide, tall, narrow.
PATCH_OFFSETS = {"square": 0, "wide": 64, "tall": 88, "narrow": 112}

KEY_PREFIX = "model.visual."


def gate_versions() -> dict:
    import PIL
    import numpy as np
    import torch
    import transformers
    from safetensors import __version__ as safetensors_version

    if (
        transformers.__version__ != "4.57.6"
        or PIL.__version__ != "12.3.0"
        or np.__version__ != "2.5.3"
        or torch.__version__ not in ("2.6.0", "2.6.0+cpu")
        or platform.python_version_tuple()[:2] != ("3", "12")
    ):
        sys.exit(
            "qwen3vl_vision requires transformers 4.57.6, Pillow 12.3.0, NumPy 2.5.3, "
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
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_hex(values) -> list[str]:
    """Exact float32 hex bits (8 lowercase hex chars) per value."""
    return [
        f"{struct.unpack('<I', struct.pack('<f', float(v)))[0]:08x}"
        for v in values
    ]


def rounded(value):
    """Round float values to 7 significant digits before JSON storage.

    Same convention as generate.py's goldens (idefics3.json): a raw float32
    value's float64 repr needs up to 17 significant digits, which would make
    the golden ~3x larger than the established goldens for the same number of
    floats. Bit-exact probes stay hex-encoded and are not rounded."""
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.7g}")
    return value


def f64_hex(values) -> list[str]:
    """Exact double hex bits (16 lowercase hex chars) per value."""
    return [
        f"{struct.unpack('<Q', struct.pack('<d', float(v)))[0]:016x}"
        for v in values
    ]


def py(obj):
    """Make numpy/torch values JSON-serializable (probe convention)."""
    if isinstance(obj, dict):
        return {k: py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    import torch

    if isinstance(obj, torch.Tensor):
        return py(obj.tolist())
    if hasattr(obj, "item"):
        return py(obj.item())
    return str(obj)


def build_image(width: int, height: int, coeffs, path: Path) -> None:
    """Write one deterministic RGB PNG; verify full-range and row distinctness."""
    from PIL import Image

    a_r, b_r, k_r = coeffs[0]
    a_g, b_g, k_g = coeffs[1]
    a_b, b_b, k_b = coeffs[2]
    px = bytearray()
    for y in range(height):
        for x in range(width):
            px += bytes(
                ((a_r * x + b_r * y + k_r) % 256,
                 (a_g * x + b_g * y + k_g) % 256,
                 (a_b * x + b_b * y + k_b) % 256)
            )
    # Full 0-255 range per channel and no two rows collide (row bytes differ).
    for channel in range(3):
        values = {px[i * 3 + channel] for i in range(width * height)}
        if values != set(range(256)):
            sys.exit(f"{path.name}: channel {channel} covers {len(values)} of 256 values")
    row_bytes = [bytes(px[y * width * 3:(y + 1) * width * 3]) for y in range(height)]
    if len(set(row_bytes)) != height:
        sys.exit(f"{path.name}: two rows collide")
    Image.frombytes("RGB", (width, height), bytes(px)).save(path, format="PNG")


def vision_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    return Qwen3VLVisionConfig(
        depth=6,
        hidden_size=256,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=512,
        num_heads=4,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=256,
        num_position_embeddings=64,
        deepstack_visual_indexes=[1, 3, 5],
        initializer_range=0.02,
    )


def perturb_layer_norms(model, seed: int) -> int:
    """Perturb every LayerNorm affine with a dedicated seeded generator.

    The reference's model init leaves every LayerNorm at its PyTorch default
    (weight 1, bias 0): with all 16 norms identical, the strict golden would
    still pass if the tower swapped norm1 and norm2, swapped a norm's weight
    and bias, used the wrong norm for the merger versus a DeepStack merger,
    or mistyped a norm's checkpoint key. Drawing each affine from a dedicated
    torch.Generator (seeded once, draws in model.modules() order, weight then
    bias per norm) keeps the perturbation deterministic and isolated from the
    global RNG the model init already consumed. Returns the norm count.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.1, generator=gen)
            torch.nn.init.normal_(module.bias, mean=0.0, std=0.05, generator=gen)
            count += 1
    return count


def slow_processor():
    # The pinned 4.57.6 slow PIL processor has no Qwen3VLImageProcessor
    # counterpart; the explicit min_pixels/max_pixels compensate for the
    # __init__ size-dict overwrite (findings §4.2). Options mirror the
    # jmlx-vision Qwen3VlProcessorConfig effective defaults.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    return Qwen2VLImageProcessor(
        min_pixels=3136,
        max_pixels=12845056,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        resample=3,
        rescale_factor=1 / 255,
        do_convert_rgb=True,
        do_rescale=True,
        do_resize=True,
        do_normalize=True,
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=script_dir / "goldens" / "checkpoints" / "qwen3vl-vision")
    parser.add_argument("--out", type=Path,
                        default=script_dir / "goldens" / "qwen3vl-vision-wp5.json")
    args = parser.parse_args()

    import torch
    from safetensors import torch as st
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel,
        Qwen3VLVisionRotaryEmbedding,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    environment = gate_versions()
    checkpoint_dir: Path = args.checkpoint_dir
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    (checkpoint_dir / "images").mkdir(parents=True)

    # ------------------------------------------------------------- checkpoint
    torch.manual_seed(SEED)
    cfg = vision_config()
    model = Qwen3VLVisionModel(cfg)
    # The model init leaves every LayerNorm at its PyTorch default (weight 1,
    # bias 0): with all norms identical the strict golden could not catch a
    # norm1/norm2 swap, a weight/bias swap, the wrong norm on the merger
    # versus a DeepStack merger, or a norm-key typo. Perturb before saving.
    norm_count = perturb_layer_norms(model, SEED)
    expected_norms = 2 * cfg.depth + 1 + len(cfg.deepstack_visual_indexes)
    if norm_count != expected_norms:
        sys.exit(f"expected {expected_norms} LayerNorm affines to perturb, found {norm_count}")
    cfg.save_pretrained(checkpoint_dir)

    bare_state = model.state_dict()
    if any(k.startswith(KEY_PREFIX) for k in bare_state):
        sys.exit("bare Qwen3VLVisionModel state dict already carries the model.visual. prefix")
    bf16_state = {KEY_PREFIX + k: v.to(torch.bfloat16) for k, v in bare_state.items()}
    st.save_file(bf16_state, checkpoint_dir / "model.safetensors")

    image_paths = {}
    for name, width, height, coeffs in IMAGE_SPECS:
        path = checkpoint_dir / "images" / f"{name}.png"
        build_image(width, height, coeffs, path)
        image_paths[name] = path

    # ------------------------------------------------------- reference model
    # The reference run loads the SAVED checkpoint back (not the in-memory
    # model) and upconverts the stored bf16 weights to float32, so the golden
    # exercises the exact stored-bf16 -> float32 path the Java tower uses.
    saved = st.load_file(checkpoint_dir / "model.safetensors")
    ref_cfg = vision_config()
    ref_cfg._attn_implementation = "eager"
    ref = Qwen3VLVisionModel(ref_cfg)
    ref.load_state_dict({k[len(KEY_PREFIX):]: v.float() for k, v in saved.items()})
    ref.eval()

    # ------------------------------------------------------------ processor
    proc = slow_processor()
    file_paths = [str(image_paths[name]) for name, _, _, _ in IMAGE_SPECS]
    batch = proc.preprocess(file_paths, return_tensors="pt")
    pixel_values = batch["pixel_values"].numpy()
    grid_thw = batch["image_grid_thw"].numpy()
    if pixel_values.shape != (144, 1536) or pixel_values.dtype != "float32":
        sys.exit(f"packed pixel_values {pixel_values.shape} {pixel_values.dtype} != (144, 1536) float32")
    observed = [list(map(int, row)) for row in grid_thw]
    expected = [EXPECTED_GRIDS[name] for name, _, _, _ in IMAGE_SPECS]
    if observed != expected:
        sys.exit(f"packed grid_thw {observed} != expected {expected}")

    # ------------------------------------------------------------- forward
    captures: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def _hook(_module, _inputs, output):
            captures[name] = output

        return _hook

    ref.patch_embed.register_forward_hook(hook("patch_embed"))
    for block_index in (0, 1, 3, 5):
        ref.blocks[block_index].register_forward_hook(hook(f"block{block_index}"))

    with torch.no_grad():
        pos_embed_interpolated = ref.fast_pos_embed_interpolate(
            torch.from_numpy(grid_thw).int()
        )
        merged, deepstack_lists = ref(
            torch.from_numpy(pixel_values).float(),
            grid_thw=torch.from_numpy(grid_thw).int(),
        )
    if list(pos_embed_interpolated.shape) != [144, 256]:
        sys.exit(f"pos_embed_interpolated {tuple(pos_embed_interpolated.shape)} != [144, 256]")
    if list(merged.shape) != [36, 256]:
        sys.exit(f"merged {tuple(merged.shape)} != [36, 256]")
    if len(deepstack_lists) != 3 or any(
        list(d.shape) != [36, 256] for d in deepstack_lists
    ):
        sys.exit("deepstack_feature_lists != 3 x [36, 256]")
    for name in ("patch_embed", "block0", "block1", "block3", "block5"):
        if list(captures[name].shape) != [144, 256]:
            sys.exit(f"{name} {tuple(captures[name].shape)} != [144, 256]")

    packed: dict[str, torch.Tensor] = {
        "pos_embed_interpolated": pos_embed_interpolated,
        "patch_embed": captures["patch_embed"],
        "block0": captures["block0"],
        "block1": captures["block1"],
        "block3": captures["block3"],
        "block5": captures["block5"],
        "merger": merged,
        "deepstack0": deepstack_lists[0],
        "deepstack1": deepstack_lists[1],
        "deepstack2": deepstack_lists[2],
    }

    # ---------------------------------------------------------------- probes
    linspace_probe = []
    for n in (4, 6, 8):
        values = torch.linspace(0, 7, n)
        linspace_probe.append({"n": n, "values_hex": f64_hex(values.tolist())})

    rot = Qwen3VLVisionRotaryEmbedding(32, 10000.0)
    if list(rot.inv_freq.shape) != [16]:
        sys.exit(f"inv_freq {tuple(rot.inv_freq.shape)} != [16]")
    inv_freq_hex = f32_hex(rot.inv_freq.tolist())

    cos_sin_sample = []
    for position in (0, 1, 2, 3, 5, 7):
        for j in (0, 1, 2, 3):
            v = torch.tensor(float(position), dtype=torch.float32) * rot.inv_freq[j]
            cos_sin_sample.append({
                "p": position,
                "j": j,
                "v_hex": f32_hex([float(v)])[0],
                "cos_hex": f32_hex([float(v.cos())])[0],
                "sin_hex": f32_hex([float(v.sin())])[0],
            })

    side = ref.num_grid_per_side
    if side != 8:
        sys.exit(f"num_grid_per_side {side} != 8")
    merge = ref.config.spatial_merge_size
    pos_embed_sample = {}
    for name in ("wide", "tall"):
        grid_h, grid_w = EXPECTED_GRIDS[name][1], EXPECTED_GRIDS[name][2]
        # Mirror fast_pos_embed_interpolate's per-axis double linspace exactly,
        # then select the FIRST patch in merge-block order whose row AND column
        # linspace values are both fractional (the grid-(0,0) patch is a
        # degenerate exact-corner case that would pin no bilinear math).
        h_idxs = torch.linspace(0, side - 1, grid_h)
        w_idxs = torch.linspace(0, side - 1, grid_w)
        selected = None
        for bh in range(grid_h // merge):
            for bw in range(grid_w // merge):
                for dh in range(merge):
                    for dc in range(merge):
                        ph, pw = bh * merge + dh, bw * merge + dc
                        h_idx = float(h_idxs[ph])
                        w_idx = float(w_idxs[pw])
                        if h_idx - int(h_idx) != 0.0 and w_idx - int(w_idx) != 0.0:
                            row_in_image = (
                                (bh * (grid_w // merge) + bw) * merge * merge
                                + dh * merge + dc
                            )
                            selected = (ph, pw, h_idx, w_idx, row_in_image)
                            break
                    if selected:
                        break
                if selected:
                    break
            if selected:
                break
        if selected is None:
            sys.exit(f"{name}: no patch with both axis values fractional")
        ph, pw, h_idx, w_idx, row_in_image = selected
        h_floor, w_floor = int(h_idx), int(w_idx)
        h_ceil = min(h_floor + 1, side - 1)
        w_ceil = min(w_floor + 1, side - 1)
        dh, dw = h_idx - h_floor, w_idx - w_floor
        weights_double = [
            (1 - dh) * (1 - dw),
            (1 - dh) * dw,
            dh * (1 - dw),
            dh * dw,
        ]
        weights_f32 = torch.tensor(weights_double, dtype=torch.float32)
        if not all(0.0 < w < 1.0 for w in weights_f32.tolist()):
            sys.exit(f"{name}: sample weights {weights_f32.tolist()} not strictly fractional")
        corner_indices = [
            h_floor * side + w_floor,
            h_floor * side + w_ceil,
            h_ceil * side + w_floor,
            h_ceil * side + w_ceil,
        ]
        corner_rows = ref.pos_embed.weight[torch.tensor(corner_indices)]
        # Sequential float32 corner add, exactly as the reference does
        # (pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]).
        weighted = corner_rows * weights_f32[:, None]
        expected_row = weighted[0] + weighted[1] + weighted[2] + weighted[3]
        # Ground truth: this patch's row in the image's packed, permuted output.
        actual_row = pos_embed_interpolated[PATCH_OFFSETS[name] + row_in_image]
        max_err = float((expected_row - actual_row).abs().max())
        if max_err > 1e-6:
            sys.exit(f"{name}: corner recombination max err {max_err} > 1e-6")
        pos_embed_sample[name] = {
            "grid": [ph, pw],
            "corner_indices": corner_indices,
            "corner_weights_hex": f32_hex(weights_f32.tolist()),
            "corner_rows": rounded(corner_rows.tolist()),
            "expected_row": rounded(actual_row.tolist()),
        }

    # ----------------------------------------------------------------- cases
    # The reference computes the merger and DeepStack arrays once over the
    # packed rows (image order), so they are stored exactly once in the
    # top-level "packed" section; each case stores only its per-image
    # captures plus the merged-row slice pointer into the packed arrays.
    per_image_keys = (
        "pos_embed_interpolated", "patch_embed", "block0", "block1", "block3", "block5"
    )
    packed_section = {}
    for key in ("merger", "deepstack0", "deepstack1", "deepstack2"):
        rows = packed[key].detach().cpu().float().contiguous()
        packed_section[key] = {"shape": list(rows.shape), "values": rounded(rows.tolist())}

    cases = []
    merged_row_offset = 0
    for name, width, height, _coeffs in IMAGE_SPECS:
        grid = EXPECTED_GRIDS[name]
        n_patches = grid[1] * grid[2]
        merged_rows = n_patches // 4
        offset = PATCH_OFFSETS[name]
        case_pixel_values = pixel_values[offset:offset + n_patches]
        case: dict = {
            "name": name,
            "file": f"images/{name}.png",
            "width": width,
            "height": height,
            "grid": grid,
            "n_patches": n_patches,
            "merged_rows": merged_rows,
            "merged_row_offset": merged_row_offset,
            "pixel_values_sha256": sha256_bytes(case_pixel_values.tobytes()),
            "captures": {},
        }
        for key in per_image_keys:
            rows = packed[key][offset:offset + n_patches].detach().cpu().float().contiguous()
            case["captures"][key] = {
                "shape": list(rows.shape),
                "values": rounded(rows.tolist()),
            }
        merged_row_offset += merged_rows
        cases.append(case)
    if merged_row_offset != 36:
        sys.exit(f"packed merged rows {merged_row_offset} != 36")

    # ------------------------------------------------------------- golden
    result = {
        "schema": "jmlx-phase7-3b-wp5-qwen3vl-vision-v1",
        "generated_by": "tools/hf-reference/qwen3vl_vision.py",
        "provenance": {
            "transformers": environment["transformers"],
            "torch": environment["torch"],
            "safetensors": environment["safetensors"],
            "python": environment["python"],
            "platform": environment["platform"],
            "pillow": environment["pillow"],
            "numpy": environment["numpy"],
            "seed": SEED,
            "norm_perturbation": {
                "seed": SEED,
                "generator": "dedicated torch.Generator (does not consume the global model-init RNG)",
                "weight_distribution": "normal(mean=1.0, std=0.1)",
                "bias_distribution": "normal(mean=0.0, std=0.05)",
                "count": norm_count,
                "note": (
                    "every LayerNorm affine (blocks.*.norm1/norm2, merger.norm, "
                    "deepstack_merger_list.*.norm) is perturbed before the "
                    "checkpoint is saved so the 16 norms are mutually distinct "
                    "and the strict golden discriminates the tower's norm "
                    "wiring (a norm1/norm2 swap, a weight/bias swap, the wrong "
                    "norm on the merger versus a DeepStack merger, or a "
                    "checkpoint-key typo would all change the captures)"
                ),
            },
            "config": py(vision_config().to_dict()),
            "checkpoint_files": {
                "config.json": sha256_file(checkpoint_dir / "config.json"),
                "model.safetensors": sha256_file(checkpoint_dir / "model.safetensors"),
            },
            "images": {
                name: {
                    "width": width,
                    "height": height,
                    "bytes": image_paths[name].stat().st_size,
                    "sha256": sha256_file(image_paths[name]),
                }
                for name, width, height, _coeffs in IMAGE_SPECS
            },
            "key_prefix_note": (
                "the bare Qwen3VLVisionModel state dict carries plain keys "
                "(patch_embed.proj.*, pos_embed.weight, blocks.*, merger.*, "
                "deepstack_merger_list.*); the generator prepends 'model.visual.' "
                "to every key so the tiny checkpoint matches the real Qwen3-VL "
                "artifact key scheme (model.visual.*); all 99 tensors are stored "
                "as bfloat16"
            ),
            "processor": {
                "class": "Qwen2VLImageProcessor",
                "module": "transformers.models.qwen2_vl.image_processing_qwen2_vl",
                "note": (
                    "transformers 4.57.6 has no Qwen3VLImageProcessor class; the "
                    "slow PIL Qwen2VLImageProcessor is the pinned preprocessing "
                    "path (WP1 probe precedent and jmlx-vision Qwen3Vl* port "
                    "target). min_pixels/max_pixels are passed explicitly because "
                    "the 4.57.6 __init__ overwrites any size dict with the "
                    "3136/1003520 defaults (findings §4.2). Images are passed as "
                    "file path strings so the load_image (exif_transpose) path "
                    "matches the Java processFile path; all areas lie inside "
                    "[3136, 12845056] so smart_resize is a no-op."
                ),
                "min_pixels": proc.min_pixels,
                "max_pixels": proc.max_pixels,
                "size": proc.size,
                "do_resize": proc.do_resize,
                "resample": repr(proc.resample),
                "do_rescale": proc.do_rescale,
                "rescale_factor": proc.rescale_factor,
                "do_normalize": proc.do_normalize,
                "image_mean": proc.image_mean,
                "image_std": proc.image_std,
                "do_convert_rgb": proc.do_convert_rgb,
                "patch_size": proc.patch_size,
                "temporal_patch_size": proc.temporal_patch_size,
                "merge_size": proc.merge_size,
            },
        },
        "probes": {
            "linspace": linspace_probe,
            "inv_freq": {
                "dim": 32,
                "theta": 10000.0,
                "values_hex": inv_freq_hex,
            },
            "cos_sin_sample": cos_sin_sample,
            "pos_embed_sample": pos_embed_sample,
        },
        "packed": packed_section,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Compact single-line JSON, matching the capture-heavy golden convention
    # (idefics3.json): indent=2 would add a line per float and roughly double
    # the file size for no readability gain at this density.
    args.out.write_text(json.dumps(py(result)) + "\n")

    # -------------------------------------------------------------- summary
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    print(f"config.json: {sha256_file(checkpoint_dir / 'config.json')}")
    print(f"model.safetensors: {sha256_file(checkpoint_dir / 'model.safetensors')} "
          f"({len(bf16_state)} tensors, bf16)")
    for name, _width, _height, _coeffs in IMAGE_SPECS:
        path = image_paths[name]
        print(f"{path.name}: {sha256_file(path)} ({path.stat().st_size} bytes)")
    print(f"grids: {observed}")
    print(f"pixel_values_sha256: {sha256_bytes(pixel_values.tobytes())}")


if __name__ == "__main__":
    main()
