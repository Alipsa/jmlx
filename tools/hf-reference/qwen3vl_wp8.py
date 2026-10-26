#!/usr/bin/env python3
"""Phase 7.3b WP8 tiny Qwen3-VL full-model checkpoint + acceptance golden generator.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0+cpu,
safetensors 0.6.2, Pillow 12.3.0, NumPy 2.5.3, Python 3.12), CPU float32,
single-threaded, deterministic (seed 20261025). It writes:

  * a tiny synthetic full Qwen3-VL checkpoint at
    tools/hf-reference/goldens/checkpoints/qwen3vl-wp8/
    (the strict 10/18/14-key config.json, a matching 14-key
    preprocessor_config.json, the committed 21-token qwen3vl fixture
    tokenizer, model.safetensors with bf16 tensors keyed with the real
    artifact's `model.visual.` / `model.language_model.` prefix and no
    lm_head -- the head is tied -- and two deterministic PNGs under
    images/);
  * the golden tools/hf-reference/goldens/qwen3vl-wp8.json capturing the
    plan Section 9 acceptance cases:

    - "one-image" (image a), greedy prefill + four decode steps;
    - "two-image" (a, b, different sizes), prefill + three decode steps;
    - "image-text-image-text" (a, b with intervening and trailing text),
      prefill + three decode steps;
    - "text-only" through the same model, prefill + three decode steps;
    - "forced image-bearing" and "forced text-only": the same forced
      continuation [7, 10, 11, 12, 13, 12, 10, 13, 6] -- every
      image/video/vision-start/vision-end special ID plus a complete
      wrapper-looking generated sequence -- on an image-bearing and a
      text-only prompt, nine steps each.

    For every case the golden records the unexpanded/expanded IDs, the
    grids, the pixel values' SHA-256, the full three-axis prefill
    positions and delta, the tower's merged and DeepStack features (base
    cases), the prefill logits, and, for every decode step, the greedy
    token ID and a fresh full uncached forward's last-position logits.

Every decode/forced step is computed with the explicit reference
forward (`explicit_full_forward` below), never through the standard
processor path: the tower runs once from the ORIGINAL pixel values, the
image rows are replaced and the DeepStack levels added only at the
original prompt's destinations, and the positions are the recorded
prefill axes followed by `logicalPosition + delta` for every generated
token. Token-based placeholder detection and get_rope_index are therefore
never re-run over generated markers (a generated wrapper-looking run is
ordinary history). At the zero-history point the explicit forward is
asserted to equal the standard processor forward (1e-6), pinning the
explicit path's faithfulness.

The geometry deliberately exercises the WP6 checkpoint's blind spots:

  * three decoder layers with two DeepStack levels: injections after
    blocks 0 and 1, block 2 un-injected -- every injection position kind
    is exercised, plus a non-injected tail layer;
  * mrope_section [16, 8, 8] (2 * 32 = 64 = head_dim): a strictly
    unequal three-way split;
  * num_position_embeddings 64 (an 8 x 8 learned table) with grids that
    reach 12 along one axis: the bilinear position interpolation runs
    beyond the table on both images (a: width 12, b: height 12);
  * unequal, anisotropic grids: a is 192 x 128 (grid [1, 8, 12], 96 raw
    rows) and b is 128 x 192 (grid [1, 12, 8], 96 raw rows).

Every norm affine is perturbed away from its PyTorch default before the
checkpoint is saved -- every nn.LayerNorm (vision blocks.*.norm1/norm2,
merger.norm, deepstack_merger_list.*.norm): weight ~ N(1.0, 0.1), bias ~
N(0, 0.05); every Qwen3VLTextRMSNorm (weight only): weight ~ N(1.0, 0.1)
-- drawn from a dedicated seed-20261025 generator in model.modules()
order (weight then bias per norm), so the 24 norms (11 + 13) are mutually
distinct and the strict golden discriminates the composite model's
norm-to-key wiring.

It is never invoked by Gradle. Every written file's SHA-256 is printed at
the end and pinned in tools/hf-reference/provenance.json (keys
relative to goldens/); the generator's own SHA-256 is pinned under
`qwen3vl_wp8_generator_sha256`.

Deviation record (transformers 4.57.6): identical to the WP6 generator --
the slow PIL Qwen2VLImageProcessor is the pinned preprocessing path (4.57.6
has no Qwen3VLImageProcessor), the explicit "eager" attention
implementation is used, the tie is applied explicitly (4.57.6's
tie_weights() does not tie the lm_head), and the reference model
overrides text_config.dtype to float32 over the stored bf16 weights --
the exact path the Java composite model takes when it upcasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

SEED = 20261025

# The committed qwen3vl fixture sentinels (qwen3vl-tokens/tokenizer.json):
# image 10, video 11, vision start 12, vision end 13.
IMAGE_TOKEN = 10
VIDEO_TOKEN = 11
START_TOKEN = 12
END_TOKEN = 13
MERGE = 2

# (name, width, height, per-channel affine coefficients (a, b, k) with
# channel = (a*x + b*y + k) % 256 for x = column, y = row). The x
# coefficient 1 makes each row a consecutive mod-256 run; the odd y
# coefficients keep the per-row residues distinct, so every channel covers
# the full 0-255 range across the image and no two rows collide (the
# script re-verifies both properties and exits loudly if a dimension change
# breaks them).
IMAGE_SPECS = [
    ("a", 192, 128, ((1, 5, 11), (1, 7, 17), (1, 9, 29))),
    ("b", 128, 192, ((1, 5, 23), (1, 11, 31), (1, 13, 41))),
]

# Expected processor grids per image, [t, h, w]: all dimensions are
# multiples of 32 and every area is within [min_pixels, max_pixels], so
# smart_resize is a no-op and the grid is exactly image dims / patch_size 16.
# a's width axis and b's height axis reach 12, beyond the 8 x 8 learned
# position table (num_position_embeddings 64), so the bilinear
# fast_pos_embed_interpolate runs on both.
EXPECTED_GRIDS = {
    "a": [1, 8, 12],
    "b": [1, 12, 8],
}

# The four base prompts (plan Section 9 cases 1-4) and the shared forced
# continuation (case 5, image-bearing and text-only): it starts in the
# ordinary text token 7, then every sentinel (image 10, video 11, vision
# start 12, vision end 13) plus the complete wrapper-looking run 12, 10,
# 13. The trailing 6 is only the last golden step's token_id label: every
# sentinel and the whole 12, 10, 13 wrapper run is fed back as history,
# but the label of the final step is never consumed.
TEXT_PROMPT = [3, 4, 5, 6, 7, 3, 4, 5, 6, 3, 19, 7]
ONE_IMAGE_PROMPT = [START_TOKEN, IMAGE_TOKEN, END_TOKEN, 3, 4, 5, 6, 7]
TWO_IMAGE_PROMPT = [
    START_TOKEN,
    IMAGE_TOKEN,
    END_TOKEN,
    START_TOKEN,
    IMAGE_TOKEN,
    END_TOKEN,
    3,
    19,
    7,
]
IMAGE_TEXT_IMAGE_TEXT_PROMPT = [
    3,
    START_TOKEN,
    IMAGE_TOKEN,
    END_TOKEN,
    4,
    5,
    START_TOKEN,
    IMAGE_TOKEN,
    END_TOKEN,
    6,
    7,
]
FORCED_HISTORY = [7, 10, 11, 12, 13, 12, 10, 13, 6]

# The strict hand-written config (10 outer keys, 18 text keys, 14 vision
# keys), mirroring the tiny-checkpoint template at the WP8 geometry.
CONFIG_JSON = {
    "model_type": "qwen3_vl",
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "qwen3_vl_text",
        "hidden_size": 256,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "vocab_size": 21,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.000001,
        "rope_theta": 5000000,
        "rope_scaling": {
            "rope_type": "default",
            "mrope_interleaved": True,
            "mrope_section": [16, 8, 8],
        },
        "attention_bias": False,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "initializer_range": 0.02,
        "use_cache": True,
    },
    "vision_config": {
        "model_type": "qwen3_vl",
        "depth": 4,
        "hidden_size": 256,
        "num_heads": 4,
        "intermediate_size": 512,
        "in_channels": 3,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 64,
        "out_hidden_size": 256,
        "deepstack_visual_indexes": [1, 3],
        "hidden_act": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
    },
    "image_token_id": IMAGE_TOKEN,
    "video_token_id": VIDEO_TOKEN,
    "vision_start_token_id": START_TOKEN,
    "vision_end_token_id": END_TOKEN,
    "transformers_version": "4.57.6",
}

# The 14-key preprocessor config (the tiny template's shape) with the
# processor's effective min/max pixel bounds.
PREPROCESSOR_JSON = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "Qwen2VLImageProcessor",
    "image_std": [0.5, 0.5, 0.5],
    "merge_size": 2,
    "patch_size": 16,
    "processor_class": "Qwen3VLProcessor",
    "resample": 3,
    "rescale_factor": 1 / 255,
    "size": {"shortest_edge": 3136, "longest_edge": 12845056},
    "temporal_patch_size": 2,
}


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
            "qwen3vl_wp8 requires transformers 4.57.6, Pillow 12.3.0, NumPy 2.5.3, "
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


def rounded(value):
    """Round float values to 7 significant digits before JSON storage."""
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.7g}")
    return value


def py(obj):
    """Make numpy/torch values JSON-serializable (probe convention)."""
    if obj is None:
        return None
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


def expected_state_keys(layers: int, depth: int, deepstack: list[int]) -> set[str]:
    """The checkpoint's exact key set (the full model state dict minus the
    tied lm_head), derived from the config geometry."""
    keys = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    for i in range(layers):
        layer = f"model.language_model.layers.{i}."
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            keys.add(layer + name)
    keys.add("model.visual.patch_embed.proj.weight")
    keys.add("model.visual.patch_embed.proj.bias")
    keys.add("model.visual.pos_embed.weight")
    for i in range(depth):
        block = f"model.visual.blocks.{i}."
        for name in (
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias",
        ):
            keys.add(block + name)
    for merger in ["merger"] + [f"deepstack_merger_list.{j}" for j in range(len(deepstack))]:
        for name in (
            "norm.weight",
            "norm.bias",
            "linear_fc1.weight",
            "linear_fc1.bias",
            "linear_fc2.weight",
            "linear_fc2.bias",
        ):
            keys.add(f"model.visual.{merger}.{name}")
    return keys


def perturb_norms(model, seed: int) -> dict:
    """Perturb every norm affine with a dedicated seeded generator (see the
    module docstring for the distributions and order). Returns the per-kind
    counts."""
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

    gen = torch.Generator().manual_seed(seed)
    counts = {"layer_norm": 0, "text_rms_norm": 0}
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.1, generator=gen)
            torch.nn.init.normal_(module.bias, mean=0.0, std=0.05, generator=gen)
            counts["layer_norm"] += 1
        elif isinstance(module, Qwen3VLTextRMSNorm):
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.1, generator=gen)
            counts["text_rms_norm"] += 1
    return counts


def slow_processor():
    # The pinned 4.57.6 slow PIL processor has no Qwen3VLImageProcessor
    # counterpart; the explicit min_pixels/max_pixels compensate for the
    # __init__ size-dict overwrite (findings Section 4.2).
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


def expand_prompt(unexpanded: list[int], grids: list[list[int]]) -> list[int]:
    """The Java Qwen3VlPromptExpander's ID-level expansion, mirrored: each
    vision start, image, vision end wrapper's image token becomes
    grid.prod() // merge^2 copies of the image token, in prompt order."""
    merged = [t * h * w // (MERGE * MERGE) for (t, h, w) in grids]
    if VIDEO_TOKEN in unexpanded:
        sys.exit("prompt contains the video token")
    out: list[int] = []
    image_index = 0
    i = 0
    while i < len(unexpanded):
        token = unexpanded[i]
        if token == START_TOKEN:
            if (
                i + 2 >= len(unexpanded)
                or unexpanded[i + 1] != IMAGE_TOKEN
                or unexpanded[i + 2] != END_TOKEN
            ):
                sys.exit(f"wrapper grammar violated at index {i}")
            out.append(START_TOKEN)
            out.extend([IMAGE_TOKEN] * merged[image_index])
            out.append(END_TOKEN)
            image_index += 1
            i += 3
        elif token in (IMAGE_TOKEN, END_TOKEN):
            sys.exit(f"unwrapped sentinel {token} at index {i}")
        else:
            out.append(token)
            i += 1
    if image_index != len(grids):
        sys.exit(f"prompt holds {image_index} wrapper(s) but {len(grids)} grid(s) given")
    return out


def image_token_positions(expanded: list[int]) -> list[int]:
    """The prompt indices of every image token, in order: the destinations
    every visual replacement and DeepStack addition is restricted to."""
    return [i for i, token in enumerate(expanded) if token == IMAGE_TOKEN]


def explicit_full_forward(
    ref,
    ids: list[int],
    pixels,
    grids,
    visual_destinations: list[int],
    positions: list[list[int]],
):
    """The fresh, full, uncached reference forward over the WHOLE history.

    The tower's outputs are applied only at the ORIGINAL prompt's image
    destinations (`visual_destinations`, a subset of the first
    `len(expanded_prompt)` positions); generated tokens are ordinary
    embeddings, so a generated wrapper-looking sequence never consumes
    pixels, re-runs the tower, or receives DeepStack. `positions` is the
    full explicit three-axis table: the recorded prefill axes followed by
    `logicalPosition + delta` per generated token. The causal mask is the
    reference's own (built from the temporal axis by
    create_causal_mask, exactly as the standard path does).
    """
    import torch

    length = len(ids)
    model = ref.model
    ids_t = torch.tensor([ids], dtype=torch.long)
    embedded = model.get_input_embeddings()(ids_t)
    visual_pos_masks = None
    deepstack_visual_embeds = None
    if pixels is not None:
        image_embeds, deepstack_image_embeds = model.get_image_features(pixels, grids)
        image_embeds = torch.cat(image_embeds, dim=0).to(embedded.dtype)
        if image_embeds.shape[0] != len(visual_destinations):
            sys.exit(
                f"tower produced {image_embeds.shape[0]} rows for "
                f"{len(visual_destinations)} destinations"
            )
        embedded = embedded.clone()
        embedded[0, torch.tensor(visual_destinations, dtype=torch.long)] = image_embeds
        visual_pos_masks = torch.zeros(1, length, dtype=torch.bool)
        visual_pos_masks[0, torch.tensor(visual_destinations, dtype=torch.long)] = True
        # The visual model returns deepstack_feature_lists, a list of one
        # [total_merged_rows, out_hidden] tensor per DeepStack level; the text
        # model consumes exactly that list (the image-only branch of
        # Qwen3VLModel.forward).
        deepstack_visual_embeds = list(deepstack_image_embeds)
    # [3, 1, L]: the text model's mrope position-id layout.
    positions_t = torch.tensor(positions, dtype=torch.long).unsqueeze(1)
    out = model.language_model(
        input_ids=None,
        position_ids=positions_t,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=embedded,
        cache_position=torch.arange(length, dtype=torch.long),
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    return ref.lm_head(out.last_hidden_state[:, -1])[0]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=script_dir / "goldens" / "checkpoints" / "qwen3vl-wp8")
    parser.add_argument("--out", type=Path,
                        default=script_dir / "goldens" / "qwen3vl-wp8.json")
    args = parser.parse_args()

    import torch
    from safetensors import torch as st
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    environment = gate_versions()
    checkpoint_dir: Path = args.checkpoint_dir
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    (checkpoint_dir / "images").mkdir(parents=True)

    # ------------------------------------------------------------- checkpoint
    checkpoint_dir.joinpath("config.json").write_text(
        json.dumps(CONFIG_JSON, indent=2) + "\n"
    )
    checkpoint_dir.joinpath("preprocessor_config.json").write_text(
        json.dumps(PREPROCESSOR_JSON, indent=2) + "\n"
    )
    shutil.copyfile(
        script_dir.parent.parent
        / "jmlx-models"
        / "src"
        / "test"
        / "resources"
        / "qwen3vl-tokens"
        / "tokenizer.json",
        checkpoint_dir / "tokenizer.json",
    )

    # The single source of truth for the Python model is the WRITTEN config
    # file: from_pretrained parses it, and the field assertions below verify
    # the hand-written values reached the transformers config objects intact.
    cfg = Qwen3VLConfig.from_pretrained(checkpoint_dir)
    text = cfg.text_config
    vision = cfg.vision_config
    if (
        text.hidden_size != 256
        or text.num_hidden_layers != 3
        or text.num_attention_heads != 4
        or text.num_key_value_heads != 2
        or text.head_dim != 64
        or text.intermediate_size != 512
        or text.vocab_size != 21
        or text.max_position_embeddings != 128
        or text.rope_theta != 5000000
        or text.rope_scaling["mrope_section"] != [16, 8, 8]
        or not text.rope_scaling["mrope_interleaved"]
    ):
        sys.exit(f"text config round-trip mismatch: {text.to_dict()}")
    if (
        vision.depth != 4
        or vision.hidden_size != 256
        or vision.num_heads != 4
        or vision.intermediate_size != 512
        or vision.patch_size != 16
        or vision.temporal_patch_size != 2
        or vision.spatial_merge_size != 2
        or vision.num_position_embeddings != 64
        or vision.out_hidden_size != 256
        or vision.deepstack_visual_indexes != [1, 3]
    ):
        sys.exit(f"vision config round-trip mismatch: {vision.to_dict()}")
    if (
        cfg.image_token_id != IMAGE_TOKEN
        or cfg.video_token_id != VIDEO_TOKEN
        or cfg.vision_start_token_id != START_TOKEN
        or cfg.vision_end_token_id != END_TOKEN
        or not cfg.tie_word_embeddings
    ):
        sys.exit(f"outer config round-trip mismatch: {cfg.to_dict()}")

    torch.manual_seed(SEED)
    model = Qwen3VLForConditionalGeneration(cfg)
    # The model init leaves every norm at its PyTorch default (weight 1,
    # bias 0): with all 24 norms identical the strict golden could not catch
    # a norm swap or a norm-key typo. Perturb before saving.
    counts = perturb_norms(model, SEED)
    if counts != {"layer_norm": 11, "text_rms_norm": 13}:
        sys.exit(f"expected 11 LayerNorm + 13 text RMSNorm affines, found {counts}")
    # Vision: 2 per block x 4 blocks + merger + 2 deepstack mergers = 11;
    # text: 4 per layer x 3 layers (input, post, q, k norms) + final norm = 13.

    # 4.57.6's tie_weights() does not tie the lm_head (the alias is applied
    # by _tie_or_clone_weights during from_pretrained only), so the
    # generator applies the tie explicitly against the config's declared
    # tie flag.
    model._tie_or_clone_weights(model.lm_head, model.model.language_model.embed_tokens)
    if model.lm_head.weight is not model.model.language_model.embed_tokens.weight:
        sys.exit("tie did not alias lm_head to the token embeddings")

    state = model.state_dict()
    expected = expected_state_keys(text.num_hidden_layers, vision.depth,
                                   vision.deepstack_visual_indexes)
    if set(state.keys()) != expected | {"lm_head.weight"}:
        sys.exit(
            f"state dict key set drift: missing {expected - set(state.keys())}; "
            f"extra {set(state.keys())} - {expected} - {{'lm_head.weight'}}"
        )
    # state_dict() clones its tensors (keep_vars=False), so the tied head is
    # verified by value, not object identity.
    if not torch.equal(
        state["lm_head.weight"], state["model.language_model.embed_tokens.weight"]
    ):
        sys.exit("tied lm_head differs from the stored embedding")
    bf16_state = {k: v.to(torch.bfloat16) for k, v in state.items() if k != "lm_head.weight"}
    st.save_file(bf16_state, checkpoint_dir / "model.safetensors")

    image_paths = {}
    for name, width, height, coeffs in IMAGE_SPECS:
        path = checkpoint_dir / "images" / f"{name}.png"
        build_image(width, height, coeffs, path)
        image_paths[name] = path

    # ------------------------------------------------------- reference model
    # The reference run loads the SAVED checkpoint back (not the in-memory
    # model) and upconverts the stored bf16 weights to float32, so the
    # golden exercises the exact stored-bf16 -> float32 path the Java
    # composite model uses (see the WP6 generator's reference_dtype_note).
    saved = st.load_file(checkpoint_dir / "model.safetensors")
    ref_cfg = Qwen3VLConfig.from_pretrained(checkpoint_dir)
    ref_cfg._attn_implementation = "eager"
    ref_cfg.text_config.dtype = torch.float32
    ref = Qwen3VLForConditionalGeneration(ref_cfg)
    ref._tie_or_clone_weights(ref.lm_head, ref.model.language_model.embed_tokens)
    ref.load_state_dict({k: v.float() for k, v in saved.items()}, strict=False)
    ref.eval()

    # ------------------------------------------------------------ processor
    proc = slow_processor()
    batch_a = proc.preprocess([str(image_paths["a"])], return_tensors="pt")
    batch_b = proc.preprocess([str(image_paths["b"])], return_tensors="pt")
    batch_ab = proc.preprocess(
        [str(image_paths["a"]), str(image_paths["b"])], return_tensors="pt"
    )
    pixels_a = batch_a["pixel_values"].numpy()
    pixels_b = batch_b["pixel_values"].numpy()
    pixels_ab = batch_ab["pixel_values"].numpy()
    grids_a = [[int(v) for v in row] for row in batch_a["image_grid_thw"].numpy()]
    grids_b = [[int(v) for v in row] for row in batch_b["image_grid_thw"].numpy()]
    grids_ab = [[int(v) for v in row] for row in batch_ab["image_grid_thw"].numpy()]
    if pixels_a.shape != (96, 1536) or pixels_a.dtype != "float32":
        sys.exit(f"a pixel_values {pixels_a.shape} {pixels_a.dtype} != (96, 1536) float32")
    if pixels_b.shape != (96, 1536) or pixels_b.dtype != "float32":
        sys.exit(f"b pixel_values {pixels_b.shape} {pixels_b.dtype} != (96, 1536) float32")
    if grids_a != [EXPECTED_GRIDS["a"]] or grids_b != [EXPECTED_GRIDS["b"]] \
            or grids_ab != [EXPECTED_GRIDS["a"], EXPECTED_GRIDS["b"]]:
        sys.exit(f"grid_thw {grids_ab} != expected")
    # The packed two-image rows must be exactly the per-image rows in prompt
    # order (the Java processor emits one row block per image in the same
    # order).
    import numpy as np

    if not (np.array_equal(pixels_ab[:96], pixels_a) and np.array_equal(pixels_ab[96:], pixels_b)):
        sys.exit("two-image pixel_values != concat(one-image a, one-image b)")

    one_expanded = expand_prompt(ONE_IMAGE_PROMPT, [EXPECTED_GRIDS["a"]])
    two_expanded = expand_prompt(TWO_IMAGE_PROMPT, [EXPECTED_GRIDS["a"], EXPECTED_GRIDS["b"]])
    itit_expanded = expand_prompt(
        IMAGE_TEXT_IMAGE_TEXT_PROMPT, [EXPECTED_GRIDS["a"], EXPECTED_GRIDS["b"]]
    )
    # Merged rows: 96 raw rows / merge^2 = 24 per image.
    if len(one_expanded) != 8 - 1 + 24:
        sys.exit(f"one-image expansion {len(one_expanded)} != 31")
    if len(two_expanded) != 9 - 2 + 48:
        sys.exit(f"two-image expansion {len(two_expanded)} != 55")
    if len(itit_expanded) != 11 - 2 + 48:
        sys.exit(f"image-text-image-text expansion {len(itit_expanded)} != 57")

    # ------------------------------------------------- case bookkeeping
    # The four base cases carry the full record (pixels, features,
    # positions, prefill logits, greedy decode steps); the two forced
    # cases reference their base case's pixels/features and add the
    # shared forced continuation's explicit positions and per-step
    # fresh-full logits.
    base_cases = [
        ("one-image", ONE_IMAGE_PROMPT, one_expanded, ["a.png"], [EXPECTED_GRIDS["a"]],
         torch.from_numpy(pixels_a), 4),
        ("two-image", TWO_IMAGE_PROMPT, two_expanded, ["a.png", "b.png"],
         [EXPECTED_GRIDS["a"], EXPECTED_GRIDS["b"]], torch.from_numpy(pixels_ab), 3),
        ("image-text-image-text", IMAGE_TEXT_IMAGE_TEXT_PROMPT, itit_expanded,
         ["a.png", "b.png"], [EXPECTED_GRIDS["a"], EXPECTED_GRIDS["b"]],
         torch.from_numpy(pixels_ab), 3),
        ("text-only", TEXT_PROMPT, list(TEXT_PROMPT), [], [], None, 3),
    ]
    forced_cases = [
        ("forced image-bearing", "one-image", FORCED_HISTORY),
        ("forced text-only", "text-only", FORCED_HISTORY),
    ]

    cases: list[dict] = []
    base_by_name = {}
    with torch.no_grad():
        for name, unexpanded, expanded, images, grids, pixels, steps in base_cases:
            destinations = image_token_positions(expanded)
            if len(destinations) != sum(
                t * h * w // (MERGE * MERGE) for (t, h, w) in grids
            ):
                sys.exit(f"{name}: destination count does not match the merged rows")
            positions, delta = ref.model.get_rope_index(
                torch.tensor([expanded]),
                image_grid_thw=torch.tensor(grids) if grids else None,
                attention_mask=None,
            )
            positions = [axis[0].tolist() for axis in positions]
            # Standard processor prefill (the model's own get_rope_index
            # path), for the prefill logits.
            standard = ref(
                input_ids=torch.tensor([expanded]),
                pixel_values=pixels if pixels is not None else None,
                image_grid_thw=torch.tensor(grids) if grids else None,
            )
            prefill_logits = standard.logits[0, -1]
            # Pin the explicit path's faithfulness at the zero-history
            # point: it must equal the standard forward's last-position
            # logits far below the 1e-4 acceptance gate.
            explicit_prefill = explicit_full_forward(
                ref, expanded, pixels, torch.tensor(grids) if grids else None,
                destinations, positions,
            )
            if float((explicit_prefill - prefill_logits).abs().max()) > 1e-6:
                sys.exit(f"{name}: explicit path diverges from the standard prefill")
            # Tower features for the golden's intermediate record.
            visual_features = None
            if pixels is not None:
                image_embeds, deepstack_image_embeds = ref.model.get_image_features(
                    pixels, torch.tensor(grids)
                )
                visual_features = {
                    "merged": rounded(torch.cat(image_embeds, dim=0).tolist()),
                    "deepstack": [rounded(t.tolist()) for t in deepstack_image_embeds],
                }
            # Pin the identity the Java side derives from the recorded axes
            # (Qwen3VlTestSupport.fullLogits): the continuation origin of the
            # original prompt must equal the model's own rope delta.
            if max(max(axis) for axis in positions) + 1 - len(expanded) != int(delta):
                sys.exit(f"{name}: rope delta identity failed")
            # Greedy decode steps: at every step the reference is the fresh,
            # full, uncached forward over the original images plus the
            # generated history, with the destination-restricted visual
            # injection and the explicit continuation positions.
            decode_steps = []
            history: list[int] = []
            for _ in range(steps):
                full_ids = expanded + history
                length = len(full_ids)
                full_positions = [
                    [axis[i] if i < len(expanded) else i + int(delta) for i in range(length)]
                    for axis in positions
                ]
                logits = explicit_full_forward(
                    ref, full_ids, pixels,
                    torch.tensor(grids) if grids else None,
                    destinations, full_positions,
                )
                token = int(logits.argmax())
                history.append(token)
                decode_steps.append({"token_id": token, "logits": rounded(logits.tolist())})
            case = {
                "name": name,
                "type": "generation",
                "unexpanded_ids": unexpanded,
                "expanded_ids": expanded,
                "images": images,
                "grids": grids,
                "visual_features": visual_features,
                "positions": {
                    "t": positions[0],
                    "h": positions[1],
                    "w": positions[2],
                    "delta": int(delta),
                    "continuation": "logicalPosition + delta on all three axes",
                },
                "prefill_logits": rounded(prefill_logits.tolist()),
                "decode_steps": decode_steps,
            }
            if pixels is not None:
                case["pixel_values_sha256"] = sha256_bytes(pixels.numpy().tobytes())
            cases.append(case)
            base_by_name[name] = {"case": case, "pixels": pixels, "grids": grids,
                                  "expanded": expanded, "destinations": destinations,
                                  "positions": positions, "delta": int(delta)}
        for name, base_name, history in forced_cases:
            base = base_by_name[base_name]
            expanded = base["expanded"]
            positions = base["positions"]
            delta = base["delta"]
            full_ids = expanded + history
            length = len(full_ids)
            full_positions = [
                [axis[i] if i < len(expanded) else i + delta for i in range(length)]
                for axis in positions
            ]
            decode_steps = []
            partial: list[int] = []
            for step in range(len(history)):
                step_ids = expanded + partial
                step_positions = [
                    [axis[i] if i < len(expanded) else i + delta for i in range(len(step_ids))]
                    for axis in positions
                ]
                logits = explicit_full_forward(
                    ref, step_ids, base["pixels"],
                    torch.tensor(base["grids"]) if base["grids"] else None,
                    base["destinations"], step_positions,
                )
                decode_steps.append({
                    "token_id": history[step],
                    "logits": rounded(logits.tolist()),
                })
                partial.append(history[step])
            cases.append({
                "name": name,
                "type": "forced",
                "base_case": base_name,
                "unexpanded_ids": base["case"]["unexpanded_ids"],
                "expanded_ids": expanded,
                "images": base["case"]["images"],
                "grids": base["case"]["grids"],
                "forced_history": history,
                "positions": {
                    "t": full_positions[0],
                    "h": full_positions[1],
                    "w": full_positions[2],
                    "delta": delta,
                    "continuation": "logicalPosition + delta on all three axes",
                },
                "decode_steps": decode_steps,
            })

    # ------------------------------------------------------------- golden
    result = {
        "schema": "jmlx-phase7-3b-wp8-qwen3vl-model-v1",
        "generated_by": "tools/hf-reference/qwen3vl_wp8.py",
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
                "draw_order": "model.modules() order, weight then bias per norm",
                "layer_norm_weight_distribution": "normal(mean=1.0, std=0.1)",
                "layer_norm_bias_distribution": "normal(mean=0.0, std=0.05)",
                "text_rms_norm_weight_distribution": "normal(mean=1.0, std=0.1)",
                "vision_layer_norms": counts["layer_norm"],
                "text_rms_norms": counts["text_rms_norm"],
                "count": counts["layer_norm"] + counts["text_rms_norm"],
                "note": (
                    "every norm affine is perturbed before the checkpoint is "
                    "saved so the 24 norms are mutually distinct and the strict "
                    "golden discriminates the composite model's norm wiring"
                ),
            },
            "config": py(cfg.to_dict()),
            "checkpoint_files": {
                "config.json": sha256_file(checkpoint_dir / "config.json"),
                "model.safetensors": sha256_file(checkpoint_dir / "model.safetensors"),
                "preprocessor_config.json": sha256_file(checkpoint_dir / "preprocessor_config.json"),
                "tokenizer.json": sha256_file(checkpoint_dir / "tokenizer.json"),
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
            "expected_grids": EXPECTED_GRIDS,
            "key_layout_note": (
                "the full Qwen3VLForConditionalGeneration state dict already carries the "
                "real artifact key scheme (model.visual.*, model.language_model.*); the "
                "tied lm_head.weight is dropped, so all 104 tensors are stored as "
                "bfloat16 (35 text + 69 vision)"
            ),
            "tied_head_note": (
                "tie_word_embeddings is true and the checkpoint stores no lm_head tensor; "
                "the reference model applies the config-declared tie (4.57.6's "
                "from_pretrained-only _tie_or_clone_weights, applied explicitly) before "
                "loading the 104 stored tensors"
            ),
            "reference_dtype_note": (
                "the checkpoint config declares text dtype bfloat16 (matching the "
                "stored bf16 tensors), but the golden is a genuine CPU float32 "
                "reference: the reference model is built with text_config.dtype "
                "overridden to float32 and loads the stored bf16 weights "
                "upconverted to float32 -- the exact path the Java composite "
                "model takes when it upcasts the checkpoint"
            ),
            "attention_implementation": "eager",
            "decode_reference_note": (
                "every decode and forced step's logits are a fresh full uncached "
                "forward over the original expanded prompt plus the generated "
                "history, computed with the explicit reference path: the tower "
                "runs once from the original pixel values, image rows and "
                "DeepStack levels are applied only at the original prompt's "
                "image-token destinations, and the positions are the recorded "
                "prefill axes followed by logicalPosition + delta. Token-based "
                "placeholder detection and get_rope_index are never re-run over "
                "generated markers; the explicit path is pinned against the "
                "standard processor forward at the zero-history point (1e-6)"
            ),
            "deepstack_note": (
                "two DeepStack levels (vision extraction indices [1, 3]) are "
                "injected after decoder blocks 0 and 1; the third decoder layer "
                "is un-injected, so every injection position kind is exercised"
            ),
            "greedy_id_note": (
                "in every base generation case the reference's greedy argmax is "
                "token 7 at each recorded step (top-two margins 0.4851 to "
                "1.1170), so the strict suite's exact-greedy-ID assertion is "
                "degenerate on this 21-token tied checkpoint and a model stuck "
                "on token 7 would pass it; the binding gate is the full "
                "logit-vector comparison against this golden at 1e-4. The "
                "forced steps' token_id fields are the forced-history labels, "
                "not argmaxes -- the strict suite asserts greedy only on the "
                "generation cases"
            ),
            "position_note": (
                "num_position_embeddings is 64 (an 8 x 8 learned table) while "
                "grids reach 12 along one axis, so the bilinear "
                "fast_pos_embed_interpolate runs beyond the table on both "
                "images; mrope_section [16, 8, 8] is a strictly unequal split "
                "with 2 * sum = 64 = head_dim"
            ),
            "processor": {
                "class": "Qwen2VLImageProcessor",
                "module": "transformers.models.qwen2_vl.image_processing_qwen2_vl",
                "note": (
                    "transformers 4.57.6 has no Qwen3VLImageProcessor class; the "
                    "slow PIL Qwen2VLImageProcessor is the pinned preprocessing "
                    "path. min_pixels/max_pixels are passed explicitly because "
                    "the 4.57.6 __init__ overwrites any size dict with the "
                    "3136/1003520 defaults (findings Section 4.2). Images are "
                    "passed as file path strings so the load_image "
                    "(exif_transpose) path matches the Java processFile path; "
                    "all areas lie inside [3136, 12845056] so smart_resize is a "
                    "no-op."
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
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Compact single-line JSON, matching the capture-heavy golden convention.
    args.out.write_text(json.dumps(py(result)) + "\n")

    # -------------------------------------------------------------- summary
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    print(f"config.json: {sha256_file(checkpoint_dir / 'config.json')}")
    print(f"model.safetensors: {sha256_file(checkpoint_dir / 'model.safetensors')} "
          f"({len(bf16_state)} tensors, bf16)")
    print(f"preprocessor_config.json: {sha256_file(checkpoint_dir / 'preprocessor_config.json')}")
    print(f"tokenizer.json: {sha256_file(checkpoint_dir / 'tokenizer.json')}")
    for name, _width, _height, _coeffs in IMAGE_SPECS:
        path = image_paths[name]
        print(f"{path.name}: {sha256_file(path)} ({path.stat().st_size} bytes)")
    print(f"grids: {grids_ab}")
    print(f"expansions: one {len(one_expanded)}, two {len(two_expanded)}, "
          f"itit {len(itit_expanded)}")
    print(f"norm counts: {counts}")
    for case in cases:
        delta = case["positions"]["delta"]
        steps = len(case["decode_steps"])
        tokens = [s["token_id"] for s in case["decode_steps"]]
        print(f"{case['name']}: delta {delta}, {steps} steps, tokens {tokens}")


if __name__ == "__main__":
    main()
