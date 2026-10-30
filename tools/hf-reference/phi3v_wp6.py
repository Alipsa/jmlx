#!/usr/bin/env python3
"""Phase 7.3c WP6 tiny Phi-3.5-Vision full-model checkpoint + acceptance golden generator.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0+cpu,
torchvision 0.21.0, tokenizers 0.22.2, safetensors, Pillow 12.3.0, NumPy
2.5.3, Python 3.12), CPU float32, single-threaded, deterministic (seed
20261025). It writes:

  * a tiny synthetic full Phi-3.5-Vision checkpoint at
    tools/hf-reference/goldens/checkpoints/phi3v-wp6/
    (the hand-written strict config.json with pad_token_id 0, a matching
    7-key preprocessor_config.json, the committed 21-token phi3v-tiny
    fixture tokenizer, model.safetensors with 76 bf16 tensors keyed with
    the real artifact's `model.*` / `model.vision_embed_tokens.*` layout
    (untied lm_head included; the embedding module the bundled model
    registers twice -- model.embed_tokens and
    model.vision_embed_tokens.wte -- stored once under
    model.embed_tokens.weight) and the three WP2 fixture PNGs under
    images/);
  * the golden tools/hf-reference/goldens/phi3v-wp6.json capturing the
    plan WP6 acceptance cases:

    - "text-only" ("toast " x 20, 120 tokens), 10 steps;
    - "wide-cross-boundary" ("c d <|image_1|> e s", wide 457 rows,
      465 tokens) -- the single case whose first recompute fires: the
      step that first reaches total width 471 re-runs the long table
      (golden recompute_step 6);
    - "square" ("toast <|image_1|>toast", square 757 rows, 768 tokens),
      6 steps -- long table from the prefill;
    - "wide-long-prompt" ("toast <|image_1|>toast code", wide 457 rows,
      473 tokens), 6 steps -- the image rows alone are 457 < 470, so only
      the finished prompt length pins the long-table selection;
    - "tall" ("toast <|image_1|>toast ", tall 469 rows, 481 tokens),
      6 steps;
    - "two-image" (wide then square, 1231 tokens), 4 steps.

    For every case the golden records the rendered prompt, the expanded
    IDs (compact run form), the images, per-image image_sizes and
    num_img_tokens, the per-image processed pixel bytes' SHA-256, the
    projector's output rows (image cases), the recompute step (if any),
    and -- for every step -- the greedy token ID and a fresh full
    uncached forward's last-position logits (entry 0 is the prefill
    point, over the expanded prompt alone).

Every decode step is computed with a fresh, full, uncached forward
(`full_forward` below) over the expanded prompt plus the generated
history, with the sentinel IDs intact: the bundled
Phi3ImageEmbedding.forward substitutes the image rows natively (the
reference's own path, never the boundary-recompute of
prepare_inputs_for_generation -- that is recorded for the design only).
Logits are lm_head(hidden).float() at the last position.

The bundled trust-remote-code sources are the WP0-pinned artifact files
under tools/image-oracle/fixtures/phi3v/ (SHA-256 verified here before
import). The bundled CLIP_VIT_LARGE_PATCH14_336_CONFIG constant (24
layers, intermediate 4096) is recorded pristine in the provenance and
then swapped in the module for the tiny 3-layer / intermediate-128
config before any model build -- the CLIP width 1024 cannot shrink
(reshape_hd_patches_2x2merge asserts L == 24*24 and C == 1024). The
reference run loads the SAVED checkpoint back (not the in-memory model)
and upconverts the stored bf16 weights to float32, so the golden
exercises the exact stored-bf16 -> float32 path the Java model uses.

Every norm affine is perturbed away from its PyTorch default before the
checkpoint is saved -- every nn.LayerNorm (CLIP pre_layrnorm + 2 x 3
blocks + post_layernorm): weight ~ N(1.0, 0.1), bias ~ N(0, 0.05);
every bundled Phi3RMSNorm (2 x 2 decoder + final): weight ~ N(1.0,
0.1) -- drawn from a dedicated seed-20261025 generator in
model.modules() order (weight then bias per norm), so the 13 norms are
mutually distinct and the strict golden discriminates the composite
model's norm-to-key wiring. The zero-initialized glb_GN / sub_GN
learnable separators get N(0, 1) draws from the same generator (glb
first, sub second), so swapping the two 4096-wide tensors is
discriminable too.

Prompt expansion mirrors Phi3VPromptExpander.expandRendered exactly:
split the rendered prompt on the marker pattern, encode every chunk
separately with the checkpoint's own tokenizer (the tiny fixture
declares no added tokens and a post-processor with no special tokens,
so add_special_tokens=True adds nothing -- no per-encoding <s>), and
expand marker i to num_img_tokens[i-1] sentinel rows of -i.

It is never invoked by Gradle. Every written file's SHA-256 is printed
at the end and pinned in tools/hf-reference/provenance.json (keys
relative to goldens/); the generator's own SHA-256 is pinned under
`phi3v_wp6_generator_sha256`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
from pathlib import Path

SEED = 20261025

# The tiny checkpoint geometry (the WP6 plan's frozen shape).
TEXT_LAYERS = 2
TEXT_HIDDEN = 128
TEXT_INTER = 256
TEXT_HEADS = 2
TEXT_KV_HEADS = 1
CLIP_LAYERS = 3
CLIP_HIDDEN = 1024
CLIP_INTER = 128
CLIP_HEADS = 16
ORIGINAL_MAX_POSITIONS = 470
MAX_POSITIONS = 2048
FACTORS = 32
VOCAB = 21

# The committed bundled sources (WP0 pins) under the image-oracle
# fixtures, verified before import.
FIXTURE_SHAS = {
    "modeling_phi3_v.py": "b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c",
    "configuration_phi3_v.py": "18ff44c5c1f9a6523184fb2e580e03ebf722a65f37b9cc57cd1a6d620a0f2666",
    "processing_phi3_v.py": "990ef3d1f6b5cc61b44cf8ec8e8886b19721b7e01e8bae3ebea32781b0fcf598",
    "preprocessor_config.json": "0f3d920fcec587bb425312d30b0660e1fabaf46a75961f2b75ddbb1ff92de01c",
}

# The bundled Phi3VModel registers the single embed_tokens module twice
# (as model.embed_tokens and, via Phi3ImageEmbedding(config,
# wte=self.embed_tokens), as model.vision_embed_tokens.wte): the state
# dict therefore carries 77 keys for 76 unique tensors. The checkpoint
# stores the shared embedding once, under model.embed_tokens.weight.
WTE_KEY = "model.vision_embed_tokens.wte.weight"

# The WP2 fixture images (copied into the checkpoint; source SHAs from
# tools/image-oracle/fixtures/phi3v/patches/manifest.json).
IMAGE_SPECS = [
    # (key, filename, width, height, source sha256, processed-bin sha256)
    ("square", "square-336x336.png", 336, 336,
     "40734e8e0fbc8fe651c65d4b789082dc8d2ca833f59b9473bb17d54ecabfc2d8",
     "0292f014488f681947391a5daf5d805c6900c2098d16e171b9228c199ee7f7e7"),
    ("wide", "wide-672x336.png", 672, 336,
     "702955febf85bce110b16120cd3a71dfd568cabbce04f29b78561a1260df7efc",
     "6cdac6a70c10c927b3345a5cd3fe683f96d8acd0a0e368854ca9ff257cc71732"),
    ("tall", "tall-336x672.png", 336, 672,
     "30703c484d9681bdbf7a6f1cfb147eadd87d73afa901fc2134257e40900575a5",
     "3869d7af80146c160ce639a95a4ad34f831feee3177acd5e677c12848febe396"),
]

# The 21-token phi3v-tiny fixture tokenizer (no added tokens, no
# post-processor special tokens: add_special_tokens=True adds nothing).
TINY_TOKENIZER_SHA = "cfb9e05ab721163bd6de2f0dcc552dae7c410fd23b0e5ceddb5791b01a9da99b"

IMAGE_FILENAME = {key: filename for key, filename, *_ in IMAGE_SPECS}

# Expected processor outputs (findings section 6, the WP2 oracle):
# padded [height, width] and the reference's per-image token formula
# (h//336 * w//336 + 1) * 144 + 1 + (h//336 + 1) * 12.
EXPECTED_SIZES = {"square": [672, 672], "wide": [336, 672], "tall": [672, 336]}
EXPECTED_TOKENS = {"square": 757, "wide": 457, "tall": 469}

# The WP6 acceptance cases: (name, rendered prompt, image keys in prompt
# order, max_new_tokens).
def marker(index: int) -> str:
    return f"<|image_{index}|>"

CASES = [
    ("text-only", "toast " * 20, [], 10),
    ("wide-cross-boundary", "c d " + marker(1) + " e s", ["wide"], 8),
    ("square", "toast " + marker(1) + "toast", ["square"], 6),
    ("wide-long-prompt", "toast " + marker(1) + "toast code", ["wide"], 6),
    ("tall", "toast " + marker(1) + "toast ", ["tall"], 6),
    ("two-image", "toast " + marker(1) + "toast " + marker(2) + "toast",
     ["wide", "square"], 4),
]

# Expected expanded-prompt lengths (the plan WP6 table).
EXPECTED_LENGTHS = [120, 465, 768, 473, 481, 1231]

# The hand-written strict config: the tiny-checkpoint template (the
# 21-token vocabulary, the two-layer GQA text side, NTK-Su 470/2048 with
# 32 short/long factors) plus pad_token_id 0 -- the bundled
# Phi3VModel.__init__ builds nn.Embedding(vocab, hidden,
# padding_idx=config.pad_token_id) and the bundled config default
# 32000 is out of range for a 21-token vocabulary. No sliding_window
# key (bundled default None -> plain causal 4d mask), no torch_dtype
# key.
CONFIG_JSON = {
    "model_type": "phi3_v",
    "architectures": ["Phi3VForCausalLM"],
    "vocab_size": VOCAB,
    "hidden_size": TEXT_HIDDEN,
    "intermediate_size": TEXT_INTER,
    "num_hidden_layers": TEXT_LAYERS,
    "num_attention_heads": TEXT_HEADS,
    "num_key_value_heads": TEXT_KV_HEADS,
    "rms_norm_eps": 0.000001,
    "rope_theta": 10000,
    "rope_scaling": {
        "type": "su",
        "short_factor": [1.0] * FACTORS,
        "long_factor": [2.0] * FACTORS,
    },
    "original_max_position_embeddings": ORIGINAL_MAX_POSITIONS,
    "max_position_embeddings": MAX_POSITIONS,
    "hidden_act": "silu",
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "img_processor": {
        "name": "clip_vision_model",
        "model_name": "openai/clip-vit-large-patch14-336",
        "image_dim_out": CLIP_HIDDEN,
        "num_img_tokens": 144,
    },
    "embd_layer": {
        "projection_cls": "mlp",
        "use_hd_transform": True,
        "hd_transform_order": "sub_glb",
        "with_learnable_separator": True,
        "embedding_cls": "image",
    },
}

# The 7-key preprocessor config (the tiny-checkpoint template's shape;
# the bundled Phi3VImageProcessor.from_pretrained reads it).
PREPROCESSOR_JSON = {
    "do_convert_rgb": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_processor_type": "Phi3VImageProcessor",
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "num_crops": 4,
    "num_img_tokens": 144,
    "processor_class": "Phi3VProcessor",
}

MARKER_RE = re.compile(r"<\|image_\d+\|>")


def gate_versions() -> dict:
    import PIL
    import numpy as np
    import torch
    import tokenizers
    import torchvision
    import transformers
    from safetensors import __version__ as safetensors_version

    if (
        transformers.__version__ != "4.57.6"
        or PIL.__version__ != "12.3.0"
        or np.__version__ != "2.5.3"
        or torch.__version__ not in ("2.6.0", "2.6.0+cpu")
        or torchvision.__version__ != "0.21.0"
        or tokenizers.__version__ != "0.22.2"
        or platform.python_version_tuple()[:2] != ("3", "12")
    ):
        sys.exit(
            "phi3v_wp6 requires transformers 4.57.6, Pillow 12.3.0, NumPy 2.5.3, "
            f"torch 2.6.0, torchvision 0.21.0, tokenizers 0.22.2 and CPython 3.12; "
            f"got transformers {transformers.__version__}, Pillow {PIL.__version__}, "
            f"NumPy {np.__version__}, torch {torch.__version__}, torchvision "
            f"{torchvision.__version__}, tokenizers {tokenizers.__version__}, "
            f"Python {platform.python_version()}"
        )
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "tokenizers": tokenizers.__version__,
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


def compact_ids(ids: list[int]) -> list:
    """The golden's compact run form: plain ints, except maximal runs of
    more than one identical negative ID, which become
    {"neg_run": id, "count": n}."""
    out: list = []
    i = 0
    while i < len(ids):
        if ids[i] < 0:
            j = i
            while j < len(ids) and ids[j] == ids[i]:
                j += 1
            if j - i > 1:
                out.append({"neg_run": ids[i], "count": j - i})
            else:
                out.append(ids[i])
            i = j
        else:
            out.append(ids[i])
            i += 1
    return out


def expected_state_keys() -> set[str]:
    """The checkpoint's exact 76-key set (text 15 + vision 61), derived
    from the geometry and matching the bundled state dict layout."""
    keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for i in range(TEXT_LAYERS):
        layer = f"model.layers.{i}."
        for name in (
            "input_layernorm.weight",
            "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_up_proj.weight",
            "mlp.down_proj.weight",
        ):
            keys.add(layer + name)
    clip = "model.vision_embed_tokens.img_processor.vision_model."
    keys.update(
        {
            "model.vision_embed_tokens.glb_GN",
            "model.vision_embed_tokens.sub_GN",
            clip + "embeddings.class_embedding",
            clip + "embeddings.patch_embedding.weight",
            clip + "embeddings.position_embedding.weight",
            clip + "pre_layrnorm.weight",
            clip + "pre_layrnorm.bias",
        }
    )
    for i in range(CLIP_LAYERS):
        block = clip + f"encoder.layers.{i}."
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "layer_norm1.weight",
            "layer_norm1.bias",
            "layer_norm2.weight",
            "layer_norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ):
            keys.add(block + name)
    keys.update(
        {
            clip + "post_layernorm.weight",
            clip + "post_layernorm.bias",
            "model.vision_embed_tokens.img_projection.0.weight",
            "model.vision_embed_tokens.img_projection.0.bias",
            "model.vision_embed_tokens.img_projection.2.weight",
            "model.vision_embed_tokens.img_projection.2.bias",
        }
    )
    return keys


def perturb_norms(model, rms_cls, seed: int) -> dict:
    """Perturb every norm affine and the two learnable separators with a
    dedicated seeded generator. Order: model.modules() order, weight
    then bias per norm; after the last norm, glb_GN then sub_GN get
    N(0, 1) draws from the same generator. Returns the per-kind counts."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    counts = {"layer_norm": 0, "phi3_rms_norm": 0}
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.1, generator=gen)
            torch.nn.init.normal_(module.bias, mean=0.0, std=0.05, generator=gen)
            counts["layer_norm"] += 1
        elif isinstance(module, rms_cls):
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.1, generator=gen)
            counts["phi3_rms_norm"] += 1
    emb = model.model.vision_embed_tokens
    torch.nn.init.normal_(emb.glb_GN, mean=0.0, std=1.0, generator=gen)
    torch.nn.init.normal_(emb.sub_GN, mean=0.0, std=1.0, generator=gen)
    return counts


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo = script_dir.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=script_dir / "goldens" / "checkpoints" / "phi3v-wp6")
    parser.add_argument("--out", type=Path,
                        default=script_dir / "goldens" / "phi3v-wp6.json")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from safetensors import torch as st
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    environment = gate_versions()
    fixture_dir = repo / "tools" / "image-oracle" / "fixtures" / "phi3v"
    for name, expected in FIXTURE_SHAS.items():
        actual = sha256_file(fixture_dir / name)
        if actual != expected:
            sys.exit(f"bundled fixture {name} drifted: {actual} != {expected}")

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
    tiny_tokenizer = repo / "jmlx-models" / "src" / "test" / "resources" \
                     / "phi3v-tiny-tokens" / "tokenizer.json"
    if sha256_file(tiny_tokenizer) != TINY_TOKENIZER_SHA:
        sys.exit(f"phi3v-tiny-tokens fixture drifted: {sha256_file(tiny_tokenizer)}")
    shutil.copyfile(tiny_tokenizer, checkpoint_dir / "tokenizer.json")

    image_paths: dict[str, Path] = {}
    for key, filename, _w, _h, source_sha, _bin_sha in IMAGE_SPECS:
        source = fixture_dir / filename
        if sha256_file(source) != source_sha:
            sys.exit(f"WP2 fixture image {filename} drifted: {sha256_file(source)}")
        destination = checkpoint_dir / "images" / filename
        shutil.copyfile(source, destination)
        if sha256_file(destination) != source_sha:
            sys.exit(f"copied image {filename} does not match its source")
        image_paths[key] = destination

    # The single source of truth for the Python model is the WRITTEN
    # config file: from_pretrained parses it, and the field assertions
    # below verify the hand-written values reached the config object
    # intact.
    config_cls = get_class_from_dynamic_module(
        "configuration_phi3_v.Phi3VConfig", str(fixture_dir)
    )
    cfg = config_cls.from_pretrained(str(checkpoint_dir))
    rope = cfg.rope_scaling
    if (
        cfg.vocab_size != VOCAB
        or cfg.hidden_size != TEXT_HIDDEN
        or cfg.intermediate_size != TEXT_INTER
        or cfg.num_hidden_layers != TEXT_LAYERS
        or cfg.num_attention_heads != TEXT_HEADS
        or cfg.num_key_value_heads != TEXT_KV_HEADS
        or cfg.rms_norm_eps != 0.000001
        or cfg.rope_theta != 10000
        or rope.get("type") != "su"
        or len(rope.get("short_factor", [])) != FACTORS
        or len(rope.get("long_factor", [])) != FACTORS
        or cfg.original_max_position_embeddings != ORIGINAL_MAX_POSITIONS
        or cfg.max_position_embeddings != MAX_POSITIONS
        or cfg.tie_word_embeddings
        or cfg.bos_token_id != 1
        or cfg.eos_token_id != 2
        or cfg.pad_token_id != 0
        or cfg.img_processor.get("name") != "clip_vision_model"
        or cfg.img_processor.get("model_name") != "openai/clip-vit-large-patch14-336"
        or cfg.img_processor.get("image_dim_out") != CLIP_HIDDEN
        or cfg.img_processor.get("num_img_tokens") != 144
        or set(cfg.embd_layer.keys()) != {
            "projection_cls", "use_hd_transform", "hd_transform_order",
            "with_learnable_separator", "embedding_cls"}
        or cfg.embd_layer.get("projection_cls") != "mlp"
        or cfg.embd_layer.get("hd_transform_order") != "sub_glb"
    ):
        sys.exit(f"config round-trip mismatch: {cfg.to_dict()}")

    model_cls = get_class_from_dynamic_module(
        "modeling_phi3_v.Phi3VForCausalLM", str(fixture_dir)
    )
    model_mod = sys.modules[model_cls.__module__]
    rms_cls = get_class_from_dynamic_module(
        "modeling_phi3_v.Phi3RMSNorm", str(fixture_dir)
    )

    # Record the pristine bundled CLIP constant, then swap it for the
    # tiny 3-layer / intermediate-128 config: Phi3ImageEmbedding reads
    # the module-level CLIP_VIT_LARGE_PATCH14_336_CONFIG at build time,
    # and the width 1024 cannot shrink (reshape_hd_patches_2x2merge
    # asserts L == 24*24 and C == 1024).
    from transformers.models.clip.configuration_clip import CLIPVisionConfig

    pristine_clip = model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG.to_dict()
    if (
        pristine_clip["hidden_size"] != 1024
        or pristine_clip["intermediate_size"] != 4096
        or pristine_clip["num_hidden_layers"] != 24
    ):
        sys.exit(f"pristine CLIP constant drifted: {pristine_clip}")
    model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
        image_size=336,
        patch_size=14,
        num_channels=3,
        hidden_size=CLIP_HIDDEN,
        intermediate_size=CLIP_INTER,
        num_attention_heads=CLIP_HEADS,
        num_hidden_layers=CLIP_LAYERS,
        projection_dim=768,
        layer_norm_eps=1e-5,
        attn_implementation="eager",
    )

    cfg._attn_implementation = "eager"
    torch.manual_seed(SEED)
    model = model_cls(cfg)
    if model.config._attn_implementation != "eager":
        sys.exit(f"model attention implementation is {model.config._attn_implementation}")
    if model.model.vision_embed_tokens.img_processor.config._attn_implementation != "eager":
        sys.exit("CLIP attention implementation is not eager")

    # The model init leaves every norm at its PyTorch default (weight 1,
    # bias 0): with all 13 norms identical the strict golden could not
    # catch a norm swap or a norm-key typo. Perturb before saving.
    # CLIP: pre_layrnorm + 2 x 3 blocks + post_layernorm = 8 LayerNorms;
    # text: 2 per layer x 2 layers + final norm = 5 bundled Phi3RMSNorms.
    counts = perturb_norms(model, rms_cls, SEED)
    if counts != {"layer_norm": 8, "phi3_rms_norm": 5}:
        sys.exit(f"expected 8 LayerNorm + 5 Phi3RMSNorm affines, found {counts}")

    state = model.state_dict()
    expected = expected_state_keys()
    actual = set(state.keys())
    if actual != expected | {WTE_KEY}:
        sys.exit(
            f"state dict key set drift: missing {expected - actual}; "
            f"extra {actual - expected - {WTE_KEY}}"
        )
    if state[WTE_KEY].data_ptr() != state["model.embed_tokens.weight"].data_ptr():
        sys.exit("vision_embed_tokens.wte is not the shared embed_tokens parameter")
    # The 76 stored tensors: the shared embedding is stored once, under
    # model.embed_tokens.weight (the TinyCheckpoints layout).
    bf16_state = {
        k: v.to(torch.bfloat16) for k, v in state.items() if k != WTE_KEY
    }
    if len(bf16_state) != 76:
        sys.exit(f"expected 76 stored tensors, found {len(bf16_state)}")
    st.save_file(bf16_state, checkpoint_dir / "model.safetensors")

    # ------------------------------------------------------- reference model
    # The reference run loads the SAVED checkpoint back (not the in-memory
    # model) and upconverts the stored bf16 weights to float32, so the
    # golden exercises the exact stored-bf16 -> float32 path the Java
    # model uses when it upcasts.
    saved = st.load_file(checkpoint_dir / "model.safetensors")
    ref_cfg = config_cls.from_pretrained(str(checkpoint_dir))
    ref_cfg._attn_implementation = "eager"
    ref = model_cls(ref_cfg)
    if ref.config._attn_implementation != "eager":
        sys.exit(f"reference attention implementation is {ref.config._attn_implementation}")
    # The saved checkpoint stores the shared embedding once; the built
    # module registers it under WTE_KEY as well, so the strict load
    # re-supplies the duplicate key from the same tensor.
    loaded = {k: v.float() for k, v in saved.items()}
    loaded[WTE_KEY] = loaded["model.embed_tokens.weight"]
    ref.load_state_dict(loaded, strict=True)
    ref.eval()
    # Spot-check that the load replaced the init weights: the stored
    # (up-cast bf16) values must round-trip exactly. Row 0 is the pad
    # row (zero by _init_weights), so probe a row-1 slice and require it
    # to be non-zero, so the check cannot pass trivially.
    probe_key = "model.embed_tokens.weight"
    stored = saved[probe_key].float().flatten()[TEXT_HIDDEN + 5 : TEXT_HIDDEN + 9]
    if not torch.equal(
        ref.model.embed_tokens.weight.flatten()[TEXT_HIDDEN + 5 : TEXT_HIDDEN + 9], stored
    ):
        sys.exit("reference state dict load did not replace the init weights")
    if int(torch.count_nonzero(stored)) == 0:
        sys.exit("probe slice of the stored embedding is all zeros; the "
                 "spot check is vacuous")

    # ------------------------------------------------------------ processor
    proc_cls = get_class_from_dynamic_module(
        "processing_phi3_v.Phi3VProcessor", str(fixture_dir)
    )
    proc_mod = sys.modules[proc_cls.__module__]
    image_processor = proc_mod.Phi3VImageProcessor.from_pretrained(str(checkpoint_dir))
    if int(image_processor.num_crops) != 4:
        sys.exit(f"processor options drifted: num_crops={image_processor.num_crops}")
    if (
        [float(x) for x in image_processor.image_mean] != PREPROCESSOR_JSON["image_mean"]
        or [float(x) for x in image_processor.image_std] != PREPROCESSOR_JSON["image_std"]
        or bool(image_processor.do_convert_rgb) is not True
    ):
        sys.exit(
            f"processor mean/std/convert_rgb drifted: {list(image_processor.image_mean)} "
            f"{list(image_processor.image_std)} {image_processor.do_convert_rgb}"
        )

    pil_images = {}
    for key, filename, width, height, _s, _b in IMAGE_SPECS:
        image = Image.open(image_paths[key])
        image.load()
        if image.mode != "RGB" or image.size != (width, height):
            sys.exit(f"fixture image {filename} is not RGB {width}x{height}")
        pil_images[key] = image

    per_image_pixels: dict[str, torch.Tensor] = {}
    processed: dict[str, dict] = {}
    for key, _filename, _width, _height, _source_sha, bin_sha in IMAGE_SPECS:
        out = image_processor(images=[pil_images[key]], return_tensors="pt")
        pv = out["pixel_values"]
        if tuple(pv.shape) != (1, 5, 3, 336, 336) or pv.dtype != torch.float32:
            sys.exit(f"{key} pixel_values {tuple(pv.shape)} {pv.dtype} unexpected")
        row = pv[0]
        if sha256_bytes(row.numpy().tobytes()) != bin_sha:
            sys.exit(
                f"{key} processed pixels drifted from the WP2 patch bin: "
                f"{sha256_bytes(row.numpy().tobytes())}"
            )
        sizes = [[int(x) for x in row_sizes] for row_sizes in out["image_sizes"]]
        tokens = [int(x) for x in out["num_img_tokens"]]
        if sizes != [EXPECTED_SIZES[key]] or tokens != [EXPECTED_TOKENS[key]]:
            sys.exit(f"{key} geometry drifted: {sizes} {tokens}")
        per_image_pixels[key] = row
        processed[key] = {
            "image_sizes": EXPECTED_SIZES[key],
            "num_img_tokens": EXPECTED_TOKENS[key],
            "pixel_values_sha256": bin_sha,
        }
    # The two-image packed rows (the "two-image" case's order) must be
    # exactly the per-image rows in prompt order.
    out_two = image_processor(
        images=[pil_images["wide"], pil_images["square"]], return_tensors="pt"
    )
    packed_two = out_two["pixel_values"]
    if tuple(packed_two.shape) != (2, 5, 3, 336, 336):
        sys.exit(f"two-image pixel_values {tuple(packed_two.shape)} unexpected")
    if not (
        torch.equal(packed_two[0], per_image_pixels["wide"])
        and torch.equal(packed_two[1], per_image_pixels["square"])
    ):
        sys.exit("two-image pixel_values != concat(one-image wide, one-image square)")

    # ------------------------------------------------------------- tokenizer
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))

    def encode_chunk(chunk: str) -> list[int]:
        return tokenizer.encode(chunk, add_special_tokens=True).ids

    # Per-chunk encoding facts pinned against the tiny vocabulary (the
    # Java expander encodes each chunk the same way).
    for chunk, expected_ids in (
        ("toast ", [20, 18, 13, 19, 20, 12]),
        ("toast", [20, 18, 13, 19, 20]),
        ("code", [15, 18, 16, 17]),
        ("toast code", [20, 18, 13, 19, 20, 12, 15, 18, 16, 17]),
        ("c d ", [15, 12, 16, 12]),
        (" e s", [12, 17, 12, 19]),
        ("", []),
    ):
        actual = encode_chunk(chunk)
        if actual != expected_ids:
            sys.exit(f"tiny tokenizer encoding drifted for {chunk!r}: {actual}")

    def expand_rendered(text: str, num_img_tokens: list[int]) -> list[int]:
        """Phi3VPromptExpander.expandRendered mirrored: split on the
        marker pattern, encode every chunk separately, expand marker i to
        num_img_tokens[i-1] sentinel rows of -i."""
        markers = list(MARKER_RE.finditer(text))
        if len(markers) != len(num_img_tokens):
            sys.exit(f"prompt holds {len(markers)} marker(s) but "
                     f"{len(num_img_tokens)} image(s) given")
        expected_index = 1
        out: list[int] = []
        last_end = 0
        for m in markers:
            if int(m.group(0)[len("<|image_"):-2]) != expected_index:
                sys.exit(f"marker {m.group(0)} is not continuous from 1")
            expected_index += 1
            out.extend(encode_chunk(text[last_end:m.start()]))
            out.extend([-expected_index + 1] * num_img_tokens[expected_index - 2])
            last_end = m.end()
        out.extend(encode_chunk(text[last_end:]))
        return out

    # ------------------------------------------------------------- forwards
    def full_forward(ids: list[int], pixels, image_sizes):
        """The fresh, full, uncached reference forward over the whole
        history (the WP6 decode reference, findings section 4): the
        bundled Phi3ImageEmbedding.forward substitutes the image rows
        natively from the sentinels still in input_ids, the all-ones
        attention mask (the reference processor's own mask) and
        use_cache=False. Returns the last-position float32 logits."""
        import torch

        length = len(ids)
        kwargs = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "use_cache": False,
        }
        if pixels is not None:
            kwargs["pixel_values"] = pixels
            kwargs["image_sizes"] = torch.tensor(image_sizes, dtype=torch.long)
        out = ref(**kwargs)
        return out.logits[0, -1]

    def projected_rows(
        pixels, image_sizes, expected_rows: int
    ) -> list[float]:
        """The bundled embedding's high-definition transform (which itself
        applies the shared img_projection, mirroring the Java tower's
        hdRows plus projector): the golden's intermediate projected-rows
        record, one 128-wide row per expanded sentinel row."""
        emb = ref.model.vision_embed_tokens
        img_features = emb.get_img_features(pixels.flatten(0, 1)).reshape(
            pixels.shape[0], pixels.shape[1], -1, CLIP_HIDDEN
        )
        rows = emb.hd_feature_transform(img_features, image_sizes)
        if tuple(rows.shape) != (expected_rows, TEXT_HIDDEN):
            sys.exit(
                f"projected rows {tuple(rows.shape)} != {(expected_rows, TEXT_HIDDEN)}"
            )
        return rows.flatten().tolist()

    # ------------------------------------------------------------- cases
    cases: list[dict] = []
    with torch.no_grad():
        for (name, prompt, image_keys, max_new), expected_length in zip(
            CASES, EXPECTED_LENGTHS
        ):
            num_img_tokens = [processed[k]["num_img_tokens"] for k in image_keys]
            expanded = expand_rendered(prompt, num_img_tokens)
            if len(expanded) != expected_length:
                sys.exit(f"{name}: expanded length {len(expanded)} != {expected_length}")
            pixels = None
            image_sizes = None
            if image_keys:
                pixels = torch.cat(
                    [per_image_pixels[k].unsqueeze(0) for k in image_keys], dim=0
                )
                image_sizes = [processed[k]["image_sizes"] for k in image_keys]

            case: dict = {
                "name": name,
                "prompt": prompt,
                "expanded_ids": compact_ids(expanded),
                "expanded_length": len(expanded),
                "max_new_tokens": max_new,
                "images": [IMAGE_FILENAME[k] for k in image_keys],
            }
            if image_keys:
                case["image_sizes"] = image_sizes
                case["num_img_tokens"] = num_img_tokens
                case["pixel_values_sha256"] = [
                    processed[k]["pixel_values_sha256"] for k in image_keys
                ]
                case["projected_rows"] = rounded(
                    projected_rows(pixels, image_sizes, sum(num_img_tokens))
                )

            # The step that first reaches total width ORIGINAL_MAX + 1
            # (entry index k -> forward width len(expanded) + k): the
            # golden's recompute step, null when the prompt is already
            # beyond the boundary (never re-triggered) or below it for
            # the whole generation.
            recompute_step = next(
                (k for k in range(max_new)
                 if len(expanded) + k == ORIGINAL_MAX_POSITIONS + 1),
                None,
            )
            if name == "wide-cross-boundary" and recompute_step != 6:
                sys.exit(f"boundary case recompute_step {recompute_step} != 6")

            decode_steps = []
            history: list[int] = []
            for _ in range(max_new):
                logits = full_forward(expanded + history, pixels, image_sizes)
                token = int(logits.argmax())
                history.append(token)
                decode_steps.append(
                    {"token_id": token, "logits": rounded(logits.tolist())}
                )
            case["recompute_step"] = recompute_step
            case["decode_steps"] = decode_steps
            cases.append(case)

    all_tokens = [s["token_id"] for c in cases for s in c["decode_steps"]]
    if len(set(all_tokens)) == 1:
        greedy_note = (
            f"every recorded step's greedy argmax is token {all_tokens[0]} on this "
            "21-token untied checkpoint, so the strict suite's exact-greedy-ID "
            "assertion is degenerate and a model stuck on that token would pass "
            "it; the binding gate is the full logit-vector comparison against "
            "this golden at 1e-4"
        )
    else:
        greedy_note = (
            "the recorded greedy IDs vary across steps (observed "
            f"{sorted(set(all_tokens))}); the strict suite asserts them exactly, "
            "but the binding gate is the full logit-vector comparison against "
            "this golden at 1e-4"
        )

    # ------------------------------------------------------------- golden
    result = {
        "schema": "jmlx-phase7-3c-wp6-phi3v-model-v1",
        "generated_by": "tools/hf-reference/phi3v_wp6.py",
        "provenance": {
            "transformers": environment["transformers"],
            "torch": environment["torch"],
            "torchvision": environment["torchvision"],
            "tokenizers": environment["tokenizers"],
            "safetensors": environment["safetensors"],
            "python": environment["python"],
            "platform": environment["platform"],
            "pillow": environment["pillow"],
            "numpy": environment["numpy"],
            "seed": SEED,
            "fixture_files": FIXTURE_SHAS,
            "norm_perturbation": {
                "seed": SEED,
                "generator": (
                    "dedicated torch.Generator (does not consume the global "
                    "model-init RNG)"
                ),
                "draw_order": (
                    "model.modules() order, weight then bias per norm; after "
                    "the last norm, glb_GN then sub_GN (N(0, 1) each)"
                ),
                "layer_norm_weight_distribution": "normal(mean=1.0, std=0.1)",
                "layer_norm_bias_distribution": "normal(mean=0.0, std=0.05)",
                "phi3_rms_norm_weight_distribution": "normal(mean=1.0, std=0.1)",
                "separator_distribution": "normal(mean=0.0, std=1.0)",
                "vision_layer_norms": counts["layer_norm"],
                "phi3_rms_norms": counts["phi3_rms_norm"],
                "note": (
                    "every norm affine is perturbed before the checkpoint is "
                    "saved so the 13 norms are mutually distinct and the two "
                    "zero-initialized 4096-wide learnable separators (glb_GN, "
                    "sub_GN) are non-zero; the strict golden discriminates "
                    "the composite model's norm and separator wiring"
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
                key: {
                    "width": width,
                    "height": height,
                    "bytes": image_paths[key].stat().st_size,
                    "sha256": sha256_file(image_paths[key]),
                    "source": f"tools/image-oracle/fixtures/phi3v/{filename}",
                    "source_sha256": source_sha,
                    "processed_pixel_values_sha256": bin_sha,
                }
                for key, filename, width, height, source_sha, bin_sha in IMAGE_SPECS
            },
            "pristine_clip_config": pristine_clip,
            "clip_swap_note": (
                "the bundled CLIP_VIT_LARGE_PATCH14_336_CONFIG constant (recorded "
                "pristine above) is swapped in the dynamic module for a tiny "
                "CLIPVisionConfig (image_size 336, patch 14, 3 channels, hidden "
                "1024, intermediate 128, 16 heads, 3 layers, projection dim 768, "
                "layer_norm_eps 1e-5, eager attention) before any model build; "
                "the width 1024 cannot shrink because "
                "reshape_hd_patches_2x2merge asserts L == 24*24 and C == 1024"
            ),
            "reference_dtype_note": (
                "every weight is rounded to bfloat16 before saving and before "
                "any golden is computed; the reference model loads the stored "
                "bf16 tensors up-converted to float32 (strict load), so the "
                "golden exercises the exact stored-bf16 -> float32 path the "
                "Java model uses when it upcasts the checkpoint"
            ),
            "shared_embedding_note": (
                "the bundled Phi3VModel registers its single embed_tokens "
                "module twice: Phi3ImageEmbedding is built with "
                "wte=self.embed_tokens (the commented-out weight copy right "
                "below the construction is a no-op because the module object "
                "is shared), so model.state_dict() carries 77 keys for 76 "
                "unique tensors. The checkpoint stores the shared embedding "
                "once, under model.embed_tokens.weight (the generator asserts "
                "the two state-dict keys share storage before saving); the "
                "reference's strict load re-supplies the duplicate "
                "model.vision_embed_tokens.wte.weight key from the same "
                "tensor. Text rows in every forward (image and text-only) "
                "are embedded from that one table, so the golden's Java-side "
                "embed_tokens lookup is the reference's own path"
            ),
            "attention_implementation": "eager",
            "attention_note": (
                "Phi3VPreTrainedModel._supports_sdpa is False, so eager (or "
                "FA2) is the only legal implementation; the generator sets "
                "_attn_implementation = 'eager' on both build configs and "
                "asserts it on the built model and its CLIP vision model after "
                "construction"
            ),
            "attention_mask_note": (
                "the reference's own processor mask is (ids > -1000000) = all "
                "ones; every forward passes an explicit all-ones long mask, "
                "which the WP1 probe pinned as equivalent to None under the "
                "bundled 4d causal-mask builder (sliding_window omitted, "
                "bundled default None)"
            ),
            "pad_token_id_note": (
                "config.json carries pad_token_id 0: the bundled "
                "Phi3VModel.__init__ builds nn.Embedding(vocab_size, "
                "hidden_size, padding_idx=config.pad_token_id) and the "
                "bundled config default 32000 is out of range for a 21-token "
                "vocabulary; _init_weights zeros the pad row (row 0) of the "
                "stored embedding"
            ),
            "embd_pdrop_note": (
                "the config omits embd_pdrop (bundled default 0.0), so "
                "Phi3VModel.embed_dropout and Phi3ImageEmbedding.drop are "
                "nn.Dropout(0.0), no-ops under the reference's eval mode"
            ),
            "encoding_note": (
                "the checkpoint tokenizer is the 21-token phi3v-tiny fixture "
                "(no added tokens, a ByteLevel post-processor with no special "
                "tokens), so the per-chunk add_special_tokens=True encoding "
                "adds nothing -- no per-encoding <s> (unlike the 32k artifact "
                "tokenizer). Chunks are encoded with the pinned tokenizers "
                "library loaded from the checkpoint's own tokenizer.json, "
                "mirroring Phi3VPromptExpander.expandRendered's per-chunk "
                "HfTokenizer.encode(chunk, addSpecialTokens=true)"
            ),
            "tower_note": (
                "the reference runs the full 3-layer CLIP encoder over all five "
                "slots per image (the zero-padded slots included) and takes "
                "hidden_states[-2] -- the state after block 1; Java executes "
                "the two executed blocks over the real slots only and loads "
                "the third block plus the pooled-output post_layernorm without "
                "executing them: the extra reference compute is discarded and "
                "the retained rows are bit-identical"
            ),
            "decode_reference_note": (
                "every step's logits (entry 0 is the prefill point) are a "
                "fresh full uncached forward over the expanded prompt plus the "
                "generated history, with the sentinel IDs intact in "
                "input_ids: the bundled Phi3ImageEmbedding.forward substitutes "
                "the image rows natively (clamp_min(0).clamp_max(vocab_size) "
                "then index_put), so the golden never exercises the "
                "boundary-recompute of prepare_inputs_for_generation -- that "
                "path (pinned lines 1650-1655) is recorded for the Java "
                "re-prefill design only"
            ),
            "rope_note": (
                "NTK-Su scaling with uniform short factors 1.0 and long "
                "factors 2.0 over the 32 head-dim/2 frequencies: every width "
                "at or below 470 runs the short table, every width above it "
                "the long table; the wide-cross-boundary case (465 prompt "
                "tokens) crosses at total width 471, re-running the whole "
                "history as a fresh prefill (golden recompute_step 6), while "
                "the wide-long-prompt case (473) shows the image rows alone "
                "(457) below the boundary pin the selection to the finished "
                "prompt length"
            ),
            "greedy_id_note": greedy_note,
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
    print(f"preprocessor_config.json: "
          f"{sha256_file(checkpoint_dir / 'preprocessor_config.json')}")
    print(f"tokenizer.json: {sha256_file(checkpoint_dir / 'tokenizer.json')}")
    for key, filename, _width, _height, _s, _b in IMAGE_SPECS:
        path = image_paths[key]
        print(f"{path.name}: {sha256_file(path)} ({path.stat().st_size} bytes)")
    print(f"norm counts: {counts}")
    for case in cases:
        steps = len(case["decode_steps"])
        tokens = [s["token_id"] for s in case["decode_steps"]]
        print(f"{case['name']}: P {len(case['prompt'])} prompt chars, "
              f"{case.get('recompute_step')} recompute, {steps} steps, "
              f"tokens {tokens}")


if __name__ == "__main__":
    main()
