#!/usr/bin/env python3
"""Phase 7.3c WP1 reference probes for microsoft/Phi-3.5-vision-instruct.

Runs in the hash-locked tools/hf-reference venv (Torch 2.6.0 CPU + torchvision 0.21.0,
transformers 4.57.6, fully offline). Usage:

    .venv/bin/python phi3v_probe.py --directory <artifact-files> --out goldens/phi3v-wp1-probe.json

<artifact-files> must contain the 13 WP0 files (the configs, the index, the tokenizer files,
special_tokens_map, LICENSE and the four bundled trust-remote-code .py files). The script
verifies every file's SHA-256 before importing or using the bundled modules. The safetensors
shards are not needed: every model probe uses a tiny randomly initialized config, and the
golden generators (WP6/WP8) are the only code that loads real weights.

Probes (plan req/plans/phase7-3c-plan.md, WP1 items):
  pins  SHA-256 verification of the WP0 files
  env   package versions and platform
  p1    fresh prefill with use_cache=True -> the exact cache-API failure (item 2)
  p2    prepare_inputs_for_generation with a populated DynamicCache past that crosses the
        original_max boundary -> the exact seen_tokens failure (item 13)
  p3    bundled tokenizer chunk ids: empty / no-leading-space / leading-space (item 9)
  p4    attention defaults: decoder flash_attention_2 from the artifact config, tower sdpa
        from the bare bundled CLIP config, and that the eager-set construction works (item 3)
        (runs first: p1/p2 swap the module-level CLIP constant for the tiny eager tower)
  p5    torchvision F.resize(PIL) vs PIL.Image.resize(BILINEAR) byte identity (items 12-13)
  p6    full bundled processor: pinned image cases and the text/marker expansion (items 4-9)
  tok   tokenizer.json structure: model flags, normalizer, decoder chain, special tokens
  stop  stop-token facts: eos/pad ids, decode behavior (item 14)

Never invoked from Gradle. The results JSON is digest-pinned in provenance.json.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

REVISION = "12b77fb40b63a2c73c68243d3f767aab688a1b2a"
# The 13-char endoftext literal is built by concatenation so this script (and any log of it)
# survives the transport that strips the literal from messages.
ENDOFTTEXT = "<|" + "endoftext" + "|>"

PINS = {
    "LICENSE": "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383",
    "config.json": "567e28c638a3e41aba1a304ae58b283845ad07e1c60e970643b0ad8fc229b382",
    "configuration_phi3_v.py": "18ff44c5c1f9a6523184fb2e580e03ebf722a65f37b9cc57cd1a6d620a0f2666",
    "generation_config.json": "bf86d2cd9250f89f09563b7f80e25e34663f673b836f531dc207362165051dc9",
    "model.safetensors.index.json": "72d628080d85b3a0e96eac71a47f7a04a666aabf8a91269100c99a8d543f345c",
    "modeling_phi3_v.py": "b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c",
    "preprocessor_config.json": "0f3d920fcec587bb425312d30b0660e1fabaf46a75961f2b75ddbb1ff92de01c",
    "processor_config.json": "286a9e7fb58a489eae6825b79be69f62068c2028435c29ee52c3a485d071707b",
    "processing_phi3_v.py": "990ef3d1f6b5cc61b44cf8ec8e8886b19721b7e01e8bae3ebea32781b0fcf598",
    "sample_inference.py": "49b1246f2a18d9b4a1e12892e50d19929e550263075d4c80cedeac2246546da4",
    "special_tokens_map.json": "f9535fada19315e0d8eb89521cebe111c008fee98d16e7c6744acfb50505d8c3",
    "tokenizer.json": "493f8ad437be806d5ccdc57e91479bdbb0c656448c1b89ff70f7653530d78518",
    "tokenizer_config.json": "581d4654c018fbcef2504b57a805dcad37126038137b5d25d22b5a02abc1ef29",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_pins(directory: Path) -> dict:
    result = {}
    for name, expected in sorted(PINS.items()):
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"missing pinned file: {path}")
        actual = sha256_bytes(path.read_bytes())
        result[name] = {"sha256": actual, "ok": actual == expected, "size": path.stat().st_size}
        if actual != expected:
            raise SystemExit(f"pinned file mismatch: {name}: {actual} != {expected}")
    return result


def env_info() -> dict:
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    import numpy
    import torch
    import torchvision

    return {
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": version("transformers"),
        "pillow": version("Pillow"),
        "numpy": numpy.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def record_exception(exc: BaseException) -> dict:
    frames = []
    for frame in traceback.extract_tb(exc.__traceback__):
        if "modeling_phi3_v" in frame.filename or "cache_utils" in frame.filename:
            frames.append(
                {
                    "file": os.path.basename(frame.filename),
                    "lineno": frame.lineno,
                    "line": (frame.line or "").strip(),
                }
            )
    return {"type": type(exc).__name__, "message": str(exc), "frames": frames[-6:]}


def tiny_clip_config(attn_implementation: str):
    from transformers import CLIPVisionConfig

    return CLIPVisionConfig(
        image_size=336,
        patch_size=14,
        num_channels=3,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        projection_dim=64,
        layer_norm_eps=1e-5,
        attn_implementation=attn_implementation,
    )


def tiny_phi3v_config(config_cls, attn_implementation: str):
    # Tiny decoder: head_dim = 64/4 = 16, so the su factor arrays have 16//2 = 8 entries.
    # original_max = 16, max = 256; the full CLIP constant is swapped for a 2-layer
    # hidden-64 tower before the model is built (module-level constant, read at init).
    return config_cls(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        original_max_position_embeddings=16,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling={
            "type": "su",
            "short_factor": [1.0] * 8,
            "long_factor": [1.5] * 8,
        },
        use_cache=True,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        sliding_window=32,
        img_processor={
            "name": "clip_vision_model",
            "model_name": "openai/clip-vit-large-patch14-336",
            "image_dim_out": 64,
            "num_img_tokens": 144,
        },
        embd_layer={
            "embedding_cls": "image",
            "hd_transform_order": "sub_glb",
            "projection_cls": "mlp",
            "use_hd_transform": True,
            "with_learnable_separator": True,
        },
        attn_implementation=attn_implementation,
    )


def p1_cache_api_failure(model_cls, tiny_cfg, model_mod) -> dict:
    import torch

    model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG = tiny_clip_config("eager")
    try:
        model = model_cls(tiny_cfg)
    except Exception as exc:
        return {"raised": "construction-failed", "exception": record_exception(exc)}
    try:
        model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        return {"raised": None, "note": "fresh prefill with use_cache=True unexpectedly succeeded"}
    except Exception as exc:
        return {"raised": "fresh-prefill", "exception": record_exception(exc)}


def p2_boundary_recompute_failure(model_cls, tiny_cfg, model_mod) -> dict:
    import torch
    from transformers import DynamicCache

    model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG = tiny_clip_config("eager")
    try:
        model = model_cls(tiny_cfg)
    except Exception as exc:
        return {"raised": "construction-failed", "exception": record_exception(exc)}

    # A populated DynamicCache with original_max = 16 cached positions; input_ids has
    # original_max + 1 = 17 rows, so the boundary block of prepare_inputs_for_generation
    # is entered.
    cache = DynamicCache()
    kv = torch.zeros(1, 2, 16, 16)  # (batch, num_kv_heads, seq, head_dim)
    for layer_idx in range(2):
        cache.update(kv, kv, layer_idx)
    input_ids = torch.arange(17, dtype=torch.long).unsqueeze(0) % 32
    attention_mask = torch.ones(1, 17, dtype=torch.long)
    try:
        model.prepare_inputs_for_generation(
            input_ids, past_key_values=cache, attention_mask=attention_mask
        )
        return {"raised": None, "note": "boundary prepare_inputs_for_generation unexpectedly succeeded"}
    except Exception as exc:
        return {"raised": "boundary-prepare", "exception": record_exception(exc)}


def p3_chunk_tokenization(directory: Path) -> dict:
    from transformers import LlamaTokenizerFast

    tok = LlamaTokenizerFast.from_pretrained(str(directory))
    cases = {
        "empty": "",
        "no_leading_space": "hello",
        "leading_space": " hello",
    }
    return {name: list(tok(text).input_ids) for name, text in cases.items()}


def p4_attention_defaults(model_cls, config, model_mod) -> dict:
    from transformers import CLIPVisionConfig, CLIPVisionModel

    const = model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG
    record = {
        "artifact_config_attn_implementation": config._attn_implementation,
        "bundled_clip_constant_attn_implementation": const._attn_implementation,
        "bare_clipvisionconfig_attn_implementation": CLIPVisionConfig()._attn_implementation,
        "clipvisionmodel_default_attn_implementation": getattr(
            CLIPVisionModel, "_default_attn_implementation", None
        ),
    }

    # The full bundled constant (hidden 1024, 24 layers) built as-is, to record which
    # attention class the tower selects by default.
    clone = CLIPVisionConfig(**{k: v for k, v in const.to_dict().items() if not k.startswith("_")})
    clone._attn_implementation = const._attn_implementation
    tower = CLIPVisionModel(clone)
    record["tower_default_self_attn_class"] = type(
        tower.vision_model.encoder.layers[0].self_attn
    ).__name__
    record["tower_config_attn_after_build"] = tower.config._attn_implementation
    # 4.57.6 resolves None -> sdpa (or -> eager if _sdpa_can_dispatch fails) into the
    # internal attribute at construction; that is the value the forward dispatch uses.
    record["tower_attn_internal_after_build"] = tower.config._attn_implementation_internal
    del tower

    # The generator construction path: decoder eager + tiny eager tower constant.
    tiny_cfg = tiny_phi3v_config(type(config), "eager")
    model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG = tiny_clip_config("eager")
    try:
        model = model_cls(tiny_cfg)
    except Exception as exc:
        record["eager_construction"] = {"failed": True, "exception": record_exception(exc)}
        return record
    record["eager_construction"] = {
        "failed": False,
        "decoder_self_attn_class": type(model.model.layers[0].self_attn).__name__,
        "tower_self_attn_class": type(
            model.model.vision_embed_tokens.img_processor.vision_model.encoder.layers[0].self_attn
        ).__name__,
        "model_config_attn_after_build": model.config._attn_implementation,
        "tower_config_attn_after_build": model.model.vision_embed_tokens.img_processor.config._attn_implementation,
    }
    del model
    return record


def p5_pil_handoff() -> dict:
    import numpy as np
    import torch  # noqa: F401  (ensure the torch runtime is up for the torchvision kernels)
    from PIL import Image
    from torchvision import transforms as T

    # (source_width, source_height) -> ([resize_h, resize_w]), seed
    results = []
    for (w, h), (rh, rw), seed in [
        ((301, 300), (669, 672), 201),  # the 300x301 truncation case, transposed space
        ((336, 336), (672, 672), 202),  # the square case
        ((672, 336), (336, 672), 203),  # the wide case
        ((251, 137), (784, 168), 204),  # odd upscale
        ((511, 223), (400, 909), 205),  # odd downscale
    ]:
        arr = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        pil = Image.fromarray(arr, "RGB")
        tv = T.functional.resize(pil, [rh, rw])
        pr = pil.resize((rw, rh), Image.Resampling.BILINEAR)
        tv_bytes = tv.tobytes()
        pr_bytes = pr.tobytes()
        results.append(
            {
                "source": [w, h],
                "resize": [rh, rw],
                "identical_bytes": tv_bytes == pr_bytes,
                "torchvision_sha256": sha256_bytes(tv_bytes),
                "pil_sha256": sha256_bytes(pr_bytes),
            }
        )
    return {"torchvision_default_interpolation": "BILINEAR", "cases": results}


def compact_ids(ids):
    out = []
    i = 0
    while i < len(ids):
        if ids[i] < 0:
            j = i
            while j < len(ids) and ids[j] == ids[i]:
                j += 1
            out.append({"neg_run": ids[i], "count": j - i})
            i = j
        else:
            out.append(ids[i])
            i += 1
    return out


def p6_processor(directory: Path) -> dict:
    import numpy as np
    from PIL import Image
    from transformers import LlamaTokenizerFast
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    proc_cls = get_class_from_dynamic_module("processing_phi3_v.Phi3VProcessor", str(directory))
    proc_mod = sys.modules[proc_cls.__module__]
    image_processor = proc_mod.Phi3VImageProcessor.from_pretrained(str(directory))
    tokenizer = LlamaTokenizerFast.from_pretrained(str(directory))
    processor = proc_cls(image_processor=image_processor, tokenizer=tokenizer)

    def make_pil(w, h, seed):
        arr = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def case_record(name, images):
        out = image_processor(images=images, return_tensors="pt")
        pv = out["pixel_values"]
        slot_shas = [
            sha256_bytes(pv[i, s].numpy().tobytes()) for i in range(pv.shape[0]) for s in range(pv.shape[1])
        ]
        return {
            "name": name,
            "pixel_values_shape": list(pv.shape),
            "pixel_values_dtype": str(pv.dtype),
            "pixel_values_sha256": sha256_bytes(pv.numpy().tobytes()),
            "slot_sha256": slot_shas,
            # BatchFeature(return_tensors="pt") converts these to tensors; record plain ints.
            "image_sizes": [[int(x) for x in row] for row in out["image_sizes"]],
            "num_img_tokens": [int(x) for x in out["num_img_tokens"]],
        }

    square = make_pil(336, 336, 101)
    wide = make_pil(672, 336, 102)
    tall = make_pil(336, 672, 103)
    trunc = make_pil(300, 301, 104)
    image_cases = [
        case_record("square-336x336", [square]),
        case_record("wide-672x336", [wide]),
        case_record("tall-336x672", [tall]),
        case_record("trunc-300x301", [trunc]),
        case_record("two-image-square+wide", [square, wide]),
    ]

    def marker(i):
        return f"<|image_{i}|>"

    def text_case(name, text, images):
        out = processor(text=text, images=images)
        ids = out["input_ids"][0].tolist()
        mask = out["attention_mask"][0]
        return {
            "name": name,
            "text": text,
            "input_ids": compact_ids(ids),
            "total_len": len(ids),
            "num_negatives": sum(1 for x in ids if x < 0),
            "attention_mask_min": int(mask.min()),
            "attention_mask_max": int(mask.max()),
            "pixel_values_shape": list(out["pixel_values"].shape),
        }

    text_cases = [
        text_case("mid-marker", "A" + marker(1) + "B", [square]),
        text_case("leading-marker", marker(1) + "tail", [square]),
        text_case("trailing-marker", "mid" + marker(1), [square]),
        text_case("adjacent-markers", marker(1) + marker(2), [square, wide]),
    ]
    return {"image_cases": image_cases, "text_cases": text_cases}


def tok_structure(directory: Path) -> dict:
    tj = json.loads((directory / "tokenizer.json").read_text())
    tc = json.loads((directory / "tokenizer_config.json").read_text())
    stm = json.loads((directory / "special_tokens_map.json").read_text())

    specials_of_interest = {}
    wanted = {1, 2, 13, 32000, 32007, 32044}
    for tok in tj.get("added_tokens", []):
        if tok.get("id") in wanted:
            content = tok.get("content", "")
            specials_of_interest[str(tok["id"])] = {
                "content": content,
                "content_ords": [ord(c) for c in content],
                "special": tok.get("special"),
            }
    model = dict(tj.get("model", {}))
    model_vocab_size = len(model.pop("vocab", {}))
    merges_size = len(model.pop("merges", []))
    added = tj.get("added_tokens", [])
    return {
        "model_vocab_size": model_vocab_size,
        "merges_count": merges_size,
        "added_tokens_count": len(added),
        "added_tokens_max_id": max((t["id"] for t in added), default=None),
        "model": model,
        "normalizer": tj.get("normalizer"),
        "pre_tokenizer": tj.get("pre_tokenizer"),
        "decoder": tj.get("decoder"),
        "post_processor": tj.get("post_processor"),
        "specials": specials_of_interest,
        "tokenizer_config": {
            "tokenizer_class": tc.get("tokenizer_class"),
            "model_max_length": tc.get("model_max_length"),
            "add_bos_token": tc.get("add_bos_token"),
            "add_prefix_space": tc.get("add_prefix_space"),
            "chat_template": tc.get("chat_template"),
            "additional_special_tokens": tc.get("additional_special_tokens"),
            "extra_ids": tc.get("extra_ids"),
        },
        "special_tokens_map": stm,
    }


def stop_tokens(directory: Path) -> dict:
    from transformers import LlamaTokenizerFast

    tok = LlamaTokenizerFast.from_pretrained(str(directory))
    gc = json.loads((directory / "generation_config.json").read_text())
    si = (directory / "sample_inference.py").read_text()
    si_generate_lines = [
        line.strip() for line in si.splitlines() if "eos_token_id" in line or "generate(" in line
    ]

    def decode(ids, skip):
        text = tok.decode(ids, skip_special_tokens=skip)
        return {"text": text, "text_ords": [ord(c) for c in text]}

    return {
        "tok_eos_token_id": tok.eos_token_id,
        "tok_pad_token_id": tok.pad_token_id,
        "tok_bos_token_id": tok.bos_token_id,
        "ids": {
            "s": tok.convert_tokens_to_ids("<s>"),
            "end_of_s": tok.convert_tokens_to_ids("</s>"),
            "end_turn": tok.convert_tokens_to_ids("<|end|>"),
            "image": tok.convert_tokens_to_ids("<|image|>"),
            "endoftext": tok.convert_tokens_to_ids(ENDOFTTEXT),
        },
        "decode_endoftext": {
            "skip_false": decode([32000], False),
            "skip_true": decode([32000], True),
        },
        "decode_end_turn": {
            "skip_false": decode([32007], False),
            "skip_true": decode([32007], True),
        },
        "decode_newline_byte": decode([13], False),
        "decode_end_of_s": decode([2], False),
        "generation_config": gc,
        "sample_inference_eos_lines": si_generate_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="directory with the WP0 files")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "goldens" / "phi3v-wp1-probe.json",
    )
    args = parser.parse_args()

    import transformers

    transformers.logging.set_verbosity_error()
    result = {"revision": REVISION, "env": env_info(), "pins": verify_pins(args.directory)}

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(str(args.directory), trust_remote_code=True)
    model_cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], str(args.directory))
    model_mod = sys.modules[model_cls.__module__]

    tiny_cfg = tiny_phi3v_config(type(config), "eager")
    # p4 must run first: it records the pristine module-level CLIP constant, and p1/p2
    # (and p4's own eager construction) swap that constant for the tiny eager tower.
    result["p4"] = p4_attention_defaults(model_cls, config, model_mod)
    result["p1"] = p1_cache_api_failure(model_cls, tiny_cfg, model_mod)
    result["p2"] = p2_boundary_recompute_failure(model_cls, tiny_cfg, model_mod)
    result["p3"] = p3_chunk_tokenization(args.directory)
    result["p5"] = p5_pil_handoff()
    result["p6"] = p6_processor(args.directory)
    result["tok"] = tok_structure(args.directory)
    result["stop"] = stop_tokens(args.directory)
    result["deferred_128k"] = {
        "status": "not accessible (gated); num_crops unverified; no support claim",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    json.dump(result, buffer, indent=2, sort_keys=True)
    buffer.write("\n")
    args.out.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"wrote {args.out} (sha256 {sha256_bytes(buffer.getvalue().encode())})")
    print(f"p1: {result['p1'].get('raised')} {result['p1'].get('exception', {}).get('message', '')}")
    print(f"p2: {result['p2'].get('raised')} {result['p2'].get('exception', {}).get('message', '')}")
    print(f"p4 tower default: {result['p4'].get('tower_default_self_attn_class')}")
    print(f"p5 identical: {[c['identical_bytes'] for c in result['p5']['cases']]}")
    for case in result["p6"]["image_cases"]:
        print(
            f"p6 {case['name']}: sizes={case['image_sizes']} tokens={case['num_img_tokens']} "
            f"shape={case['pixel_values_shape']}"
        )


if __name__ == "__main__":
    main()
