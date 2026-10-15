#!/usr/bin/env python3
"""Phase 7.3b WP1 item 5: quantized 8B candidate probe (MLX-affine runtime).

Runs under tools/qwen-vl-reference/.venv (CPython 3.12, mlx 0.31.2,
mlx-metal 0.31.2, mlx-vlm 0.3.9, transformers 4.57.6) on an Apple Silicon
GPU. It is never invoked by Gradle; it is a manual, opt-in evidence tool.
Its output JSON is SHA-256-pinned in tools/qwen-vl-reference/provenance.json.

Runtime facts recorded here:

  * the artifact is an MLX-affine 4-bit (group 64) conversion of
    ``Qwen/Qwen3-VL-8B-Instruct`` (converter identity/version per its README);
    language projection/embedding/lm_head weights are U32-packed (8 int4 per
    u32) with per-group scales and biases; the vision tower, DeepStack
    mergers and all norms are BF16;
  * ``pixel_values`` are injected from the lossless ``slow_compensated``
    exports written by ``tools/image-oracle/fast-comparison/runner.py`` and
    re-verified (SHA-256) against that runner's manifest, so this probe
    consumes exactly the preprocessing patches the HF reference consumed;
  * ``input_ids`` are built self-contained: render the artifact's own chat
    template (byte-identical SHA to the 2B reference template), tokenize
    with the artifact's tokenizer, replace the single image token with
    ``N = prod(grid) // merge**2`` copies. mlx-vlm's ``prepare_inputs``
    inserts exactly one image token and cannot be reused for the multi-token
    Qwen3-VL expansion;
  * the ``AutoProcessor`` compatibility probe records only deterministic
    fields on failure: the exception class name and the SORTED list of
    missing framework names parsed from the message (the "requires the
    <Framework> library but it was not found" lines). The raw message is
    never recorded: transformers assembles its two missing-library
    paragraphs in a process-dependent order, so the raw text would break
    byte-for-byte golden reproducibility;
  * the KV cache is ``mlx_lm.models.cache.KVCache`` via
    ``mlx_vlm.models.cache.make_prompt_cache(model.language_model)``;
    prefill is ``model(input_ids, pixel_values, mask, cache,
    image_grid_thw=...)`` and decode steps call
    ``model.language_model(current, cache=cache)`` (the mlx-vlm
    ``generate_step`` pattern).

No parity claim: this is a different runtime (Metal, 4-bit quantized) than
the pinned HF CPU float32 reference. The output is compatibility and
provenance evidence only, cross-referenced against the 2B HF probe golden
for the sequence-expansion facts that MUST match (same template, prompt,
tokenizer and grids).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ORG = "mlx-community"
REPO = "Qwen3-VL-8B-Instruct-4bit"
REVISION = "defcdea7cc7a4b0858fea563cbbce171d328e457"
SCHEMA = "jmlx-phase7-3b-wp1-qwen3vl-mlx8b-v1"
IMAGE_TOKEN_ID = 151655
MODEL_IMAGES = ("model-rgb-256x256.png", "model-rgb-64x32.png")
PROMPT_MESSAGES = [
    {"role": "user",
     "content": [
         {"type": "image"},
         {"type": "text", "text": "Describe the image in one short sentence."},
     ]}
]

# Pinned digests of the fast-comparison runner's manifest.json, the output of
# tools/image-oracle/fast-comparison/runner.py produced by its pinned venv
# (a manual, reviewed step: regenerating the runner output changes this). The
# lock digest is that comparison environment's requirements.lock digest,
# carried in the manifest's own "lock_sha256" field. A manifest that does not
# match BOTH digests came from a drifted environment and its exports must not
# be fed to a model.
FAST_COMPARISON_MANIFEST_SHA256 = (
    "c3af7ebe5591c365f8fa199bcc2d54e5f7f7a6b02233f9213d8869082b21bb00"
)
FAST_COMPARISON_LOCK_SHA256 = (
    "4f2e898b8ad91228741f8adb4ca7e1f321c6ec37619711cf9c6a10879eb442d5"
)

# Cross-reference values from the committed HF reference evidence
# (tools/hf-reference/goldens/qwen3-vl-2b-wp1-probe.json, schema
# jmlx-phase7-3b-wp1-qwen3vl-probe-v1). The rendered prompt comes from the
# byte-identical chat template (sha 3636d0f0bd6bef02...) with the identical
# prompt, and the sequence facts follow from the same grids, so every one of
# these must hold for the 8B probe to be comparable.
EXPECTED_RENDERED_PROMPT_SHA256 = (
    "c58e38df71e55a90a320171f1f007a2cffc975056b33398ee704830799c4dfcc"
)
HF_REFERENCE = {
    "model-rgb-256x256.png": {
        "grid_thw": [1, 16, 16],
        "prompt_length_after_expansion": 82,
        "image_token_span": [4, 67],
        "rope_delta": -56,
        "greedy_ids_2b": [32, 10876, 374, 15, 220],
    },
    "model-rgb-64x32.png": {
        "grid_thw": [1, 12, 24],
        "prompt_length_after_expansion": 90,
        "image_token_span": [4, 75],
        "rope_delta": -60,
        "greedy_ids_2b": [32, 10876, 374, 13, 13],
    },
}


def gate_versions() -> dict:
    import importlib.metadata as md

    def ver(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "missing"

    environment = {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "mlx": ver("mlx"),
        "mlx_metal": ver("mlx-metal"),
        "mlx_lm": ver("mlx-lm"),
        "mlx_vlm": ver("mlx-vlm"),
        "transformers": ver("transformers"),
        "huggingface_hub": ver("huggingface_hub"),
        "tokenizers": ver("tokenizers"),
        "safetensors": ver("safetensors"),
        "numpy": ver("numpy"),
        "pillow": ver("pillow"),
    }
    required = {
        "python": platform.python_version_tuple()[:2] == ("3", "12"),
        "mlx": environment["mlx"] == "0.31.2",
        "mlx_metal": environment["mlx_metal"] == "0.31.2",
        "mlx_lm": environment["mlx_lm"] == "0.29.1",
        "mlx_vlm": environment["mlx_vlm"] == "0.3.9",
        "transformers": environment["transformers"] == "4.57.6",
        "huggingface_hub": environment["huggingface_hub"] == "0.36.2",
        "tokenizers": environment["tokenizers"] == "0.22.2",
        "safetensors": environment["safetensors"] == "0.8.0",
        "numpy": environment["numpy"] == "2.5.3",
        "pillow": environment["pillow"] == "12.3.0",
    }
    failed = sorted(name for name, ok in required.items() if not ok)
    if failed:
        sys.exit(f"qwen3vl mlx probe pin mismatch for {failed}: {environment}")
    return environment


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mx_dtype(value) -> str:
    return str(value).removeprefix("mlx.core.")


def verify_exports(patches_dir: Path) -> dict:
    """Load the slow_compensated lossless exports, re-verified by SHA-256."""
    import numpy as np

    manifest_path = patches_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"fast-comparison manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "jmlx-phase7-3b-wp1-qwen3vl-fastcomparison-v1":
        raise SystemExit("unexpected fast-comparison manifest schema")
    manifest_digest = sha256_bytes(manifest_path.read_bytes())
    if manifest_digest != FAST_COMPARISON_MANIFEST_SHA256:
        raise SystemExit(
            f"fast-comparison manifest sha256 {manifest_digest} does not match "
            f"the pinned {FAST_COMPARISON_MANIFEST_SHA256}; the patches came "
            "from a drifted fast-comparison environment")
    if manifest["lock_sha256"] != FAST_COMPARISON_LOCK_SHA256:
        raise SystemExit(
            f"fast-comparison manifest lock_sha256 "
            f"{manifest['lock_sha256']} does not match the pinned "
            f"{FAST_COMPARISON_LOCK_SHA256}; the patches came from a drifted "
            "fast-comparison environment")
    by_file = {rec["file"]: rec for rec in manifest["images"]}
    exports = {}
    for image_name in MODEL_IMAGES:
        record = by_file[image_name]
        cfg = record["configs"]["slow_compensated"]
        export_name = cfg["lossless_export"]
        if not export_name:
            raise SystemExit(f"missing lossless export for {image_name}/slow_compensated")
        array = np.load(patches_dir / export_name)
        digest = sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())
        if digest != cfg["pixel_values_sha256"]:
            raise SystemExit(
                f"export {export_name} no longer matches the runner manifest")
        exports[image_name] = {
            "export": export_name,
            "grid_thw": cfg["grid_thw"],
            "dtype": cfg["dtype"],
            "shape": cfg["shape"],
            "pixel_values_sha256": digest,
            "manifest_uint8_patches_sha256": cfg["uint8_patches_sha256"],
        }
    return {
        "exports": exports,
        "manifest_sha256": manifest_digest,
        "lock_sha256": manifest["lock_sha256"],
        "environment": manifest["environment"],
    }


def leaf_info(module) -> dict:
    entry = {"class": type(module).__name__}
    if hasattr(module, "weight"):
        entry["weight_dtype"] = mx_dtype(module.weight.dtype)
        entry["weight_shape"] = list(module.weight.shape)
    return entry


def linear_kernels(root) -> dict:
    counts = Counter()
    for module in root.modules():
        if type(module).__name__ == "QuantizedLinear":
            counts["nn.QuantizedLinear"] += 1
        elif type(module).__name__ == "Linear":
            counts["nn.Linear"] += 1
    return dict(sorted(counts.items()))


def module_histogram(root) -> dict:
    counts = Counter(type(m).__name__ for m in root.modules())
    return dict(sorted(counts.items()))


def top20(last_logits):
    import mlx.core as mx

    order = mx.argsort(-last_logits)
    indices = order[:20]
    values = last_logits[indices]
    return [[int(i), float(v)]
            for i, v in zip(indices.tolist(), values.tolist())]


def parse_missing_frameworks(message: str):
    """Extract the deterministic missing-library finding from a transformers
    ImportError message: the framework names in "requires the <Framework>
    library but it was not found" lines, sorted for determinism. Returns
    None when no such line is found, so the caller records only the
    exception class name and never the raw (order-unstable) message."""
    import re

    names = re.findall(
        r"requires the ([A-Z]\w*) library but it was not found", message)
    return sorted(set(names)) if names else None


def check_facts(image_name: str, field: str, actual, expected, note: str = ""):
    """Sequence-expansion fact gate: an explicit check (not an assert), so it
    cannot be disabled by python -O, and a mismatch exits non-zero, as the
    README promises. Returns whether the fact matched."""
    if actual != expected:
        suffix = f" ({note})" if note else ""
        sys.exit(f"{image_name}: {field} {actual} != HF reference {expected}"
                 + suffix)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = (Path(__file__).resolve().parent.parent.parent
                   / ".tier-b" / "qwen3-vl-8b-instruct-4bit")
    parser.add_argument("--directory", type=Path, default=default_dir,
                        help="the pinned mlx-community Qwen3-VL-8B-Instruct-4bit snapshot")
    parser.add_argument("--patches-dir", type=Path, required=True,
                        help="fast-comparison runner output directory")
    default_out = (Path(__file__).resolve().parent
                   / "goldens" / "qwen3-vl-8b-wp1-mlx-probe.json")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--skip-model", action="store_true",
                        help="run inventory + configs only (no model loading)")
    args = parser.parse_args()

    directory: Path = args.directory
    if not (directory / "config.json").is_file():
        sys.exit(f"no config.json under {directory}; pass --directory to the snapshot")

    environment = gate_versions()
    lock_path = Path(__file__).resolve().parent / "requirements.lock"
    environment["requirements_lock_sha256"] = sha256_file(lock_path)
    import mlx.core as mx
    environment["mlx_default_device"] = str(mx.default_device())

    result: dict = {
        "schema": SCHEMA,
        "generated_by": "tools/qwen-vl-reference/run_qwen3vl_mlx_probe.py",
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
            if hub_entry["lfs_sha256"] is not None and entry["sha256"] != hub_entry["lfs_sha256"]:
                sys.exit(f"hub LFS SHA-256 mismatch for {entry['path']}")
            if hub_entry["bytes"] is not None and entry["bytes"] != hub_entry["bytes"]:
                sys.exit(f"hub size mismatch for {entry['path']}")
    result["artifact"]["hub"] = {
        "commit_sha": hub.sha,
        "tags": sorted(hub.tags),
        "license": (hub.cardData or {}).get("license") if hub.cardData else None,
    }
    result["artifact"]["files"] = files
    result["artifact"]["total_bytes"] = total_bytes

    import math

    stensors = []
    shard_metadata = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="np") as st:
            metadata = st.metadata() if callable(st.metadata) else st.metadata
            if metadata is not None:
                shard_metadata[path.name] = dict(metadata)
            for key in st.keys():
                slice_ = st.get_slice(key)
                shape = list(slice_.get_shape())
                itemsize = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2,
                            "I8": 1, "U8": 1, "U32": 4, "F64": 8}[slice_.get_dtype()]
                stensors.append({"shard": path.name, "key": key,
                                 "dtype": slice_.get_dtype(),
                                 "shape": shape,
                                 "bytes": math.prod(shape) * itemsize})
    stensors.sort(key=lambda e: (e["shard"], e["key"]))
    by_prefix: dict[str, int] = {}
    by_dtype: dict[str, int] = {}
    for entry in stensors:
        prefix = entry["key"].split(".")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
        by_dtype[entry["dtype"]] = by_dtype.get(entry["dtype"], 0) + 1
    result["artifact"]["tensors"] = {
        "count": len(stensors),
        "by_prefix": by_prefix,
        "by_dtype": by_dtype,
        "storage_dtypes": sorted({e["dtype"] for e in stensors}),
        "shard_metadata": shard_metadata,
        "entries": stensors,
    }

    # ------------------------------------------------------------------ configs
    config = json.loads((directory / "config.json").read_text())
    preprocessor = json.loads((directory / "preprocessor_config.json").read_text())
    video_preprocessor = json.loads((directory / "video_preprocessor_config.json").read_text())
    generation = json.loads((directory / "generation_config.json").read_text())
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
    jinja_template = (directory / "chat_template.jinja").read_text(encoding="utf-8")
    json_template_doc = json.loads((directory / "chat_template.json").read_text())
    readme = (directory / "README.md").read_text(encoding="utf-8")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    result["configs"] = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
        "quantization": config.get("quantization"),
        "vision_config": config.get("vision_config"),
        "text_config": config.get("text_config"),
        "image_token_id": config.get("image_token_id"),
        "video_token_id": config.get("video_token_id"),
        "vision_start_token_id": config.get("vision_start_token_id"),
        "vision_end_token_id": config.get("vision_end_token_id"),
        "preprocessor_config": preprocessor,
        "preprocessor_config_note": (
            "declared image processor is Qwen2VLImageProcessorFast with the "
            "same 65536/16777216 pixel budget as the 2B reference; the "
            "2026-10-08 decision (slow PIL Qwen2VLImageProcessor as "
            "documented deviation) applies to this artifact as well"
        ),
        "video_preprocessor_config": video_preprocessor,
        "generation_config": generation,
        "tokenizer": {
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "add_bos_token": tokenizer_config.get("add_bos_token"),
            "chat_template_in_config_sha256": sha256_bytes(
                tokenizer_config["chat_template"].encode("utf-8"))
            if "chat_template" in tokenizer_config else None,
            "chat_template_jinja_sha256": sha256_bytes(jinja_template.encode("utf-8")),
            "chat_template_json_matches_jinja":
                json_template_doc.get("chat_template") == jinja_template,
            "chat_template_identical_to_2b_reference":
                sha256_bytes(jinja_template.encode("utf-8"))
                == "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
        },
        "index": {
            "metadata": index.get("metadata"),
            "weight_map_entries": len(index.get("weight_map", {})),
        },
        "converter_provenance": {
            "readme_sha256": sha256_bytes(readme.encode("utf-8")),
            "readme": readme,
            "note": ("converter identity/version per the artifact README "
                     "(mlx-vlm 0.3.4, from Qwen/Qwen3-VL-8B-Instruct); the "
                     "runtime probe runs a different, reviewed mlx-vlm "
                     "release (0.3.9) - see environment"),
        },
    }

    # --------------------------------------------------------------- prompt ids
    from transformers import Qwen2TokenizerFast

    tokenizer = Qwen2TokenizerFast.from_pretrained(str(directory))
    rendered = tokenizer.apply_chat_template(PROMPT_MESSAGES,
                                             add_generation_prompt=True,
                                             tokenize=False)
    rendered_sha = sha256_bytes(rendered.encode("utf-8"))
    if rendered_sha != EXPECTED_RENDERED_PROMPT_SHA256:
        sys.exit(f"rendered prompt sha {rendered_sha} does not match the "
                 f"2B reference {EXPECTED_RENDERED_PROMPT_SHA256}")
    base_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if base_ids.count(IMAGE_TOKEN_ID) != 1:
        sys.exit(f"expected exactly one image token, found "
                 f"{base_ids.count(IMAGE_TOKEN_ID)}")
    result["prompt"] = {
        "rendered": rendered,
        "rendered_sha256": rendered_sha,
        "rendered_matches_2b_reference": True,
        "unexpanded_ids": list(base_ids),
        "unexpanded_length": len(base_ids),
        "image_token_position": base_ids.index(IMAGE_TOKEN_ID),
        "expansion_rule": "single image token replaced by prod(grid)//merge**2 copies",
    }

    fast = verify_exports(args.patches_dir)
    result["fast_comparison"] = {
        "note": ("pixel_values come from the lossless slow_compensated exports "
                 "of the fast-comparison runner; the HF-reference runtime fed "
                 "these same buffers through the 2B checkpoint"),
        "manifest_sha256": fast["manifest_sha256"],
        "lock_sha256": fast["lock_sha256"],
        "environment": fast["environment"],
    }

    # -------------------------------------------------------------------- model
    model_info: dict = {"no_parity_claim": (
        "this runtime (mlx-vlm 0.3.9 on Metal, 4-bit affine quantized 8B "
        "checkpoint) is a different runtime and checkpoint than the pinned "
        "HF CPU float32 reference; the recorded facts are compatibility and "
        "provenance evidence only, not a parity claim")}
    if not args.skip_model:
        import mlx_vlm.models.cache as mlx_cache
        from mlx_vlm.utils import load_model

        # mlx_vlm.load would also build the AutoProcessor, which requires
        # the video-processor backend (torchvision) that this venv
        # deliberately does not carry; the probe never runs an image
        # processor (patches are injected) or the processor's chat-template
        # path (the tokenizer is used directly), so load_model alone is the
        # exact surface it exercises. The AutoProcessor failure is recorded
        # below as a compatibility finding (same call load_processor makes)
        # using deterministic fields only (exception class + sorted missing
        # framework names), never the raw message: transformers assembles
        # its missing-library paragraphs in a process-dependent order.
        try:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(str(directory))
            processor_probe = {"auto_processor_loaded": True, "error": None}
        except Exception as exc:  # noqa: BLE001 - the probe records the failure
            missing = parse_missing_frameworks(str(exc))
            processor_probe = {"auto_processor_loaded": False,
                               "error": type(exc).__name__,
                               "missing_frameworks": missing}

        model = load_model(directory)

        vision = model.vision_tower
        language = model.language_model
        lm = language.model
        layer0 = lm.layers[0]
        model_info["loading"] = {
            "lazy": False,
            "loader": "mlx_vlm.utils.load_model (no AutoProcessor)",
            "auto_processor_probe": processor_probe,
        }
        model_info["tied_embedding"] = {
            "config_tie_word_embeddings": config.get("tie_word_embeddings"),
            "lm_head_is_embed_tokens":
                language.lm_head.weight is lm.embed_tokens.weight,
        }

        def iter_arrays(module):
            # parameters() returns a nested dict of dicts/lists of arrays.
            def rec(value):
                if isinstance(value, mx.array):
                    yield value
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        yield from rec(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from rec(item)

            yield from rec(module.parameters())

        def params_of(module) -> dict:
            # U32 weights are 8-packed int4 (4-bit, group 64): count storage
            # elements separately from effective (unpacked) parameters.
            float_elements = 0
            packed_u32_elements = 0
            for array in iter_arrays(module):
                if array.dtype == mx.uint32:
                    packed_u32_elements += array.size
                else:
                    float_elements += array.size
            return {
                "float_elements": float_elements,
                "packed_uint32_elements": packed_u32_elements,
                "effective_parameters":
                    float_elements + packed_u32_elements * 8,
            }

        model_info["parameter_counts"] = {
            "vision_tower": params_of(vision),
            "language_model": params_of(language),
        }

        def guarded(label: str, factory) -> dict:
            try:
                return leaf_info(factory())
            except Exception as exc:  # noqa: BLE001 - record, do not fail
                return {"class": None, "error": f"{type(exc).__name__}: {exc}"}

        model_info["kernels"] = {
            "vision_tower": linear_kernels(vision),
            "language_model_model": linear_kernels(lm),
            "vision_module_histogram": module_histogram(vision),
            "language_module_histogram": module_histogram(language),
            "samples": {
                "language_embed_tokens": guarded("embed_tokens", lambda: lm.embed_tokens),
                "language_final_norm": guarded("final_norm", lambda: lm.norm),
                "layer0_input_layernorm": guarded("input_layernorm",
                                                  lambda: layer0.input_layernorm),
                "layer0_q_proj": guarded("q_proj", lambda: layer0.self_attn.q_proj),
                "layer0_k_proj": guarded("k_proj", lambda: layer0.self_attn.k_proj),
                "layer0_v_proj": guarded("v_proj", lambda: layer0.self_attn.v_proj),
                "layer0_o_proj": guarded("o_proj", lambda: layer0.self_attn.o_proj),
                "layer0_q_norm": guarded("q_norm", lambda: layer0.self_attn.q_norm),
                "layer0_k_norm": guarded("k_norm", lambda: layer0.self_attn.k_norm),
                "layer0_gate_proj": guarded("gate_proj", lambda: layer0.mlp.gate_proj),
                "layer0_up_proj": guarded("up_proj", lambda: layer0.mlp.up_proj),
                "layer0_down_proj": guarded("down_proj", lambda: layer0.mlp.down_proj),
                "lm_head": guarded("lm_head", lambda: language.lm_head),
                "vision_patch_embed_proj":
                    guarded("patch_embed", lambda: vision.patch_embed.proj),
                "vision_pos_embed": guarded("pos_embed", lambda: vision.pos_embed),
                "vision_block0_qkv": guarded("block0_qkv",
                                             lambda: vision.blocks[0].attn.qkv),
                "vision_merger_fc2": guarded("merger_fc2",
                                             lambda: vision.merger.linear_fc2),
            },
        }
        model_info["compute_dtypes"] = {
            "vision_pixel_values_cast_to":
                mx_dtype(vision.patch_embed.proj.weight.dtype),
            "language_layer0_q_proj_weight": mx_dtype(layer0.self_attn.q_proj.weight.dtype),
            "vision_block0_qkv_weight": mx_dtype(vision.blocks[0].attn.qkv.weight.dtype),
        }

        import numpy as np

        cases = {}
        for image_name in MODEL_IMAGES:
            expected = HF_REFERENCE[image_name]
            exp = fast["exports"][image_name]
            grid_ok = check_facts(
                image_name, "grid_thw", exp["grid_thw"],
                expected["grid_thw"], "grid mismatch vs HF reference")
            pixel_values = np.load(args.patches_dir / exp["export"])
            grid = exp["grid_thw"]
            n_merged = grid[0] * grid[1] * grid[2] // 4
            image_pos = base_ids.index(IMAGE_TOKEN_ID)
            ids = base_ids[:image_pos] + [IMAGE_TOKEN_ID] * n_merged \
                + base_ids[image_pos + 1:]
            length_ok = check_facts(
                image_name, "prompt_length_after_expansion",
                len(ids), expected["prompt_length_after_expansion"])
            span = [image_pos, image_pos + n_merged - 1]
            span_ok = check_facts(image_name, "image_token_span", span,
                                  expected["image_token_span"])

            ids_2d = mx.array([ids])
            mask_2d = mx.array([[1] * len(ids)])
            pv = mx.array(pixel_values.astype(np.float32))
            grid_arr = mx.array([grid])

            cache = mlx_cache.make_prompt_cache(model.language_model)
            out = model(ids_2d, pixel_values=pv, mask=mask_2d, cache=cache,
                        image_grid_thw=grid_arr)
            mx.eval(out.logits)

            last = out.logits[0, -1]
            last_f32 = last.astype(mx.float32)
            logits_np = np.asarray(last_f32, dtype=np.float32)
            prefill = {
                "logits_dtype": mx_dtype(last.dtype),
                "logits_shape": list(out.logits.shape),
                "last_position_sha256": sha256_bytes(logits_np.tobytes()),
                "argmax": int(mx.argmax(last).item()),
                "top20": top20(last),
            }

            # mlx-lm 0.29.1 KVCache preallocates capacity in 256-token
            # chunks; the used length is cache.offset, not keys.shape[2].
            keys0 = cache[0].keys
            kv_after_prefill = {
                "keys_dtype": mx_dtype(keys0.dtype),
                "keys_capacity_shape": list(keys0.shape),
                "values_dtype": mx_dtype(cache[0].values.dtype),
                "used_length": int(cache[0].offset),
            }

            # Determinism: a fresh cache + fresh prefill must reproduce
            # the last-position logits bit-exactly on this runtime.
            cache_repeat = mlx_cache.make_prompt_cache(model.language_model)
            out_repeat = model(ids_2d, pixel_values=pv, mask=mask_2d,
                               cache=cache_repeat, image_grid_thw=grid_arr)
            mx.eval(out_repeat.logits)
            repeat_f32 = np.asarray(out_repeat.logits[0, -1].astype(mx.float32),
                                    dtype=np.float32)
            determinism_match = sha256_bytes(repeat_f32.tobytes()) \
                == prefill["last_position_sha256"]

            deltas = language._rope_deltas
            delta = int(deltas[0].item() if deltas.ndim else deltas.item())
            delta_match = delta == expected["rope_delta"]
            if not delta_match:
                sys.exit(f"{image_name}: MRoPE delta {delta} != HF reference "
                         f"{expected['rope_delta']}")

            current = mx.array([[prefill["argmax"]]])
            greedy = [prefill["argmax"]]
            decode_steps = []
            for step in range(4):
                out_step = model.language_model(current, cache=cache)
                mx.eval(out_step.logits)
                step_logits = out_step.logits[0, -1].astype(mx.float32)
                step_np = np.asarray(step_logits, dtype=np.float32)
                nxt = int(mx.argmax(step_logits).item())
                greedy.append(nxt)
                decode_steps.append({
                    "step": step,
                    "cache_position": len(ids) + step,
                    "kv_used_length": int(cache[0].offset),
                    "logits_dtype": mx_dtype(out_step.logits[0, -1].dtype),
                    "logits_sha256": sha256_bytes(step_np.tobytes()),
                    "argmax": nxt,
                })
                current = mx.array([[nxt]])

            cases[image_name] = {
                "export": exp,
                "sequence": {
                    "length": len(ids),
                    "image_token_span": span,
                    "image_token_count": n_merged,
                    "matches_hf_reference":
                        grid_ok and length_ok and span_ok and delta_match,
                    "hf_reference_greedy_ids_2b":
                        expected["greedy_ids_2b"],
                },
                "grid": grid,
                "rope": {
                    "delta_mlx": delta,
                    "delta_hf_reference": expected["rope_delta"],
                    "delta_matches_hf_reference": delta_match,
                },
                "kv_after_prefill": kv_after_prefill,
                "prefill": prefill,
                "determinism": {"repeat_prefill_sha256_match": determinism_match},
                "decode": decode_steps,
                "greedy_ids": greedy,
            }
        model_info["cases"] = cases
    else:
        model_info["skipped"] = True
    result["model"] = model_info

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")
    for name, case in result["model"].get("cases", {}).items():
        print(f"{name}: grid={case['grid']} delta={case['rope']['delta_mlx']} "
              f"greedy={case['greedy_ids']} "
              f"determinism={case['determinism']['repeat_prefill_sha256_match']}")


if __name__ == "__main__":
    main()
