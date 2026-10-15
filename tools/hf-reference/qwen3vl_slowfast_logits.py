#!/usr/bin/env python3
"""Phase 7.3b WP1 slow-vs-fast model-side comparison.

Loads the lossless patch exports written by the fast-comparison runner
(``tools/image-oracle/fast-comparison/runner.py``) and feeds BOTH the
slow-compensated and the fast image processor's ``pixel_values`` through the
SAME HF-reference runtime (pinned checkpoint, torch 2.6.0, float32, eager,
single-threaded, deterministic): prefill plus cached decode with argmax
selection. This isolates preprocessing divergence: the model, weights and
stepping are identical, only the input patch buffers differ. The comparison
runtime's own lock digest and exact Torch version are carried in the
fast-comparison manifest and recorded here.

It reuses ``qwen3vl_probe``'s environment gate, chat prompt, processor
construction (slow compensated image processor + the ``ProbeProcessor``
video-processor workaround) and rope re-derivation. It never re-runs the image
processors for pixel values: the patch buffers come from the runner's lossless
exports, whose SHA-256s are re-verified against the runner manifest before
use. The fixture images it tokenizes come from the committed fast-comparison
fixtures directory (``--fixtures``, default
``tools/image-oracle/fast-comparison/fixtures``) and are digest-verified
against the manifest's per-file fixtures digests before any processor or model
work; after the processor call that builds ``input_ids``, the processor's own
``image_grid_thw`` is verified against the exported ``grid_thw`` of the
slow_compensated config, and the slow and fast export grids are verified to
agree with each other.

Usage (from the repository root):

    tools/hf-reference/.venv/bin/python tools/hf-reference/qwen3vl_slowfast_logits.py \
        --directory .tier-b/qwen3-vl-2b-instruct \
        --patches-dir <fast-comparison output dir> \
        --out tools/hf-reference/goldens/qwen3-vl-2b-wp1-slow-fast-logits.json
        # optional: --fixtures <dir> (defaults to the committed
        # tools/image-oracle/fast-comparison/fixtures)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import qwen3vl_probe as probe

SCHEMA = "jmlx-phase7-3b-wp1-qwen3vl-slowfast-v1"
CONFIGS = ("slow_compensated", "fast")
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


def load_manifest(patches_dir: Path) -> tuple[dict, Path]:
    """Load the runner manifest, pinned to the fast-comparison environment.

    Checks the schema, then the manifest's own SHA-256 and its
    ``lock_sha256`` field against the pinned digests above, before any export
    is read or any model code runs.
    """
    manifest_path = patches_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"fast-comparison manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "jmlx-phase7-3b-wp1-qwen3vl-fastcomparison-v1":
        raise SystemExit("unexpected fast-comparison manifest schema")
    manifest_digest = probe.sha256_bytes(manifest_path.read_bytes())
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
    return manifest, manifest_path


def verify_exports(patches_dir: Path, manifest: dict) -> dict:
    exports = {}
    by_file = {rec["file"]: rec for rec in manifest["images"]}
    for image_name in MODEL_IMAGES:
        record = by_file[image_name]
        per_config = {}
        for name in CONFIGS:
            cfg = record["configs"][name]
            export_name = cfg["lossless_export"]
            if not export_name:
                raise SystemExit(f"missing lossless export for {image_name}/{name}")
            array = np.load(patches_dir / export_name)
            digest = probe.sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())
            if digest != cfg["pixel_values_sha256"]:
                raise SystemExit(
                    f"export {export_name} no longer matches the runner manifest")
            per_config[name] = {
                "export": export_name,
                "grid_thw": cfg["grid_thw"],
                "dtype": cfg["dtype"],
                "shape": cfg["shape"],
                "pixel_values_sha256": digest,
                "manifest_uint8_patches_sha256": cfg["uint8_patches_sha256"],
            }
        exports[image_name] = per_config
    return exports


def resolve_fixture(fixtures_dir: Path, image_name: str, manifest: dict) -> Path:
    """Resolve a committed fixture image, verified against the pinned digest.

    The digest comes from the manifest's per-file ``fixtures`` section when
    present (the manifest is itself digest-pinned by ``load_manifest``, so the
    chain is complete), otherwise from the committed
    fast-comparison/provenance.json. A missing file or a digest mismatch
    exits before any processor or model work.
    """
    path = fixtures_dir / image_name
    if not path.is_file():
        raise SystemExit(f"fast-comparison fixture is missing: {path}")
    pinned = manifest.get("fixtures", {}).get(image_name)
    if pinned is None:
        provenance = json.loads(
            (Path(__file__).resolve().parents[1] / "image-oracle" /
             "fast-comparison" / "provenance.json").read_text(encoding="utf-8"))
        pinned = provenance["fixtures"].get(image_name)
        if pinned is None:
            raise SystemExit(
                f"no pinned digest for {image_name} in the manifest's "
                "fixtures section or in fast-comparison/provenance.json")
    digest = probe.sha256_bytes(path.read_bytes())
    if digest != pinned:
        raise SystemExit(
            f"fast-comparison fixture {path} sha256 {digest} does not match "
            f"the pinned {pinned}")
    return path


def verify_grids(processor, rendered: str, image, image_name: str,
                 exports: dict) -> dict:
    """Run the shared processor call and gate on the exported grids.

    One processor call supplies input_ids/attention_mask for BOTH patch sets,
    so before any model forward: the processor's ``image_grid_thw`` must equal
    the slow_compensated export's ``grid_thw``, and the slow and fast export
    grids must agree with each other.
    """
    batch = processor(images=[image.convert("RGB")], text=[rendered])
    processor_grid = [[int(v) for v in row] for row in batch["image_grid_thw"]]
    slow_grid = exports[image_name]["slow_compensated"]["grid_thw"]
    fast_grid = exports[image_name]["fast"]["grid_thw"]
    if processor_grid != [slow_grid]:
        sys.exit(
            f"{image_name}: processor image_grid_thw {processor_grid} does not "
            f"match the slow_compensated export grid_thw {slow_grid}; the "
            "input_ids expansion and the exported patch buffer are "
            "inconsistent")
    if slow_grid != fast_grid:
        sys.exit(
            f"{image_name}: slow_compensated export grid_thw {slow_grid} does "
            f"not match fast export grid_thw {fast_grid}; the shared input_ids "
            "cannot be valid for both patch sets")
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True,
                        help="pinned Qwen3-VL-2B-Instruct snapshot")
    parser.add_argument("--patches-dir", type=Path, required=True,
                        help="fast-comparison runner output directory")
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "image-oracle" / "fast-comparison" / "fixtures",
                        help="committed fast-comparison fixture images "
                             "(default: tools/image-oracle/fast-comparison/fixtures)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    environment = probe.gate_versions()

    import torch

    # Same execution mode as the reference probe: one CPU thread and the
    # deterministic-algorithms flag, so the prefill/decode logits are
    # bit-reproducible process over process (the probe's 256x256 prefill
    # SHA-256 is the cross-check for this).
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    manifest, manifest_path = load_manifest(args.patches_dir)
    exports = verify_exports(args.patches_dir, manifest)
    fixture_paths = {
        name: resolve_fixture(args.fixtures, name, manifest)
        for name in MODEL_IMAGES
    }

    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from PIL import Image

    preprocessor = json.loads(
        (args.directory / "preprocessor_config.json").read_text(encoding="utf-8"))
    declared_min = int(preprocessor["size"]["shortest_edge"])
    declared_max = int(preprocessor["size"]["longest_edge"])
    img_proc = Qwen2VLImageProcessor.from_pretrained(
        str(args.directory), min_pixels=declared_min, max_pixels=declared_max)
    tokenizer_config = json.loads(
        (args.directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(args.directory))
    video_proc = Qwen3VLVideoProcessor.from_pretrained(str(args.directory))

    class ProbeProcessor(Qwen3VLProcessor):
        def check_argument_for_proper_class(self, name, arg):
            if name == "video_processor":
                return Qwen3VLVideoProcessor
            return super().check_argument_for_proper_class(name, arg)

    processor = ProbeProcessor(image_processor=img_proc, tokenizer=tokenizer,
                               video_processor=video_proc,
                               chat_template=tokenizer_config["chat_template"])
    rendered = processor.apply_chat_template(PROMPT_MESSAGES, add_generation_prompt=True,
                                             tokenize=False)

    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.directory), dtype=torch.float32, attn_implementation="eager")
    model.eval()
    llm = model.model

    result: dict = {
        "schema": SCHEMA,
        "environment": environment,
        "comparison_runtime": {
            "note": "the patch sets were produced in the fast-comparison venv; this "
            "runtime feeds both through one model in the HF-reference venv",
            "fast_comparison_environment": manifest["environment"],
            "fast_comparison_lock_sha256": manifest["lock_sha256"],
            "fast_comparison_manifest_sha256":
                probe.sha256_bytes(manifest_path.read_bytes()),
        },
        "exports": exports,
        "cases": {},
    }

    with torch.no_grad():
        for image_name in MODEL_IMAGES:
            per_config_exports = exports[image_name]
            # One processor call supplies input_ids/attention_mask for both
            # patch sets; verify_grids gates that the processor grid matches
            # the exported grid_thw for both configs before any forward.
            image = Image.open(fixture_paths[image_name])
            batch = verify_grids(processor, rendered, image, image_name, exports)
            input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
            attention_mask = torch.as_tensor(batch["attention_mask"], dtype=torch.long)
            ids_row = [int(v) for v in input_ids[0].tolist()]
            span_start, span_end = probe.image_token_span(ids_row)

            case = {
                "sequence": {
                    "length": len(ids_row),
                    "image_token_span": [span_start, span_end],
                },
                "configs": {},
            }
            logits_by_config = {}
            for name in CONFIGS:
                exp = per_config_exports[name]
                pixel_values = torch.as_tensor(
                    np.load(args.patches_dir / exp["export"]), dtype=torch.float32)
                image_grid_thw = torch.as_tensor([exp["grid_thw"]], dtype=torch.long)

                expected = probe.rederive_rope(ids_row, exp["grid_thw"])
                position_ids, rope_deltas = llm.get_rope_index(
                    input_ids, image_grid_thw, None, attention_mask=attention_mask)
                actual = [[int(v) for v in row] for row in position_ids[:, 0].tolist()]
                delta = int(rope_deltas[0][0].item())
                assert actual == expected["position_ids"], "position_ids mismatch"
                assert delta == expected["rope_delta"], "rope delta mismatch"

                model.model.rope_deltas = None
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw)
                assert torch.equal(model.model.rope_deltas, rope_deltas)
                last_logits = out.logits[0, -1].float().contiguous()
                first_step = {
                    "prefill_sha256": probe.sha256_bytes(last_logits.numpy().tobytes()),
                    "prefill_argmax": int(last_logits.argmax().item()),
                    "prefill_top20": [
                        [int(i), float(v)] for i, v in zip(*torch.topk(last_logits, 20))
                    ],
                    "logits_shape": list(out.logits.shape),
                }

                model.model.rope_deltas = None
                out_repeat = model(input_ids=input_ids, attention_mask=attention_mask,
                                   pixel_values=pixel_values, image_grid_thw=image_grid_thw)
                repeat_logits = out_repeat.logits[0, -1].float().contiguous()
                determinism_match = probe.sha256_bytes(repeat_logits.numpy().tobytes()) \
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
                        "expected_decode_positions_all_axes": cache_position + delta,
                        "logits_sha256": probe.sha256_bytes(step_logits.numpy().tobytes()),
                        "argmax": nxt,
                    })
                logits_by_config[name] = last_logits
                case["configs"][name] = {
                    "grid": exp["grid_thw"],
                    "rope_delta": delta,
                    "prefill": first_step,
                    "determinism": {"repeat_prefill_sha256_match": determinism_match},
                    "decode": decode_steps,
                    "greedy_ids": greedy,
                }

            slow_logits = logits_by_config["slow_compensated"]
            fast_logits = logits_by_config["fast"]
            diff = (slow_logits - fast_logits).abs()
            case["cross"] = {
                "prefill_logits_identical": bool(torch.equal(slow_logits, fast_logits)),
                "greedy_ids_identical":
                    case["configs"]["slow_compensated"]["greedy_ids"]
                    == case["configs"]["fast"]["greedy_ids"],
                "diff": {
                    "differing_entries": int((diff > 0).sum().item()),
                    "entries": slow_logits.numel(),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                },
            }
            result["cases"][image_name] = case

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(probe.py(result), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for name, case in result["cases"].items():
        cross = case["cross"]
        greedy_match = "match" if cross["greedy_ids_identical"] else "DIVERGE"
        print(f"{name}: logits_identical={cross['prefill_logits_identical']} "
              f"max_abs={cross['diff']['max_abs']:.6f} greedy {greedy_match}")


if __name__ == "__main__":
    main()
