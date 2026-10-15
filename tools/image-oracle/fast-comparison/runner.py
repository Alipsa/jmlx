#!/usr/bin/env python3
"""Slow-vs-fast Qwen2-VL image processor comparison (Phase 7.3b WP1, opt-in).

Runs three explicit processor configurations against the pinned representative
fixtures in ``fixtures/`` and records, per (configuration, image): source size,
resized size, grid, dtype, shape, the SHA-256 of the normalized float32 patch
buffer, and the SHA-256 of the pre-rescale uint8 patches (which splits resize
divergence from normalization divergence). The two model fixtures are additionally exported losslessly as ``.npy``
files so the HF-reference model runtime
(``tools/hf-reference/qwen3vl_slowfast_logits.py``) can feed both patch sets
through the same checkpoint in one runtime; the geometry probe runs only the
uint8 pass (its normalized buffer would be ~400 MB of float32 and has no
consumer), so its records carry ``pixel_values_sha256: null`` with a note.
This tool produces processed patches only; it never computes model logits.

The configurations:

* ``fast`` -- ``Qwen2VLImageProcessorFast.from_pretrained(fixtures)``. The class
  the artifact declares (``image_processor_type``); the committed
  ``preprocessor_config.json`` copy supplies the artifact-declared
  ``size`` (65536/16777216).
* ``slow_compensated`` -- ``Qwen2VLImageProcessor.from_pretrained(fixtures,
  min_pixels=65536, max_pixels=16777216)``. The documented jmlx preprocessing
  reference: the slow PIL processor made to honor the artifact-declared options
  explicitly (the 4.57.6 config-only defaults are wrong, see the WP1 findings).
* ``slow_default`` -- ``Qwen2VLImageProcessor.from_pretrained(fixtures)``. The
  4.57.6 config-only resolution, recorded as evidence of the slow processor's
  size-dict default bug (3136/1003520).

Usage (from the repository root):

    tools/image-oracle/fast-comparison/.venv/bin/python \
        tools/image-oracle/fast-comparison/runner.py --out <output-dir>
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

PINS = {
    "pillow": "12.3.0",
    "numpy": "2.5.3",
    "transformers": "4.57.6",
}
TORCH_VERSIONS = ("2.6.0", "2.6.0+cpu")
TORCHVISION_VERSIONS = ("0.21.0", "0.21.0+cpu")
SCHEMA = "jmlx-phase7-3b-wp1-qwen3vl-fastcomparison-v1"
# The model fixtures are exported losslessly; they are the model inputs. The
# geometry probe exceeds the 16777216-pixel budget (its resized patch buffer
# would be ~400 MB of float32), so its full-normalized pass is skipped, it is
# fingerprinted by grid/shape/uint8 SHA-256 only, and it is never exported.
LOSSLESS_EXPORT = ("model-rgb-256x256.png", "model-rgb-64x32.png")
CONFIG_ORDER = ("fast", "slow_compensated", "slow_default")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def gate() -> dict:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("fast comparison requires CPython 3.12")
    versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pillow": importlib.metadata.version("Pillow"),
        "numpy": importlib.metadata.version("numpy"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
    }
    for name, expected in PINS.items():
        if versions[name] != expected:
            raise SystemExit(f"fast comparison requires {name} {expected}, found {versions[name]}")
    if versions["torch"] not in TORCH_VERSIONS:
        raise SystemExit(
            f"fast comparison requires torch {'/'.join(TORCH_VERSIONS)}, "
            f"found {versions['torch']}"
        )
    if versions["torchvision"] not in TORCHVISION_VERSIONS:
        raise SystemExit(
            f"fast comparison requires torchvision {'/'.join(TORCHVISION_VERSIONS)}, "
            f"found {versions['torchvision']}"
        )
    import torch  # noqa: F401  -- proves the comparison runtime is importable
    import torchvision  # noqa: F401  -- same

    return versions


def verify_fixtures(fixtures: Path, provenance: dict) -> dict:
    recorded = provenance["fixtures"]
    actual = {}
    for name in sorted(recorded):
        path = fixtures / name
        if not path.is_file():
            raise SystemExit(f"fast comparison fixture is missing: {path}")
        digest = sha256_file(path)
        if digest != recorded[name]:
            raise SystemExit(f"fast comparison fixture {name} no longer matches provenance")
        actual[name] = digest
    if set(actual) != {p.name for p in fixtures.iterdir() if p.is_file()}:
        raise SystemExit("fast comparison fixtures directory carries unpinned files")
    return actual


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=script_dir / "fixtures")
    parser.add_argument("--provenance", type=Path, default=script_dir / "provenance.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    versions = gate()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    fixture_digests = verify_fixtures(args.fixtures, provenance)
    lock_path = script_dir / "requirements.lock"

    import numpy as np
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
        Qwen2VLImageProcessorFast,
    )

    processors = {
        "fast": Qwen2VLImageProcessorFast.from_pretrained(str(args.fixtures)),
        "slow_compensated": Qwen2VLImageProcessor.from_pretrained(
            str(args.fixtures), min_pixels=65536, max_pixels=16777216
        ),
        "slow_default": Qwen2VLImageProcessor.from_pretrained(str(args.fixtures)),
    }
    def record_processor(name, construction, note):
        p = processors[name]
        return {
            "class": p.__class__.__name__,
            "construction": construction,
            "note": note,
            "min_pixels": None if p.min_pixels is None else int(p.min_pixels),
            "max_pixels": None if p.max_pixels is None else int(p.max_pixels),
            "size": p.size,
        }

    processor_records = {
        "fast": record_processor(
            "fast",
            "Qwen2VLImageProcessorFast.from_pretrained(fixtures)",
            "the artifact-declared image_processor_type; it keeps the declared size dict "
            "(min/max pixels stay None and are not used)",
        ),
        "slow_compensated": record_processor(
            "slow_compensated",
            "Qwen2VLImageProcessor.from_pretrained(fixtures, "
            "min_pixels=65536, max_pixels=16777216)",
            "documented jmlx preprocessing reference: the slow PIL processor honors the "
            "artifact-declared size via explicit options (the effective knobs for the slow "
            "processor are min_pixels/max_pixels, not the retained size dict)",
        ),
        "slow_default": record_processor(
            "slow_default",
            "Qwen2VLImageProcessor.from_pretrained(fixtures)",
            "the 4.57.6 config-only resolution, recorded as evidence of the slow "
            "processor's size-dict default bug (3136/1003520)",
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    image_records = []
    for path in sorted(p for p in args.fixtures.iterdir() if p.suffix == ".png"):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        source = {"width": image.width, "height": image.height}
        configs = {}
        for name in CONFIG_ORDER:
            # The pre-rescale uint8 patch pixels (no rescale, no normalization), to split
            # resize divergence from normalization divergence. The raw pass also carries
            # the grid, so the geometry probe never needs a second pass.
            raw_result = processors[name](
                images=[image], do_rescale=False, do_normalize=False, return_tensors="pt"
            )
            raw_patches = np.ascontiguousarray(
                raw_result["pixel_values"].detach().cpu().numpy(), dtype=np.uint8
            )
            grid = [int(v) for v in raw_result["image_grid_thw"][0].tolist()]
            if path.name in LOSSLESS_EXPORT:
                # The full-normalized float32 pass, for the model fixtures only. The fast
                # processor only returns PyTorch tensors; use "pt" for all three so the
                # buffers are compared on identical footing, then reduce to float32.
                result = processors[name](images=[image], return_tensors="pt")
                pixel_values = np.ascontiguousarray(
                    result["pixel_values"].detach().cpu().numpy(), dtype=np.float32
                )
                export_name = f"{name}__{path.stem}.npy"
                np.save(args.out / export_name, pixel_values)
                configs[name] = {
                    "resized": {"width": grid[2] * 16, "height": grid[1] * 16},
                    "grid_thw": grid,
                    "dtype": pixel_values.dtype.name,
                    "shape": list(pixel_values.shape),
                    "pixel_values_sha256": sha256_bytes(pixel_values.tobytes()),
                    "uint8_dtype": raw_patches.dtype.name,
                    "uint8_shape": list(raw_patches.shape),
                    "uint8_patches_sha256": sha256_bytes(raw_patches.tobytes()),
                    "lossless_export": export_name,
                }
            else:
                # The geometry-only fixture: the full-normalized pass would materialize a
                # ~400 MB float32 buffer nothing consumes, so it is skipped and the record
                # is derived from the raw pass alone.
                configs[name] = {
                    "resized": {"width": grid[2] * 16, "height": grid[1] * 16},
                    "grid_thw": grid,
                    "dtype": raw_patches.dtype.name,
                    "shape": list(raw_patches.shape),
                    "pixel_values_sha256": None,
                    "note": "normalized pass skipped (geometry-only fixture, ~400 MB "
                            "float32); fingerprinted by grid/shape/uint8 SHA-256 only",
                    "uint8_dtype": raw_patches.dtype.name,
                    "uint8_shape": list(raw_patches.shape),
                    "uint8_patches_sha256": sha256_bytes(raw_patches.tobytes()),
                    "lossless_export": None,
                }
        image_records.append(
            {
                "file": path.name,
                "sha256": fixture_digests[path.name],
                "source": source,
                "configs": configs,
            }
        )

    comparison = {}
    for record in image_records:
        if record["file"] not in LOSSLESS_EXPORT:
            continue
        fast = record["configs"]["fast"]
        compensated = record["configs"]["slow_compensated"]
        default = record["configs"]["slow_default"]
        comparison[record["file"]] = {
            "slow_compensated_vs_fast_uint8_identical":
                fast["uint8_patches_sha256"] == compensated["uint8_patches_sha256"],
            "slow_compensated_vs_fast_pixel_values_identical":
                fast["pixel_values_sha256"] == compensated["pixel_values_sha256"],
            "slow_default_grid_differs_from_fast": default["grid_thw"] != fast["grid_thw"],
        }

    manifest = {
        "schema": SCHEMA,
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            **versions,
        },
        "lock_sha256": sha256_file(lock_path),
        "fixtures": fixture_digests,
        "processors": processor_records,
        "images": image_records,
        "comparison": comparison,
    }
    out_path = args.out / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    for name in CONFIG_ORDER:
        rec = processor_records[name]
        print(f"{name}: min_pixels={rec['min_pixels']} max_pixels={rec['max_pixels']}")
    for key, flags in comparison.items():
        print(f"{key}: {flags}")


if __name__ == "__main__":
    main()
