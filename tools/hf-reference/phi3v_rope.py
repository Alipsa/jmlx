#!/usr/bin/env python3
"""Phase 7.3c WP4 reference generator: the NTK-Su RoPE golden for Phi-3.5-Vision.

Runs in the hash-locked tools/hf-reference venv (Torch 2.6.0 CPU, transformers 4.57.6,
fully offline). Usage:

    .venv/bin/python phi3v_rope.py --directory <artifact-files> --out goldens/phi3v-rope.json

<artifact-files> must contain at least the pinned config.json, configuration_phi3_v.py and
modeling_phi3_v.py of microsoft/Phi-3.5-vision-instruct (the WP0 file set). The script
verifies their SHA-256 before importing or using the bundled modules, and imports the
bundled modules as a synthetic package so that the exact digest-pinned files are executed.
transformers 4.57.6 has no native phi3_v, so the bundled Phi3SuScaledRotaryEmbedding is
the reference (plan req/plans/phase7-3c-plan.md, WP4; findings section 8).

The golden has two sections at the artifact's head_dim (96), theta (10000) and the
short/long boundary (4096/4097):
  tables   the inv_freq the bundled embedding selects for the kv lengths 128, 4096, 4097
           and 131072 (the short/long boundary and the max context) -- this is what
           RopeSpec.Su.frequencies(96, L) must reproduce
  samples  single-position rotations through the bundled apply_rotary_pos_emb at offsets
           0, 17, 4095, 4096 and 131071 (kv length offset + 1, so 4095 is the last
           short position and 4096 the first long one). Each sample carries a tolerance
           derived from the float32 ulp of its largest position angle, because both the
           reference (a float32 matmul of position and inv_freq) and the MLX kernel round
           the angle to float32 precision; at offset 0 this degenerates to the fixed
           2e-4 floor.
"""

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
import types
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch

REVISION = "12b77fb40b63a2c73c68243d3f767aab688a1b2a"
PINS = {
    "config.json": "567e28c638a3e41aba1a304ae58b283845ad07e1c60e970643b0ad8fc229b382",
    "configuration_phi3_v.py": "18ff44c5c1f9a6523184fb2e580e03ebf722a65f37b9cc57cd1a6d620a0f2666",
    "modeling_phi3_v.py": "b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c",
}

TABLE_KV_LENGTHS = (128, 4096, 4097, 131072)
SAMPLE_OFFSETS = (0, 17, 4095, 4096, 131071)
ROTATED_FLOOR = 2e-4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_pins(directory: Path) -> dict:
    result = {}
    for name, expected in sorted(PINS.items()):
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"missing pinned file: {path}")
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            raise SystemExit(f"pinned file mismatch: {name}: {actual} != {expected}")
        result[name] = actual
    return result


def load_bundled_modeling(directory: Path):
    """Imports the pinned bundled modules as a synthetic package (exact-file execution)."""
    pkg_name = "phi3v_bundled"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(directory)]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_phi3_v", directory / "modeling_phi3_v.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[f"{pkg_name}.modeling_phi3_v"] = module
    spec.loader.exec_module(module)
    return module


def rounded(value):
    if isinstance(value, torch.Tensor):
        return rounded(value.detach().cpu().tolist())
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.7g}")
    return value


def ulp32(value: float) -> float:
    if value == 0.0:
        return 2.0**-149
    return math.ldexp(1.0, int(math.floor(math.log2(value))) - 23)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True, help="pinned artifact files")
    parser.add_argument("--out", type=Path, required=True, help="output golden path")
    args = parser.parse_args()

    file_hashes = verify_pins(args.directory)
    config_dict = json.loads((args.directory / "config.json").read_text())
    modeling = load_bundled_modeling(args.directory)
    config_cls = sys.modules["phi3v_bundled.configuration_phi3_v"].Phi3VConfig
    config = config_cls(**config_dict)

    head_dim = config.hidden_size // config.num_attention_heads
    original_max = config.original_max_position_embeddings
    max_positions = config.max_position_embeddings
    rope = modeling.Phi3SuScaledRotaryEmbedding(head_dim, config)
    x_dummy = torch.zeros(1, 1, 1, head_dim, dtype=torch.float32)

    # The static cos/sin multiplier: the reference computes it in float64 from the config
    # and casts it to float32 when it multiplies cos and sin. The position-0 cos of an
    # executed forward must equal it, since cos(0) * scaling = scaling.
    scale = max_positions / original_max
    if scale <= 1.0:
        scaling = 1.0
    else:
        scaling = math.sqrt(1 + math.log(scale) / math.log(original_max))
    cos0, _ = rope(x_dummy, torch.zeros(1, 1, dtype=torch.long), seq_len=1)
    observed = float(cos0.flatten()[0])
    if abs(observed - scaling) > ulp32(scaling):
        raise SystemExit(
            f"position-0 cos {observed} disagrees with the scaling formula {scaling}"
        )

    short_factors = [float(f) for f in config.rope_scaling["short_factor"]]
    long_factors = [float(f) for f in config.rope_scaling["long_factor"]]
    if len(short_factors) != head_dim // 2 or len(long_factors) != head_dim // 2:
        raise SystemExit("rope_scaling factor length does not match head_dim // 2")

    tables = []
    for kv_length in TABLE_KV_LENGTHS:
        cos, _ = rope(x_dummy, torch.zeros(1, 1, dtype=torch.long), seq_len=kv_length)
        inv_freq = rope.inv_freq.detach().float().cpu()
        tables.append(
            {
                "seq_len": kv_length,
                "table": "long" if kv_length > original_max else "short",
                "inv_freq": rounded(inv_freq.tolist()),
            }
        )

    samples = []
    for offset in SAMPLE_OFFSETS:
        position = torch.tensor([[offset]], dtype=torch.long)
        fixed = (torch.arange(head_dim, dtype=torch.float32) / head_dim).reshape(
            1, 1, 1, head_dim
        )
        cos, sin = rope(fixed, position)
        inv_freq = rope.inv_freq.detach().float().cpu()
        rotated = modeling.apply_rotary_pos_emb(fixed, fixed, cos, sin)[0]
        # Largest |angle| over the rotary pairs; both the reference (float32 matmul) and
        # the MLX kernel round it to float32, so the tolerance follows its ulp.
        max_angle = max(float(offset * f) for f in inv_freq.tolist())
        samples.append(
            {
                "offset": offset,
                "seq_len": offset + 1,
                "table": "long" if offset + 1 > original_max else "short",
                "inv_freq": rounded(inv_freq.tolist()),
                "input": rounded(fixed.flatten().tolist()),
                "rotated": rounded(rotated.flatten().tolist()),
                "tolerance": max(ROTATED_FLOOR, 4.0 * ulp32(max_angle)),
            }
        )

    result = {
        "schema": "jmlx-phase7-3c-wp4-phi3v-rope-v1",
        "generated_by": "tools/hf-reference/phi3v_rope.py",
        "environment": {
            "python": platform.python_version(),
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
        },
        "artifact": {
            "org": "microsoft",
            "repo": "Phi-3.5-vision-instruct",
            "revision": REVISION,
            "file_sha256": file_hashes,
        },
        "head_dim": head_dim,
        "theta": float(config.rope_theta),
        "max_positions": max_positions,
        "original_max_positions": original_max,
        "attention_scaling": rounded(scaling),
        "short_factor": rounded(short_factors),
        "long_factor": rounded(long_factors),
        "tables": tables,
        "samples": samples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    json.dump(result, buffer, indent=2)
    buffer.write("\n")
    args.out.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"wrote {args.out} (sha256 {sha256_bytes(buffer.getvalue().encode())})")
    for table in tables:
        print(f"table seq_len={table['seq_len']}: {table['table']}")
    for sample in samples:
        print(
            f"sample offset={sample['offset']}: {sample['table']} "
            f"tolerance={sample['tolerance']:.3g}"
        )


if __name__ == "__main__":
    main()
