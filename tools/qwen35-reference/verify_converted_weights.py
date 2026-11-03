#!/usr/bin/env python3
"""Compare actual saved float converter outputs to the source, without generating layout files."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def load(root):
    result = {}
    for path in sorted(root.glob("*.safetensors")):
        for name, value in load_file(path).items():
            if name in result:
                raise ValueError(f"duplicate tensor: {name}")
            result[name] = value
    return result


def expected_name(name):
    if name.startswith("model.language_model."):
        return name.replace("model.language_model.", "language_model.model.", 1)
    if name.startswith("model.visual."):
        return name.replace("model.visual.", "vision_tower.", 1)
    if name.startswith("lm_head."):
        return "language_model." + name
    raise ValueError(f"unexpected source key: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load(args.fixtures / "hf")
    records = []
    for family in ("vlm", "lm"):
        actual = load(args.fixtures / (family + "-float"))
        expected = {}
        shifted = []
        for name, value in source.items():
            if family == "lm" and name.startswith("model.visual."):
                continue
            target = expected_name(name)
            if name.endswith("conv1d.weight"):
                value = value.transpose(0, 2, 1)
            elif name.endswith("patch_embed.proj.weight"):
                value = value.transpose(0, 2, 3, 4, 1)
            if name.startswith("model.language_model.") and (
                    name.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight",
                                   ".q_norm.weight", ".k_norm.weight"))
                    or name == "model.language_model.norm.weight"):
                value = value + np.float32(1)
                shifted.append(target)
            if target in expected:
                raise ValueError(f"colliding remap: {target}")
            expected[target] = value
        assert set(expected) == set(actual), (set(expected)-set(actual), set(actual)-set(expected))
        for name, value in expected.items():
            stored = actual[name]
            assert value.dtype == stored.dtype and value.shape == stored.shape, name
            assert value.tobytes() == stored.tobytes(), name
        records.append(dict(family=family, tensor_count=len(actual), bit_exact=True,
                            shifted_norms=shifted, convolution_axes="C,K,1",
                            linear_attn_norm_shifted=False,
                            vision_present=any(name.startswith("vision_tower.") for name in actual)))
    args.output.write_text(json.dumps(dict(
        kind="saved actual-converter value checks, not neural acceptance",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=records), indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
