#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import mlx.core as mx


def rounded(values):
    if isinstance(values, list):
        return [rounded(value) for value in values]
    if isinstance(values, float):
        return round(values, 7)
    return values


def dtype_name(array) -> str:
    for name, dtype in (("float32", mx.float32), ("uint32", mx.uint32)):
        if array.dtype == dtype:
            return name
    raise ValueError(f"unexpected oracle dtype: {array.dtype}")


def run(specification: dict) -> dict:
    if specification.get("fixture") != "phase6-tier-a-array":
        raise ValueError(f"unknown fixture: {specification.get('fixture')}")
    logits = mx.array(specification["logits"], dtype=mx.float32)
    doubled = logits * mx.array(2.0, dtype=mx.float32)
    softmax = mx.softmax(logits, axis=1)
    argmax = mx.argmax(logits, axis=1)
    mx.eval(doubled, softmax, argmax)
    return {
        "fixture": specification["fixture"],
        "argmax": {
            "dtype": dtype_name(argmax),
            "shape": list(argmax.shape),
            "values": argmax.tolist(),
        },
        "doubled": {
            "dtype": dtype_name(doubled),
            "shape": list(doubled.shape),
            "values": doubled.tolist(),
        },
        "softmax": {
            "dtype": dtype_name(softmax),
            "shape": list(softmax.shape),
            "values": rounded(softmax.tolist()),
        },
    }


def canonical(value: dict) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    args = parser.parse_args()

    actual = canonical(run(json.loads(args.input.read_text())))
    if args.output:
        args.output.write_text(actual)
        return
    expected = args.verify.read_text()
    if actual != expected:
        raise SystemExit(
            f"oracle fixture is stale: {args.verify}\nexpected: {expected}actual:   {actual}"
        )
    print(f"MLX oracle fixture verified: {args.verify}")


if __name__ == "__main__":
    main()
