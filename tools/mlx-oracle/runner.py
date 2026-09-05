#!/usr/bin/env python3
import argparse
import difflib
import json
import math
from pathlib import Path

import mlx.core as mx


def rounded(values):
    if isinstance(values, list):
        return [rounded(value) for value in values]
    if isinstance(values, float):
        if math.isinf(values):
            return "Infinity" if values > 0 else "-Infinity"
        return round(values, 7)
    return values


def dtype_name(array) -> str:
    for name, dtype in (("float32", mx.float32), ("uint32", mx.uint32)):
        if array.dtype == dtype:
            return name
    raise ValueError(f"unexpected oracle dtype: {array.dtype}")


def array_fixture(specification: dict) -> dict:
    logits = mx.array(specification["logits"], dtype=mx.float32)
    doubled = logits * mx.array(2.0, dtype=mx.float32)
    softmax = mx.softmax(logits, axis=1)
    argmax = mx.argmax(logits, axis=1)
    mx.eval(doubled, softmax, argmax)
    return {
        "argmax": {
            "dtype": dtype_name(argmax),
            "shape": list(argmax.shape),
            "values": argmax.tolist(),
        },
        "doubled": {
            "dtype": dtype_name(doubled),
            "shape": list(doubled.shape),
            "values": rounded(doubled.tolist()),
        },
        "softmax": {
            "dtype": dtype_name(softmax),
            "shape": list(softmax.shape),
            "values": rounded(softmax.tolist()),
        },
    }


def sampling_fixture(specification: dict) -> dict:
    results = []
    for case in specification["cases"]:
        logits = mx.array(case["logits"], dtype=mx.float32)
        policy = case["policy"]
        penalized = logits
        history = case.get("history", {})
        if history:
            ids = mx.array([int(token) for token in history], dtype=mx.int32)
            counts = mx.array([history[token] for token in history], dtype=mx.float32)
            selected_logits = mx.take_along_axis(penalized, ids, axis=-1)
            repetition = policy["repetitionPenalty"]
            selected_logits = mx.where(
                selected_logits >= 0,
                selected_logits / repetition,
                selected_logits * repetition,
            )
            selected_logits -= counts * policy["frequencyPenalty"]
            selected_logits -= policy["presencePenalty"]
            penalized = mx.put_along_axis(penalized, ids, selected_logits, axis=-1)

        temperature = policy["temperature"]
        if temperature == 0:
            selected = mx.argmax(penalized, axis=-1)
            mx.eval(penalized, selected)
            results.append(
                {
                    "name": case["name"],
                    "penalizedLogits": rounded(penalized.tolist()),
                    "selectedToken": int(selected.item()),
                    "selectedLogProbability": 0.0,
                }
            )
            continue

        tempered = penalized / temperature
        order = mx.argsort(-tempered, axis=-1)
        sorted_logits = mx.take_along_axis(tempered, order, axis=-1)
        top_k_logits = sorted_logits
        top_k = policy["topK"]
        if top_k:
            ranks = mx.arange(sorted_logits.shape[-1], dtype=mx.int32)
            top_k_logits = mx.where(ranks < top_k, sorted_logits, float("-inf"))
        top_p_logits = top_k_logits
        top_p = policy["topP"]
        if top_p < 1:
            if top_p == 0:
                keep = mx.arange(sorted_logits.shape[-1], dtype=mx.int32) < 1
            else:
                probabilities = mx.softmax(top_k_logits, axis=-1)
                previous = mx.cumsum(probabilities, axis=-1) - probabilities
                keep = previous < top_p
            top_p_logits = mx.where(keep, top_k_logits, float("-inf"))
        min_p_logits = top_p_logits
        min_p = policy["minP"]
        if min_p:
            probabilities = mx.softmax(top_p_logits, axis=-1)
            keep = probabilities >= min_p * probabilities[..., :1]
            min_p_logits = mx.where(keep, top_p_logits, float("-inf"))
        vocabulary_logits = mx.put_along_axis(
            mx.zeros_like(min_p_logits), order, min_p_logits, axis=-1
        )
        key = mx.random.key(case["seed"])
        children = mx.random.split(key, num=2)
        selected = mx.random.categorical(vocabulary_logits, axis=-1, key=children[1])
        selected_logit = mx.take_along_axis(vocabulary_logits, selected[..., None], axis=-1)
        selected_log_probability = selected_logit - mx.logsumexp(
            vocabulary_logits, axis=-1, keepdims=True
        )
        mx.eval(
            penalized,
            tempered,
            order,
            top_k_logits,
            top_p_logits,
            min_p_logits,
            vocabulary_logits,
            children,
            selected,
            selected_log_probability,
        )
        results.append(
            {
                "name": case["name"],
                "penalizedLogits": rounded(penalized.tolist()),
                "temperedLogits": rounded(tempered.tolist()),
                "sortedTokenIds": order.tolist(),
                "topKLogits": rounded(top_k_logits.tolist()),
                "topPLogits": rounded(top_p_logits.tolist()),
                "minPLogits": rounded(min_p_logits.tolist()),
                "vocabularyLogits": rounded(vocabulary_logits.tolist()),
                "splitKeys": children.tolist(),
                "selectedToken": int(selected.item()),
                "selectedLogProbability": rounded(float(selected_log_probability.item())),
            }
        )
    return {"cases": results}


def run(specification: dict, provenance: dict) -> dict:
    device = specification.get("device")
    recorded_device = provenance["device"]["type"]
    if device != recorded_device:
        raise ValueError(
            f"oracle fixture device {device} does not match recorded device {recorded_device}"
        )
    devices = {"cpu": mx.cpu, "gpu": mx.gpu}
    if device not in devices:
        raise ValueError(f"unsupported oracle device: {device}")
    mx.set_default_device(devices[device])
    fixture = specification.get("fixture")
    if fixture == "phase6-tier-a-array":
        result = array_fixture(specification)
    elif fixture == "phase6-1-sampling":
        result = sampling_fixture(specification)
    else:
        raise ValueError(f"unknown fixture: {fixture}")
    return {
        "fixture": specification["fixture"],
        "device": device,
        **result,
    }


def canonical(value: dict) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"


def stale_fixture_message(path: Path, expected: str, actual: str) -> str:
    def pretty_lines(document: str) -> list[str]:
        try:
            parsed = json.loads(document)
        except json.JSONDecodeError:
            lines = document.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += "\n"
            return lines
        pretty = json.dumps(parsed, indent=2, sort_keys=True) + "\n"
        return pretty.splitlines(keepends=True)

    difference = "".join(
        difflib.unified_diff(
            pretty_lines(expected),
            pretty_lines(actual),
            fromfile=f"{path} (committed)",
            tofile=f"{path} (generated)",
        )
    )
    if not difference:
        difference = (
            "documents are structurally identical; the committed file is not in canonical form "
            "(key order, separators, indentation, or trailing newline)\n"
        )
    return f"oracle fixture is stale: {path}\n{difference}"


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    output.add_argument("--generate-all", action="store_true")
    output.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()

    if args.fixtures_dir:
        if not (args.generate_all or args.verify_all):
            parser.error("--fixtures-dir requires --generate-all or --verify-all")
        inputs = {path.name.removesuffix(".input.json"): path for path in args.fixtures_dir.glob("*.input.json")}
        expected = {path.name.removesuffix(".expected.json"): path for path in args.fixtures_dir.glob("*.expected.json")}
        orphan_expected = sorted(expected.keys() - inputs.keys())
        if orphan_expected:
            raise SystemExit(f"orphan oracle expected fixtures: {', '.join(orphan_expected)}")
        missing_expected = sorted(inputs.keys() - expected.keys())
        if args.verify_all and missing_expected:
            raise SystemExit(f"oracle inputs missing expected fixtures: {', '.join(missing_expected)}")
        provenance = json.loads(args.provenance.read_text())
        for name, input_path in sorted(inputs.items()):
            expected_path = args.fixtures_dir / f"{name}.expected.json"
            actual = canonical(run(json.loads(input_path.read_text()), provenance))
            if args.generate_all:
                expected_path.write_text(actual)
            else:
                expected_text = expected_path.read_text()
                if actual != expected_text:
                    raise SystemExit(
                        stale_fixture_message(expected_path, expected_text, actual)
                    )
                print(f"MLX oracle fixture verified: {expected_path}")
        return

    if not (args.output or args.verify):
        parser.error("--input requires --output or --verify")

    actual = canonical(
        run(json.loads(args.input.read_text()), json.loads(args.provenance.read_text()))
    )
    if args.output:
        args.output.write_text(actual)
        return
    expected = args.verify.read_text()
    if actual != expected:
        raise SystemExit(stale_fixture_message(args.verify, expected, actual))
    print(f"MLX oracle fixture verified: {args.verify}")


if __name__ == "__main__":
    main()
