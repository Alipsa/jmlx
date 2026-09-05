#!/usr/bin/env python3
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def byte_offsets(text: str, offsets: list[tuple[int, int]]) -> list[list[int]]:
    return [
        [len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))]
        for start, end in offsets
    ]


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fixture(input_path: Path, provenance: dict) -> dict:
    from tokenizers import Tokenizer

    expected_input_digest = provenance["fixtureSources"].get(input_path.name)
    if expected_input_digest is None or source_digest(input_path) != expected_input_digest:
        raise SystemExit(
            f"tokenizer oracle input differs from provenance: {input_path.name}"
        )
    specification = json.loads(input_path.read_text())
    fixtures = []
    root = input_path.parent.resolve()
    for fixture in specification["fixtures"]:
        tokenizer_path = (root / fixture["tokenizer"]).resolve()
        if root not in tokenizer_path.parents or not tokenizer_path.is_file():
            raise SystemExit(f"tokenizer fixture must be a committed local file: {tokenizer_path}")
        expected_digest = provenance["fixtureSources"].get(tokenizer_path.name)
        if expected_digest is None:
            raise SystemExit(f"tokenizer fixture has no provenance digest: {tokenizer_path.name}")
        actual_digest = source_digest(tokenizer_path)
        if actual_digest != expected_digest:
            raise SystemExit(
                f"tokenizer fixture digest differs from provenance: {tokenizer_path.name}"
            )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        cases = []
        for case in fixture["cases"]:
            tokenizer.no_truncation()
            tokenizer.no_padding()
            if "truncation" in case:
                tokenizer.enable_truncation(**case["truncation"])
            if "padding" in case:
                tokenizer.enable_padding(**case["padding"])
            text = case["text"]
            encoding = tokenizer.encode(
                text, add_special_tokens=case["addSpecialTokens"]
            )
            cases.append(
                {
                    "attentionMask": encoding.attention_mask,
                    "decoded": tokenizer.decode(
                        encoding.ids, skip_special_tokens=True
                    ),
                    "ids": encoding.ids,
                    "name": case["name"],
                    "offsets": byte_offsets(text, encoding.offsets),
                    "specialTokensMask": encoding.special_tokens_mask,
                    "tokens": encoding.tokens,
                    "typeIds": encoding.type_ids,
                }
            )
        fixtures.append({"cases": cases, "name": fixture["name"]})
    return {"fixtures": fixtures}


def stale_message(path: Path, expected: str, actual: str) -> str:
    try:
        expected_value = json.loads(expected)
        actual_value = json.loads(actual)
        expected_lines = (
            json.dumps(expected_value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).splitlines(keepends=True)
        actual_lines = (
            json.dumps(actual_value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).splitlines(keepends=True)
    except json.JSONDecodeError:
        expected_lines = expected.splitlines(keepends=True)
        actual_lines = actual.splitlines(keepends=True)
    difference = "".join(
        difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile=str(path),
            tofile="oracle output",
        )
    )
    if not difference:
        difference = (
            "documents are structurally identical; the committed file is not in "
            "canonical form (key order, separators, indentation, or trailing newline)\n"
        )
    return f"tokenizer oracle fixture is stale: {path}\n{difference}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate-all", action="store_true")
    action.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    provenance = json.loads(args.provenance.read_text())
    inputs = sorted(args.fixtures_dir.glob("*.input.json"))
    if not inputs:
        raise SystemExit("no tokenizer oracle input fixtures found")
    expected_names = {path.name for path in args.fixtures_dir.glob("*.expected.json")}
    declared_names = {
        path.name.replace(".input.json", ".expected.json") for path in inputs
    }
    orphaned = sorted(expected_names - declared_names)
    if orphaned:
        raise SystemExit(f"orphaned tokenizer oracle fixtures: {orphaned}")
    for input_path in inputs:
        expected_path = input_path.with_name(
            input_path.name.replace(".input.json", ".expected.json")
        )
        actual = canonical(run_fixture(input_path, provenance))
        if args.generate_all:
            expected_path.write_text(actual)
        else:
            if not expected_path.is_file():
                raise SystemExit(f"tokenizer oracle fixture is missing: {expected_path}")
            expected = expected_path.read_text()
            if actual != expected:
                raise SystemExit(stale_message(expected_path, expected, actual))
            print(f"Tokenizer oracle fixture verified: {expected_path}")


if __name__ == "__main__":
    main()
