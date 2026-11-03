#!/usr/bin/env python3
"""Hash complete local artifacts and inspect safetensors headers without loading weights."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inventory(root):
    files = {}
    tensors = {}
    metadata = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        files[relative] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        if path.suffix != ".safetensors":
            continue
        with path.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
            if header_size > 100 * 1024 * 1024:
                raise ValueError(f"Unreasonable header size: {path}")
            header = json.loads(source.read(header_size))
        metadata[relative] = header.pop("__metadata__", None)
        for name, descriptor in sorted(header.items()):
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            start, end = descriptor["data_offsets"]
            if not 0 <= start <= end <= path.stat().st_size - 8 - header_size:
                raise ValueError(f"Invalid data range: {name}")
            tensors[name] = dict(shard=relative, **descriptor)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        expected = json.loads(index_path.read_text())["weight_map"]
        actual = {name: tensor["shard"] for name, tensor in tensors.items()}
        if actual != expected:
            raise ValueError(f"Shard/index mismatch; missing={set(expected)-set(actual)}, "
                             f"extra={set(actual)-set(expected)}")
    if not tensors:
        raise ValueError("No safetensors found")
    return dict(files=files, tensor_count=len(tensors), tensors=tensors, shard_metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--revision")
    args = parser.parse_args()
    record = inventory(args.root)
    record.update(repo=args.repo, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f'{record["tensor_count"]} tensors: {args.output}')


if __name__ == "__main__":
    main()
