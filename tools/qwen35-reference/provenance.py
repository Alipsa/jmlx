#!/usr/bin/env python3
"""Explicitly record, or verify, the committed local reference file inventory."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def files(root):
    result = {}
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(name for name in folders if name not in (".venv", "__pycache__"))
        for name in sorted(names):
            path = Path(directory) / name
            if path == root / "provenance.json" or path.suffix in (".pyc", ".pyo"):
                continue
            result[path.relative_to(root).as_posix()] = digest(path)
    return dict(sorted(result.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parent.parent
    shared = ["tools/image-oracle/fast-comparison/fixtures/model-rgb-256x256.png",
              "tools/tier-b/qwen35-08b.json", "tools/tier-b/qwen38-27b-4bit.json"]
    record = dict(schema="jmlx-qwen35-reference-v1",
                  status="reference probes, component and converter fixtures, and real-artifact "
                         "acceptance evidence (0.8B Tier-B and 27B manual run)",
                  files=files(root), sharedFiles={path: digest(repo/path) for path in shared})
    path = root / "provenance.json"
    if args.record:
        path.write_text(json.dumps(record, indent=2)+"\n")
        print(path)
    else:
        expected = json.loads(path.read_text())
        assert record == expected, "Reference files differ from the reviewed provenance"
        print(f'Verified {len(record["files"])} local files and {len(record["sharedFiles"])} shared files')


if __name__ == "__main__":
    main()
