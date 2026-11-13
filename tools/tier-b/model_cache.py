#!/usr/bin/env python3
"""Remove files outside a Tier-B manifest from a persistent model cache."""

import argparse
import json
from pathlib import Path


def clean_model_dir(model_dir: Path, manifest: Path) -> None:
    """Keep only manifest-listed regular files in a persistent model cache."""
    listed = set(json.loads(manifest.read_text(encoding="utf-8"))["files"])
    model_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(model_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        relative = path.relative_to(model_dir).as_posix()
        if path.is_symlink() or (path.is_file() and relative not in listed):
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    clean_model_dir(args.model_dir, args.manifest)
