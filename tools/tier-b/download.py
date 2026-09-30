#!/usr/bin/env python3
"""Download one pinned public Tier-B artifact with per-file and total size caps."""

import hashlib
import json
import os
import pathlib
import sys
import urllib.request


def main() -> int:
    manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
    target = pathlib.Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, spec in manifest["files"].items():
        if pathlib.Path(name).name != name:
            raise ValueError(f"invalid artifact filename: {name}")
        url = (
            f'https://huggingface.co/{manifest["repository"]}/resolve/'
            f'{manifest["revision"]}/{name}'
        )
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            size = response.headers.get("Content-Length")
            if size is None or int(size) > spec["max_bytes"]:
                raise ValueError(f"{name}: missing or over-cap Content-Length: {size}")
            if total + int(size) > manifest["max_total_bytes"]:
                raise ValueError("artifact exceeds total size cap")
        destination = target / name
        if destination.is_file() and sha256(destination) == spec["sha256"]:
            total += destination.stat().st_size
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        count = 0
        try:
            with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > spec["max_bytes"] or total + count > manifest["max_total_bytes"]:
                        raise ValueError(f"{name}: download exceeded size cap")
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != spec["sha256"]:
                raise ValueError(f"{name}: SHA-256 mismatch")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        total += count
        print(f"{name}: {count} bytes, SHA-256 verified", flush=True)
    print(f"artifact total: {total} bytes", flush=True)
    return 0


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
