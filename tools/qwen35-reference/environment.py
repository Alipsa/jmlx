#!/usr/bin/env python3
"""Record installed distributions, imported-source files and exact selected wheel artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("hf", "mlx"), required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    lock = root / args.profile / "requirements.lock"
    lock_text = lock.read_text()
    wheels = []
    for path in sorted(args.wheels.glob("*.whl")):
        checksum = digest(path)
        assert "sha256:" + checksum in lock_text, path
        with zipfile.ZipFile(path) as archive:
            wheel_metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
            tags = [line.removeprefix("Tag: ") for line in archive.read(wheel_metadata).decode().splitlines()
                    if line.startswith("Tag: ")]
        wheels.append(dict(file=path.name, sha256=checksum, bytes=path.stat().st_size, tags=tags))
    assert wheels
    sources = {}
    for package in ("transformers", "mlx-vlm", "mlx-lm"):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        for relative in distribution.files:
            name = relative.as_posix()
            if name.endswith(".py") and ("/qwen3_5/" in name or name.endswith("/qwen3_5.py")
                                          or name.endswith("/gated_delta.py")
                                          or name in ("mlx_vlm/convert.py", "mlx_lm/convert.py")
                                          or name.endswith("/image_processing_pil_qwen2_vl.py")):
                sources[name] = digest(Path(distribution.locate_file(relative)))
    record = dict(profile=args.profile, platform=platform.platform(), machine=platform.machine(),
                  python=platform.python_version(), executable_version=sys.version,
                  lock_sha256=digest(lock), selected_wheels=wheels,
                  installed={d.metadata["Name"]: d.version for d in sorted(
                      importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower())},
                  bootstrap_note="pip/setuptools are installer tooling; compute dependencies are hash locked",
                  installed_sources=sources, recorder_sha256=digest(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
