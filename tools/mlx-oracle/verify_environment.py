#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path


def require_equal(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: expected {expected}, found {actual}")


def read_properties(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    )


def read_lock(path: Path) -> dict[str, tuple[str, str]]:
    logical = re.sub(r"\\\n\s*", " ", path.read_text())
    entries: dict[str, tuple[str, str]] = {}
    pattern = re.compile(r"([A-Za-z0-9-]+)==([^ ]+)\s+--hash=sha256:([a-f0-9]{64})")
    for line in logical.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if not match or match.group(1) in entries:
            raise SystemExit(f"invalid or duplicate locked requirement: {line}")
        entries[match.group(1)] = (match.group(2), match.group(3))
    return entries


def verify_runtime(device: str) -> None:
    try:
        import mlx.core as mx

        devices = {"cpu": mx.cpu, "gpu": mx.gpu}
        mx.set_default_device(devices[device])
        value = mx.array([1.0], dtype=mx.float32)
        mx.eval(value)
    except Exception as error:
        raise SystemExit(f"MLX oracle runtime smoke check failed: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--staged-pins", type=Path, required=True)
    parser.add_argument("--mlx-metal-version", required=True)
    parser.add_argument("--mlx-metal-url", required=True)
    parser.add_argument("--mlx-metal-sha256", required=True)
    parser.add_argument("--mlx-c-commit", required=True)
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text())
    require_equal("platform system", platform.system(), "Darwin")
    require_equal("platform machine", platform.machine(), "arm64")
    require_equal("recorded system", provenance["platform"]["system"], "Darwin")
    require_equal("recorded machine", provenance["platform"]["machine"], "arm64")
    macos_major = platform.mac_ver()[0].split(".")[0]
    require_equal("macOS major version", macos_major, provenance["platform"]["macOSMajor"])
    require_equal(
        "Python",
        f"{sys.version_info.major}.{sys.version_info.minor}",
        provenance["python"]["majorMinor"],
    )
    require_equal("mlx", importlib.metadata.version("mlx"), provenance["mlx"]["version"])
    require_equal(
        "mlx-metal",
        importlib.metadata.version("mlx-metal"),
        provenance["mlxMetal"]["version"],
    )
    require_equal(
        "mlx-metal pin", provenance["mlxMetal"]["version"], args.mlx_metal_version
    )
    require_equal(
        "mlx-metal URL", provenance["mlxMetal"]["url"], args.mlx_metal_url
    )
    require_equal(
        "mlx-metal SHA-256",
        provenance["mlxMetal"]["sha256"],
        args.mlx_metal_sha256,
    )
    require_equal("mlx-c commit", provenance["mlxCCommit"], args.mlx_c_commit)
    device = provenance["device"]["type"]
    if device not in {"cpu", "gpu"}:
        raise SystemExit(f"unsupported oracle device: {device}")
    verify_runtime(device)

    if args.staged_pins.is_file():
        staged = read_properties(args.staged_pins)
        require_equal(
            "staged mlx-metal version",
            staged.get("mlxMetalVersion", "<missing>"),
            args.mlx_metal_version,
        )
        require_equal(
            "staged mlx-c commit", staged.get("mlxcCommit", "<missing>"), args.mlx_c_commit
        )

    lock_entries = read_lock(args.lock)
    if set(lock_entries) != {"mlx", "mlx-metal"}:
        raise SystemExit("requirements.lock must contain exactly mlx and mlx-metal")
    for package in ("mlx", "mlx-metal"):
        metadata = provenance["mlx" if package == "mlx" else "mlxMetal"]
        expected = (metadata["version"], metadata["sha256"])
        if lock_entries.get(package) != expected:
            raise SystemExit(
                f"requirements.lock must pin {package}=={expected[0]} "
                f"with --hash=sha256:{expected[1]}"
            )
    print(
        f"MLX oracle verified: Python {platform.python_version()}, "
        f"mlx {provenance['mlx']['version']}, mlx-metal {provenance['mlxMetal']['version']}"
    )


if __name__ == "__main__":
    main()
