#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/.venv"

[ -x "$VENV/bin/python" ] || {
  echo "Decode baseline environment is missing; run ./tools/decode-baseline/install.sh" >&2
  exit 1
}

MLX_ENABLE_TF32=1 "$VENV/bin/python" - "$SCRIPT_DIR" <<'EOF'
import importlib.metadata as md
import json
import sys

script_dir = sys.argv[1]
pinned = {}
with open(f"{script_dir}/requirements.in", "r", encoding="utf-8") as fid:
    for line in fid:
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pinned[name.lower().replace("_", "-")] = version

installed = {}
for dist in md.distributions():
    name = dist.metadata["Name"]
    if name:
        installed[name.lower().replace("_", "-")] = dist.version

missing = {k: v for k, v in pinned.items() if installed.get(k) != v}
if missing:
    raise SystemExit(f"installed distributions do not match requirements.in: {missing}")
for name, version in sorted(pinned.items()):
    print(f"pinned {name}=={version}")

import mlx.core as mx
import mlx_lm
import safetensors
import tokenizers
import transformers

import hashlib

digests_path = f"{script_dir}/installed-source-digests.json"
with open(digests_path, "r", encoding="utf-8") as fid:
    digest_doc = json.load(fid)
site_pkg = f"{script_dir}/.venv/lib/python3.12/site-packages"
checked = 0
for rel, expected in sorted(digest_doc["files"].items()):
    with open(f"{site_pkg}/mlx_lm/{rel}", "rb") as fid:
        actual = hashlib.sha256(fid.read()).hexdigest()
    if actual != expected:
        raise SystemExit(f"installed mlx_lm source differs from the WP0 freeze: {rel}")
    checked += 1
if checked != digest_doc["file_count"]:
    raise SystemExit(f"installed-source-digests.json records {digest_doc['file_count']} files, checked {checked}")
print(f"checked {checked} mlx_lm source digests against the WP0 freeze")

print("mlx_lm path:", mlx_lm.__file__)
assert ".venv/lib/python3.12/site-packages" in mlx_lm.__file__, "mlx_lm not from the local venv"
print("mlx core path:", mx.__file__ if hasattr(mx, "__file__") else "built-in")
print("mlx version:", md.version("mlx"))
print("mlx-metal version:", md.version("mlx-metal"))
print("default device:", str(mx.default_device()))
assert mx.default_device().type.name == "gpu", "expected the GPU device"
info = mx.device_info()
print("device info:", json.dumps({k: str(v) for k, v in info.items()}, default=str))
print("VERIFIED")
EOF
