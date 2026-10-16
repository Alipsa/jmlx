#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV="${SCRIPT_DIR}/.venv"

command -v "$PYTHON_BIN" >/dev/null || {
  echo "Qwen-VL MLX reference requires CPython 3.12 ($PYTHON_BIN was not found)" >&2
  exit 1
}
"$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || {
  echo "Qwen-VL MLX reference requires CPython 3.12 ($PYTHON_BIN has a different version)" >&2
  exit 1
}

# The lock was resolved for macOS arm64 (the mlx wheels have no other matching
# platform for this pin set); this venv is an Apple-Silicon-only opt-in tool.
"$PYTHON_BIN" -m venv --clear "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --require-hashes \
  --only-binary=:all: -r "$SCRIPT_DIR/requirements.lock"
