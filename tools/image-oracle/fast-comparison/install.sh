#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV="${SCRIPT_DIR}/.venv"

command -v "$PYTHON_BIN" >/dev/null || {
  echo "Fast comparison environment requires CPython 3.12 ($PYTHON_BIN was not found)" >&2
  exit 1
}
"$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || {
  echo "Fast comparison environment requires CPython 3.12 ($PYTHON_BIN has a different version)" >&2
  exit 1
}

"$PYTHON_BIN" -m venv --clear "$VENV"
# The non-Darwin lock entry is Torch 2.6.0+cpu, which only exists on the PyTorch
# CPU wheel index (the lock was compiled with it as an extra index).
"$VENV/bin/python" -m pip install --disable-pip-version-check --require-hashes \
  --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$SCRIPT_DIR/requirements.lock"
