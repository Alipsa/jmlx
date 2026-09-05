#!/usr/bin/env bash
set -euo pipefail

ORACLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV="$ORACLE_DIR/.venv"

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
  echo "MLX oracle requires macOS on arm64" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null || {
  echo "MLX oracle requires CPython 3.12 ($PYTHON_BIN was not found)" >&2
  exit 1
}
"$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || {
  echo "MLX oracle requires CPython 3.12 ($PYTHON_BIN has a different version)" >&2
  exit 1
}

"$PYTHON_BIN" -m venv --clear "$VENV"
"$VENV/bin/python" -m pip install \
  --require-hashes \
  --only-binary=:all: \
  --requirement "$ORACLE_DIR/requirements.lock"
