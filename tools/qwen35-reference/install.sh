#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PROFILE="${1:-}"
case "$PROFILE" in
  hf|mlx) ;;
  *) echo "Usage: $0 hf|mlx" >&2; exit 1 ;;
esac
"$PYTHON_BIN" -c 'import platform, sys; assert sys.version_info[:2] == (3, 12); assert platform.system() == "Darwin" and platform.machine() == "arm64"' || {
  echo "These locks require CPython 3.12 on macOS arm64" >&2
  exit 1
}
VENV="$SCRIPT_DIR/$PROFILE/.venv"
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --require-hashes \
  --only-binary=:all: -r "$SCRIPT_DIR/$PROFILE/requirements.lock"
"$VENV/bin/python" -m pip check
