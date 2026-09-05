#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${SCRIPT_DIR}/.venv"

"$PYTHON_BIN" -m venv --clear "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --require-hashes --only-binary=:all: -r "$SCRIPT_DIR/requirements.lock"
