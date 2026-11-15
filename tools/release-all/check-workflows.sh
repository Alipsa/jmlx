#!/usr/bin/env bash
# Pin actionlint so unsupported GitHub expression contexts fail before workflow dispatch.
set -euo pipefail
workflow_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)
    workflow_platform=linux_amd64
    workflow_sha=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8 ;;
  Linux/aarch64)
    workflow_platform=linux_arm64
    workflow_sha=325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6 ;;
  Darwin/arm64)
    workflow_platform=darwin_arm64
    workflow_sha=aba9ced2dee8d27fecca3dc7feb1a7f9a52caefa1eb46f3271ea66b6e0e6953f ;;
  *) echo 'Unsupported actionlint host' >&2; exit 2 ;;
esac
workflow_temp=$(mktemp -d)
trap 'rm -rf "$workflow_temp"' EXIT
curl --fail --silent --show-error --location \
  "https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_$workflow_platform.tar.gz" \
  --output "$workflow_temp/actionlint.tar.gz"
python3 - "$workflow_temp/actionlint.tar.gz" "$workflow_sha" <<'PY'
import hashlib, sys
with open(sys.argv[1], 'rb') as stream:
    if hashlib.sha256(stream.read()).hexdigest() != sys.argv[2]:
        raise SystemExit('actionlint archive digest mismatch')
PY
tar -xzf "$workflow_temp/actionlint.tar.gz" -C "$workflow_temp" actionlint
"$workflow_temp/actionlint" -config-file "$workflow_root/.github/actionlint.yaml" \
  -shellcheck= -pyflakes= "$workflow_root"/.github/workflows/*.yml
