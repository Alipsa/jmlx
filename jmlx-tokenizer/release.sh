#!/usr/bin/env bash
set -euo pipefail

# Byte-identical in every releasable module: the module name is derived from this script's own
# directory, so there is nothing module-specific to keep in sync. The Gradle wrapper lives only
# at the monorepo root, so we run from there with a qualified task path.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODULE=$(basename "$SCRIPT_DIR")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# An independent release requires the module to carry its own version. Without this it silently
# inherits the root version and every module would move in lockstep -- the exact accident this
# tooling exists to prevent.
if ! grep -qE "^version[[:space:]]*=[[:space:]]*[\"']" "$MODULE/build.gradle"; then
  echo "$MODULE/build.gradle declares no version of its own; it would inherit the root version." >&2
  exit 1
fi

# Ask Gradle for the effective version rather than parsing build.gradle: authoritative, and not
# fragile to formatting. Use awk, not `sed -n 's/^version: //p;q'`: that `q` carries no address, so
# it quits unconditionally after line 1 -- almost always before "version: " is even reached in
# `properties`' long, alphabetized output -- producing empty output every time, not just under a
# race. awk's `/pattern/{...; exit}` correctly restricts both the print and the exit to the
# matching line.
VERSION=$(./gradlew -q ":$MODULE:properties" | awk '/^version: /{print $2; exit}')
if [[ -z "$VERSION" ]]; then
  echo "Could not determine the version of $MODULE" >&2
  exit 1
fi
if [[ "$VERSION" == *-SNAPSHOT ]]; then
  echo "$MODULE $VERSION is a snapshot and cannot be published" >&2
  exit 1
fi

echo "Building and validating the release bundle for $MODULE $VERSION before publishing to Maven Central..."
./gradlew ":$MODULE:clean" ":$MODULE:build" ":$MODULE:release"
echo "$MODULE $VERSION published"
echo "see https://central.sonatype.com/publishing/deployments for more info"
