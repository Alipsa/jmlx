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
# fragile to formatting. `properties` emits exactly one `version:` line, but stopping sed at the
# first match (rather than piping through `head -n 1`) means a future duplicate can't turn into a
# silent SIGPIPE/141 abort under `set -o pipefail`.
VERSION=$(./gradlew -q ":$MODULE:properties" | sed -n 's/^version: //p;q')
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
