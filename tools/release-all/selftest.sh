#!/usr/bin/env bash
# Run portable release regressions, including real Gradle repository/collector fixtures.
set -euo pipefail
release_test_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
"$release_test_root/gradlew" --version >/dev/null
release_test_version=$(sed -n 's|.*gradle-\([0-9.]*\)-bin.zip.*|\1|p' "$release_test_root/gradle/wrapper/gradle-wrapper.properties")
release_test_dists="${GRADLE_USER_HOME:-$HOME/.gradle}/wrapper/dists"
release_test_bin=$(find "$release_test_dists/gradle-$release_test_version-bin" -path '*/bin/gradle' -type f -print -quit)
[[ -x "$release_test_bin" ]] || { echo 'Pinned Gradle distribution unavailable' >&2; exit 1; }
JMLX_RELEASE_TEST_GRADLE="$release_test_bin" python3 -m unittest discover \
  -s "$release_test_root/tools/release-all/tests" -v
