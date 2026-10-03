#!/usr/bin/env bash
# Runs the release smoke consumer against the six jmlx modules.
#
#   run.sh ci                 publish the current SNAPSHOTs to a disposable repo, then smoke them
#   run.sh candidate          same, with SNAPSHOT suffixes stripped from the consumer's versions
#                             (set the modules to their release versions first; rejects SNAPSHOTs)
#   run.sh central            resolve everything from Maven Central, after all six are published
#   run.sh ... --record       (ci only) rewrite goldens/mistral-sampled.properties
#
# Every run uses a fresh GRADLE_USER_HOME, so no cached or mavenLocal jmlx artifact can be reused.
# macOS ARM64 + Java 25 only; needs the native runtime staged for the publish step.
set -euo pipefail

mode="${1:-ci}"
shift || true
record=false
[ "${1:-}" = "--record" ] && record=true
# Rejected before any work: the ci/candidate block deletes the smoke repo and publishes all six
# modules, so a later check would not undo that.
if [ "$record" = true ] && [ "$mode" != "ci" ]; then
  echo "error: --record is ci only: it would overwrite the committed golden with whatever '$mode' serves" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
smoke="$root/tools/release-smoke"
repo="$root/build/smoke-repo"
modules="jmlx-jinja jmlx-tokenizer jmlx-native-macos-arm64 jmlx-ffi jmlx-core jmlx-models"

# Locate a Gradle distribution through the checkout's wrapper (default home), then run the consumer
# with that binary under a throwaway home. `--version` makes the wrapper download it if needed.
"$root/gradlew" --version >/dev/null
wrapper_version="$(sed -n 's|.*gradle-\([0-9.]*\)-bin.zip.*|\1|p' "$root/gradle/wrapper/gradle-wrapper.properties")"
dists_dir="${GRADLE_USER_HOME:-$HOME/.gradle}/wrapper/dists"
# || true: under `set -euo pipefail` a bare `ls ... | head -1` assignment exits the script on a
# missing distribution (the original failure mode), before the check below can report it.
gradle_bin="$(ls -d "$dists_dir"/gradle-"$wrapper_version"-bin/*/gradle-"$wrapper_version"/bin/gradle 2>/dev/null | head -1 || true)"
if [ -z "$gradle_bin" ] || [ ! -x "$gradle_bin" ]; then
  echo "error: no usable Gradle $wrapper_version distribution under $dists_dir" >&2
  echo "hint: the consumer must run with the checkout's wrapper distribution; run" >&2
  echo "      '$root/gradlew --version' once from this checkout (with its default" >&2
  echo "      GRADLE_USER_HOME) so the wrapper downloads it" >&2
  exit 2
fi

args=()
case "$mode" in
  ci | candidate)
    rm -rf "$repo"
    tasks=()
    for m in $modules; do tasks+=(":$m:publishMavenPublicationToSmokeRepository"); done
    (cd "$root" && ./gradlew -q "${tasks[@]}")
    args+=("-PjmlxRepoUrl=file://$repo")
    [ "$mode" = candidate ] && args+=("-PjmlxMode=release") || args+=("-PjmlxMode=ci")
    ;;
  central)
    args+=("-PjmlxMode=release")
    ;;
  *)
    echo "usage: $0 ci|candidate|central [--record]" >&2
    exit 2
    ;;
esac
$record && args+=("-PsmokeRecord=true")

home="$(mktemp -d)"
trap 'rm -rf "$home"' EXIT
# Forward any version overrides (e.g. JMLX_MODELS_VERSION) as -P properties.
for pair in JMLX_MODELS_VERSION:jmlxModelsVersion JMLX_NATIVE_VERSION:jmlxNativeVersion \
  JMLX_CORE_VERSION:jmlxCoreVersion JMLX_FFI_VERSION:jmlxFfiVersion \
  JMLX_TOKENIZER_VERSION:jmlxTokenizerVersion JMLX_JINJA_VERSION:jmlxJinjaVersion; do
  var="${pair%%:*}"
  [ -n "${!var:-}" ] && args+=("-P${pair#*:}=${!var}")
done

env -u JMLX_LIBRARY_PATH GRADLE_USER_HOME="$home" \
  "$gradle_bin" --project-dir "$smoke" --no-daemon --no-build-cache "${args[@]}" smoke
