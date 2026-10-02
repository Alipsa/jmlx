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

root="$(cd "$(dirname "$0")/../.." && pwd)"
smoke="$root/tools/release-smoke"
repo="$root/build/smoke-repo"
modules="jmlx-jinja jmlx-tokenizer jmlx-native-macos-arm64 jmlx-ffi jmlx-core jmlx-models"

# Locate a Gradle distribution through the checkout's wrapper (default home), then run the consumer
# with that binary under a throwaway home. `--version` makes the wrapper download it if needed.
"$root/gradlew" --version >/dev/null
wrapper_version="$(sed -n 's|.*gradle-\([0-9.]*\)-bin.zip.*|\1|p' "$root/gradle/wrapper/gradle-wrapper.properties")"
gradle_bin="$(ls -d "$HOME"/.gradle/wrapper/dists/gradle-"$wrapper_version"-bin/*/gradle-"$wrapper_version"/bin/gradle | head -1)"

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
