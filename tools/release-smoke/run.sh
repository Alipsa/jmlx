#!/usr/bin/env bash
# Runs the release smoke consumer against the selected jmlx modules.
#
#   run.sh ci                 publish the selected modules' current versions to a disposable
#                             repository, then smoke them (fresh consumer home, exclusive repos)
#   run.sh candidate          same, in release mode: the consumer expects released (non-SNAPSHOT)
#                             versions; the candidate preflight rejects snapshot producers
#   run.sh central            resolve everything from Maven Central, after the modules are published
#
#   --profile all             (default) the seven-module native consumer: Java 25, macOS ARM64, and
#                             the publish step needs the staged native runtime
#   --profile pure-java       the Java 21 headless consumer for jinja/tokenizer/vision:
#                             cross-platform, publishes only those modules, no native runtime needed
#
#   --record                  (ci --profile all only) rewrite goldens/mistral-sampled.properties
#
# Every run uses a fresh GRADLE_USER_HOME, so no cached or mavenLocal jmlx artifact can be reused.
# Version overrides are read from the JMLX_<MODULE>_VERSION environment variables and forwarded as
# -P properties; in the pure profile's release mode an override may only repeat the committed
# released-8.1.properties value. See README.md for JDK requirements and failure diagnosis.
#
# Sourcing this file (selftest.sh) defines parse_args, preflight_candidate and the path helpers and
# returns before main runs. The JMLX_SMOKE_PRODUCER_VERSIONS (a <module>=<version> file),
# JMLX_SMOKE_TUPLE_FILE and JMLX_SMOKE_REPO_DIR environment variables point the preflight at
# disposable fixtures; they are test hooks, not production configuration.
set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }

# This script's own directory, captured at source time so the path helpers keep working when the
# functions are called later from a shell that sourced this file (selftest.sh).
RUN_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path layout, honoring the JMLX_SMOKE_REPO_DIR test hook.
smoke_paths() {
  ROOT="$(cd "$RUN_SH_DIR/../.." && pwd)"
  SMOKE_DIR="$ROOT/tools/release-smoke"
  REPO_DIR="${JMLX_SMOKE_REPO_DIR:-$ROOT/build/smoke-repo}"
}

# parse_args <args...> : sets MODE, PROFILE, RECORD. Rejects unknown modes/profiles, duplicate or
# conflicting arguments and any unused trailing argument, and --record outside ci --profile all.
parse_args() {
  local mode="" profile="" record="false"
  while [ $# -gt 0 ]; do
    case "$1" in
      ci|candidate|central)
        [ -z "$mode" ] || die "duplicate mode '$1'"
        mode="$1"
        ;;
      --profile)
        [ $# -ge 2 ] || die "--profile needs a value"
        [ -z "$profile" ] || die "duplicate --profile"
        profile="$2"
        shift
        ;;
      --profile=*)
        [ -z "$profile" ] || die "duplicate --profile"
        profile="${1#--profile=}"
        ;;
      --record)
        [ "$record" = false ] || die "duplicate --record"
        record="true"
        ;;
      *)
        die "unknown argument '$1' (usage: run.sh ci|candidate|central [--profile all|pure-java] [--record])"
        ;;
    esac
    shift
  done
  [ -n "$mode" ] || die "missing mode (usage: run.sh ci|candidate|central [--profile all|pure-java] [--record])"
  [ -z "$profile" ] && profile="all"
  case "$profile" in
    all|pure-java) ;;
    *) die "unknown profile '$profile' (expected all or pure-java)" ;;
  esac
  # --record rewrites the committed golden, so it is the full profile's ci mode only.
  if [ "$record" = true ] && { [ "$mode" != ci ] || [ "$profile" != all ]; }; then
    die "--record is ci --profile all only: it would overwrite the committed golden"
  fi
  MODE="$mode"
  PROFILE="$profile"
  RECORD="$record"
}

# producer_version <module> : the module's own version line, or the JMLX_SMOKE_PRODUCER_VERSIONS
# fixture's value for it (test hook: a file of <module>=<version> lines).
producer_version() {
  local module="$1" line v
  if [ -n "${JMLX_SMOKE_PRODUCER_VERSIONS:-}" ]; then
    v="$(grep -m1 "^${module}=" "$JMLX_SMOKE_PRODUCER_VERSIONS" | cut -d= -f2- || true)"
    [ -n "$v" ] || die "producer fixture ${JMLX_SMOKE_PRODUCER_VERSIONS} lacks ${module}"
    printf '%s\n' "$v"
    return
  fi
  line="$(grep -m1 "^version = " "$ROOT/$module/build.gradle" || true)"
  [ -n "$line" ] || die "no version line in $module/build.gradle"
  v="${line#version = }"
  v="${v#\'}"
  v="${v%\'}"
  v="${v%$'\n'}"
  printf '%s\n' "$v"
}

# tuple_value <key> : the key's value in the released pure-Java tuple file (JMLX_SMOKE_TUPLE_FILE
# test hook overrides the path).
tuple_value() {
  local tuple="${JMLX_SMOKE_TUPLE_FILE:-$SMOKE_DIR/released-8.1.properties}"
  grep -m1 "^$1=" "$tuple" 2>/dev/null | cut -d= -f2- || true
}

# preflight_candidate : in candidate mode, check the actual selected producer versions against what
# the consumer will resolve, BEFORE the smoke repository is deleted or anything is published.
#
# pure-java: released-8.1.properties must hold all three entries (no snapshot values), each equal to
# the producer's own version with any -SNAPSHOT suffix stripped; an environment override must repeat
# the file value exactly. (A matching snapshot producer still fails the consumer stage: the producer
# publishes -SNAPSHOT while the consumer expects the released version.)
# all: every producer must equal the consumer's effective expected version (the gradle.properties
# default or the environment override, -SNAPSHOT stripped) and must not itself be a snapshot:
# removing -SNAPSHOT from a consumer default must never disguise snapshot producer output.
preflight_candidate() {
  if [ "$PROFILE" = pure-java ]; then
    local tuple="${JMLX_SMOKE_TUPLE_FILE:-$SMOKE_DIR/released-8.1.properties}"
    [ -f "$tuple" ] || die "released tuple file missing: $tuple"
    local entry module key file_value producer override stripped
    for entry in jmlx-jinja:jmlxJinjaVersion:JMLX_JINJA_VERSION \
                 jmlx-tokenizer:jmlxTokenizerVersion:JMLX_TOKENIZER_VERSION \
                 jmlx-vision:jmlxVisionVersion:JMLX_VISION_VERSION; do
      module="${entry%%:*}"
      key="${entry#*:}"; key="${key%%:*}"
      override_var="${entry#*:}"; override_var="${override_var#*:}"
      file_value="$(tuple_value "$key")"
      [ -n "$file_value" ] || die "released-8.1.properties is missing entry $key"
      case "$file_value" in *-SNAPSHOT) die "released-8.1.properties entry $key is a snapshot: $file_value" ;; esac
      producer="$(producer_version "$module")"
      stripped="${producer%-SNAPSHOT}"
      [ "$stripped" = "$file_value" ] || die "producer $module version $producer does not match tuple value $file_value"
      override="${!override_var:-}"
      if [ -n "$override" ]; then
        [ "$override" = "$file_value" ] || die "$override_var=$override must repeat the tuple value $file_value"
      fi
      echo "preflight: $module=$file_value (released-8.1.properties; producer $producer)"
    done
  else
    local entry module key def override expected producer
    for entry in jmlx-models:jmlxModelsVersion:JMLX_MODELS_VERSION \
                 jmlx-native-macos-arm64:jmlxNativeVersion:JMLX_NATIVE_VERSION \
                 jmlx-core:jmlxCoreVersion:JMLX_CORE_VERSION \
                 jmlx-ffi:jmlxFfiVersion:JMLX_FFI_VERSION \
                 jmlx-tokenizer:jmlxTokenizerVersion:JMLX_TOKENIZER_VERSION \
                 jmlx-jinja:jmlxJinjaVersion:JMLX_JINJA_VERSION \
                 jmlx-vision:jmlxVisionVersion:JMLX_VISION_VERSION; do
      module="${entry%%:*}"
      key="${entry#*:}"; key="${key%%:*}"
      override_var="${entry#*:}"; override_var="${override_var#*:}"
      def="$(grep -m1 "^${key}=" "$SMOKE_DIR/gradle.properties" | cut -d= -f2- || true)"
      [ -n "$def" ] || die "consumer default $key missing from gradle.properties"
      override="${!override_var:-}"
      expected="${override:-$def}"
      expected="${expected%-SNAPSHOT}"
      producer="$(producer_version "$module")"
      [ "$producer" = "$expected" ] || die "producer $module version $producer != consumer expected $expected"
      case "$producer" in *-SNAPSHOT) die "producer $module is a snapshot: $producer" ;; esac
      echo "preflight: $module=$producer (producer matches consumer expected)"
    done
  fi
}

# main runs only when executed, not when sourced.
if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

smoke_paths
parse_args "$@"

# The profile's selected modules, in publish order (jinja before tokenizer: the tokenizer's POM
# depends on jinja, so the consumer cannot resolve it while jinja is missing from the repo).
if [ "$PROFILE" = pure-java ]; then
  modules="jmlx-jinja jmlx-tokenizer jmlx-vision"
  consumer_task=":pure-java:smoke"
else
  modules="jmlx-jinja jmlx-tokenizer jmlx-native-macos-arm64 jmlx-ffi jmlx-core jmlx-vision jmlx-models"
  consumer_task=":app:smoke"
fi

# Clean the consumer's project state after argument parsing and before any Gradle invocation
# (including the preflight): a fresh home alone leaves this intact, and a stale build directory from
# another mode could mask a failure.
rm -rf "$SMOKE_DIR/.gradle" "$SMOKE_DIR/build" "$SMOKE_DIR"/*/build

# Locate a Gradle distribution through the checkout's wrapper (default home), then run the consumer
# with that binary under a throwaway home. `--version` makes the wrapper download it if needed.
"$ROOT/gradlew" --version >/dev/null
wrapper_version="$(sed -n 's|.*gradle-\([0-9.]*\)-bin.zip.*|\1|p' "$ROOT/gradle/wrapper/gradle-wrapper.properties")"
dists_dir="${GRADLE_USER_HOME:-$HOME/.gradle}/wrapper/dists"
# || true: under `set -euo pipefail` a bare `ls ... | head -1` assignment exits the script on a
# missing distribution (the original failure mode), before the check below can report it.
gradle_bin="$(ls -d "$dists_dir"/gradle-"$wrapper_version"-bin/*/gradle-"$wrapper_version"/bin/gradle 2>/dev/null | head -1 || true)"
if [ -z "$gradle_bin" ] || [ ! -x "$gradle_bin" ]; then
  echo "error: no usable Gradle $wrapper_version distribution under $dists_dir" >&2
  echo "hint: the consumer must run with the checkout's wrapper distribution; run" >&2
  echo "      '$ROOT/gradlew --version' once from this checkout (with its default" >&2
  echo "      GRADLE_USER_HOME) so the wrapper downloads it" >&2
  exit 2
fi

args=()
case "$MODE" in
  ci|candidate)
    if [ "$MODE" = candidate ]; then
      preflight_candidate
    fi
    rm -rf "$REPO_DIR"
    tasks=()
    for m in $modules; do tasks+=(":$m:publishMavenPublicationToSmokeRepository"); done
    (cd "$ROOT" && ./gradlew -q "${tasks[@]}")
    args+=("-PjmlxRepoUrl=file://$REPO_DIR")
    if [ "$MODE" = candidate ]; then
      args+=("-PjmlxMode=release")
    else
      args+=("-PjmlxMode=ci")
    fi
    ;;
  central)
    args+=("-PjmlxMode=release")
    ;;
esac
args+=("-PsmokeProfile=$PROFILE")
[ "$RECORD" = true ] && args+=("-PsmokeRecord=true")

# Forward any version overrides (e.g. JMLX_MODELS_VERSION) as -P properties.
for pair in JMLX_MODELS_VERSION:jmlxModelsVersion JMLX_NATIVE_VERSION:jmlxNativeVersion \
  JMLX_CORE_VERSION:jmlxCoreVersion JMLX_FFI_VERSION:jmlxFfiVersion \
  JMLX_TOKENIZER_VERSION:jmlxTokenizerVersion JMLX_JINJA_VERSION:jmlxJinjaVersion \
  JMLX_VISION_VERSION:jmlxVisionVersion; do
  var="${pair%%:*}"
  [ -n "${!var:-}" ] && args+=("-P${pair#*:}=${!var}")
done

home="$(mktemp -d)"
trap 'rm -rf "$home"' EXIT

env -u JMLX_LIBRARY_PATH GRADLE_USER_HOME="$home" \
  "$gradle_bin" --project-dir "$SMOKE_DIR" --no-daemon --no-build-cache "${args[@]}" "$consumer_task"
