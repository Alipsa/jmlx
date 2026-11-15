#!/usr/bin/env bash
# Runs the release smoke consumer against the selected jmlx modules.
# Modes: ci publishes checkout snapshots locally; candidate verifies the 8.4 source manifest;
# central resolves released artifacts without publishing. Profiles: all (nine modules, Java 25,
# macOS ARM64) and pure-java (three modules, Java 21, headless). Historical 8.1 verification is
# central --profile pure-java --release 8.1. Release modes need Python 3.11+.
# --bundle ZIP --bundle-record JSON --bundle-record-sha256 HASH verifies an already signed ZIP
# without invoking producer tasks. --record is restricted to ci --profile all.
# Sourcing this file defines the legacy parser/preflight test helpers without executing main.
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

# parse_args <args...> : sets MODE, PROFILE, RECORD. Rejects unknown modes/profiles, duplicate,
# empty or conflicting arguments and any unused trailing argument, and --record outside
# ci --profile all.
parse_args() {
  local mode="" profile="" profile_set=false record="false" release="" bundle="" bundle_record="" record_hash=""
  while [ $# -gt 0 ]; do
    case "$1" in
      ci|candidate|central)
        [ -z "$mode" ] || die "duplicate mode '$1'"
        mode="$1"
        ;;
      --profile)
        [ $# -ge 2 ] || die "--profile needs a value"
        [ "$profile_set" = false ] || die "duplicate --profile"
        profile_set=true
        profile="$2"
        [ -n "$profile" ] || die "--profile needs a value"
        shift
        ;;
      --profile=*)
        [ "$profile_set" = false ] || die "duplicate --profile"
        profile_set=true
        profile="${1#--profile=}"
        [ -n "$profile" ] || die "--profile needs a value"
        ;;
      --record)
        [ "$record" = false ] || die "duplicate --record"
        record="true"
        ;;
      --release|--bundle|--bundle-record|--bundle-record-sha256)
        [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value"
        case "$1" in
          --release) [ -z "$release" ] || die 'duplicate --release'; release="$2" ;;
          --bundle) [ -z "$bundle" ] || die 'duplicate --bundle'; bundle="$2" ;;
          --bundle-record) [ -z "$bundle_record" ] || die 'duplicate --bundle-record'; bundle_record="$2" ;;
          --bundle-record-sha256) [ -z "$record_hash" ] || die 'duplicate --bundle-record-sha256'; record_hash="$2" ;;
        esac
        shift
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
  RELEASE="${release:-8.4}"
  case "$RELEASE" in 8.1|8.4) ;; *) die 'unknown release' ;; esac
  if [ "$MODE" = ci ] && [ -n "$release$bundle$bundle_record$record_hash" ]; then
    die 'release/bundle inputs are not accepted in CI mode'
  fi
  if [ "$RELEASE" = 8.1 ] && { [ "$MODE" != central ] || [ "$PROFILE" != pure-java ]; }; then
    die '--release 8.1 is historical central --profile pure-java only'
  fi
  if [ -n "$bundle$bundle_record$record_hash" ]; then
    [ "$MODE" = candidate ] && [ "$RELEASE" = 8.4 ] || die 'bundle inputs require candidate 8.4'
    [ -n "$bundle" ] && [ -n "$bundle_record" ] && [ -n "$record_hash" ] || die 'all bundle inputs are required'
  fi
  BUNDLE="$bundle"
  BUNDLE_RECORD="$bundle_record"
  BUNDLE_RECORD_SHA256="$record_hash"
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
# test hook overrides the path). The file is tightly formatted -- exactly <key>=<value> with no
# whitespace; a value containing whitespace is a clear error here, matching the Gradle reader
# (which also rejects it) rather than silently trimming.
tuple_value() {
  local tuple="${JMLX_SMOKE_TUPLE_FILE:-$SMOKE_DIR/released-8.1.properties}" v
  v="$(grep -m1 "^$1=" "$tuple" 2>/dev/null | cut -d= -f2- || true)"
  case "$v" in
    # %q: a CRLF-rewritten value carries a non-printing \r that would make the error unreadable.
    *[[:space:]]*) die "released-8.1.properties entry $1 contains whitespace: $(printf '%q' "$v") (the file is tightly formatted)" ;;
  esac
  printf '%s\n' "$v"
}

# preflight_candidate : in candidate mode, check the actual selected producer versions against what
# the consumer will resolve. main runs it immediately after argument parsing, so a failure exits
# before any side effect: no cleanup, no Gradle invocation, no repository deletion, no publish.
#
# pure-java: released-8.1.properties must hold all three entries (no snapshot values), and the
# producer's own version must equal the file value and be a release version (candidate never
# publishes snapshot output, so a snapshot producer fails here, not later at consumer resolution);
# an environment override must repeat the file value exactly.
# all: every producer must equal the consumer's effective expected version (the gradle.properties
# default or the environment override, -SNAPSHOT stripped) and must not itself be a snapshot:
# removing -SNAPSHOT from a consumer default must never disguise snapshot producer output.
preflight_candidate() {
  if [ "$PROFILE" = pure-java ]; then
    local tuple="${JMLX_SMOKE_TUPLE_FILE:-$SMOKE_DIR/released-8.1.properties}"
    [ -f "$tuple" ] || die "released tuple file missing: $tuple"
    local entry module key file_value producer override override_var
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
      case "$producer" in *-SNAPSHOT) die "producer $module is a snapshot: $producer (candidate requires release producers)" ;; esac
      [ "$producer" = "$file_value" ] || die "producer $module version $producer does not match tuple value $file_value"
      override="${!override_var:-}"
      if [ -n "$override" ]; then
        [ "$override" = "$file_value" ] || die "$override_var=$override must repeat the tuple value $file_value"
      fi
      echo "preflight: $module=$file_value (released-8.1.properties; producer $producer)"
    done
  else
    local entry module key def override override_var expected producer
    for entry in jmlx-spring-ai:jmlxSpringAiVersion:JMLX_SPRING_AI_VERSION \
                 jmlx-langchain4j:jmlxLangchain4jVersion:JMLX_LANGCHAIN4J_VERSION \
                 jmlx-models:jmlxModelsVersion:JMLX_MODELS_VERSION \
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

# Candidate mode checks the actual producer versions against what the consumer will resolve
# immediately after argument parsing, before any side effect: a failing preflight exits with the
# smoke repository and the consumer's state untouched.
if [ "$MODE" = candidate ]; then
  if [ -n "$BUNDLE" ]; then
    python3 "$ROOT/tools/release-all/release_inputs.py" bundle "$BUNDLE" \
      --record "$BUNDLE_RECORD" --record-sha256 "$BUNDLE_RECORD_SHA256" \
      --manifest "$SMOKE_DIR/released-8.4.json" --root "$ROOT"
  elif [ "$RELEASE" = 8.4 ]; then
    python3 "$ROOT/tools/release-all/release_inputs.py" preflight \
      "$SMOKE_DIR/released-8.4.json" --profile "$PROFILE" --root "$ROOT"
  else
    preflight_candidate
  fi
fi

# The profile's selected modules, in publish order (jinja before tokenizer: the tokenizer's POM
# depends on jinja, so the consumer cannot resolve it while jinja is missing from the repo).
if [ "$PROFILE" = pure-java ]; then
  modules="jmlx-jinja jmlx-tokenizer jmlx-vision"
  consumer_task=":pure-java:smoke"
else
  modules="jmlx-jinja jmlx-tokenizer jmlx-native-macos-arm64 jmlx-ffi jmlx-core jmlx-vision jmlx-models jmlx-spring-ai jmlx-langchain4j"
  consumer_task=":app:smoke"
fi

if [ "$MODE" != ci ] && [ "$RELEASE" = 8.4 ]; then
  selection="$(python3 "$ROOT/tools/release-all/release_inputs.py" select \
    "$SMOKE_DIR/released-8.4.json" --profile "$PROFILE")"
  modules="$(printf '%s\n' "$selection" | awk -F '\t' '$3 == "local" {print $1}')"
fi

# Clean the consumer's project state after argument parsing and before any Gradle invocation:
# a fresh home alone leaves this intact, and a stale build directory from another mode could mask
# a failure.
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
    if [ -n "$BUNDLE" ]; then
      REPO_DIR="$(mktemp -d)/repository"
      python3 "$ROOT/tools/release-all/release_inputs.py" bundle "$BUNDLE" \
        --record "$BUNDLE_RECORD" --record-sha256 "$BUNDLE_RECORD_SHA256" \
        --manifest "$SMOKE_DIR/released-8.4.json" --root "$ROOT" --extract "$REPO_DIR"
    else
      rm -rf "$REPO_DIR"
      tasks=()
      for m in $modules; do tasks+=(":$m:publishMavenPublicationToSmokeRepository"); done
      if [ ${#tasks[@]} -gt 0 ]; then
        (cd "$ROOT" && ./gradlew -q "${tasks[@]}")
      fi
    fi
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
args+=("-PsmokeRelease=$RELEASE" "-PsmokeSourceMode=$MODE")
[ "$RECORD" = true ] && args+=("-PsmokeRecord=true")

# Forward any version overrides (e.g. JMLX_MODELS_VERSION) as -P properties.
for pair in JMLX_SPRING_AI_VERSION:jmlxSpringAiVersion JMLX_LANGCHAIN4J_VERSION:jmlxLangchain4jVersion \
  JMLX_MODELS_VERSION:jmlxModelsVersion JMLX_NATIVE_VERSION:jmlxNativeVersion \
  JMLX_CORE_VERSION:jmlxCoreVersion JMLX_FFI_VERSION:jmlxFfiVersion \
  JMLX_TOKENIZER_VERSION:jmlxTokenizerVersion JMLX_JINJA_VERSION:jmlxJinjaVersion \
  JMLX_VISION_VERSION:jmlxVisionVersion; do
  var="${pair%%:*}"
  [ -n "${!var:-}" ] && args+=("-P${pair#*:}=${!var}")
done

home="$(mktemp -d)"
trap 'rm -rf "$home"; if [ -n "$BUNDLE" ]; then rm -rf "${REPO_DIR%/repository}"; fi' EXIT

env -u JMLX_LIBRARY_PATH GRADLE_USER_HOME="$home" \
  "$gradle_bin" --project-dir "$SMOKE_DIR" --no-daemon --no-build-cache "${args[@]}" "$consumer_task"
