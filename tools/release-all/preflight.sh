#!/usr/bin/env bash
# Shared tool preflight for the releaseAll workflow. It checks tools only; it does not create
# or remove files. Maven is not required: the plugin's releaseAllPrepareMaven task runs only
# for consumers with static Maven (BOM/aggregate) candidates, and jmlx has none.
release_tools_preflight() {
  (( BASH_VERSINFO[0] >= 4 )) || { echo 'Use Bash 4+ (macOS: brew install bash)' >&2; return 1; }
  local tool
  for tool in java groovy git; do
    command -v "$tool" >/dev/null || { echo "Required tool missing: $tool" >&2; return 1; }
  done
  # The launcher may be Java 21; native verification executes with a Java 25 toolchain.
  local java_version
  java_version=$(java -version 2>&1)
  if [[ ! "$java_version" =~ version\ \"([0-9]+) ]]; then
    echo 'Java 21+ is required' >&2; return 1
  fi
  (( BASH_REMATCH[1] >= 21 )) || { echo 'Java 21+ is required' >&2; return 1; }
  [ "${1:-}" = resume ] && return 0
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
    echo 'The Phase 8.4 native release requires macOS ARM64' >&2; return 1;
  }
  command -v python3 >/dev/null || { echo 'Python 3 is required for release evidence' >&2; return 1; }
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required for release evidence")' || return 1
  local release_jdk="${JMLX_RELEASE_JAVA_HOME:-${JAVA_HOME:-}}"
  # Keep selection explicit: Java 21 on PATH is not evidence that Java 25 tests can execute.
  [[ -x "$release_jdk/bin/java" ]] || { echo 'Set JMLX_RELEASE_JAVA_HOME to a Java 25 installation' >&2; return 1; }
  java_version=$("$release_jdk/bin/java" -version 2>&1)
  [[ "$java_version" =~ version\ \"25[.\"] ]] || { echo 'JMLX_RELEASE_JAVA_HOME must provide Java 25' >&2; return 1; }
  RELEASE_JAVA_HOME="$release_jdk"
  local release_jdk21="${JMLX_RELEASE_JAVA21_HOME:-${JAVA_HOME:-}}"
  if [[ ! -x "$release_jdk21/bin/java" ]] || ! "$release_jdk21/bin/java" -version 2>&1 | grep -q 'version "21[."]'; then
    echo 'Set JMLX_RELEASE_JAVA21_HOME to a Java 21 installation for pure-Java release checks' >&2
    return 1
  fi
  RELEASE_JAVA21_HOME="$release_jdk21"
  python3 "$RELEASE_ROOT/tools/release-all/runtime_evidence.py" native >/dev/null || return 1
  python3 "$RELEASE_ROOT/tools/release-all/release_inputs.py" native \
    "$RELEASE_ROOT/native/install/lib" --bootstrap "$RELEASE_ROOT/scripts/bootstrap-native.sh" \
    --forbid "$HOME" --forbid "$RELEASE_ROOT" >/dev/null
}
