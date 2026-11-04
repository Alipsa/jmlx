#!/usr/bin/env bash
# Shared tool and portable-filesystem preflight for the releaseAll workflow.
# It checks tools and path sanity only; it does not create or remove files.
release_tools_preflight() {
  (( BASH_VERSINFO[0] >= 4 )) || { echo 'Use Bash 4+ (macOS: brew install bash)' >&2; return 1; }
  local tool
  for tool in java mvn groovy grep git; do
    command -v "$tool" >/dev/null || { echo "Required tool missing: $tool" >&2; return 1; }
  done
  # The Gradle daemon runs on the java on PATH. The releaseAll workflow only executes the
  # pure-Java modules' tasks (Java 21 toolchain), so 21+ is sufficient; the native-chain
  # modules are configured but not built.
  local java_version
  java_version=$(java -version 2>&1)
  if [[ ! "$java_version" =~ version\ \"([0-9]+) ]]; then
    echo 'Java 21+ is required' >&2; return 1
  fi
  (( BASH_REMATCH[1] >= 21 )) || { echo 'Java 21+ is required' >&2; return 1; }
  # mvn is required by the plugin's PrepareMavenTask contract even when the plan selects no
  # static Maven component (jmlx's case): the task runs the settings helper in every mode.
  local maven_version
  maven_version=$(mvn --version)
  maven_version=${maven_version%%$'\n'*}
  [[ "$maven_version" =~ Apache\ Maven\ ([0-9]+)\.([0-9]+)\.([0-9]+) ]] || {
    echo 'Could not determine Maven version' >&2; return 1
  }
  (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && (BASH_REMATCH[2] > 9 || (BASH_REMATCH[2] == 9 && BASH_REMATCH[3] >= 9))) )) || {
    echo 'Maven 3.9.9+ is required' >&2; return 1;
  }
  RELEASE_PATHS_STRICT_COUNT=0 groovy "$RELEASE_ROOT/tools/release-all/Paths.groovy" "$RELEASE_ROOT" "$HOME" "$HOME/.m2/repository" >/dev/null
}
