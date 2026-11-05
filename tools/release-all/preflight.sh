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
  # The Gradle daemon runs on the java on PATH. The releaseAll workflow only executes the
  # pure-Java modules' tasks (Java 21 toolchain), so 21+ is sufficient; the native-chain
  # modules are configured but not built.
  local java_version
  java_version=$(java -version 2>&1)
  if [[ ! "$java_version" =~ version\ \"([0-9]+) ]]; then
    echo 'Java 21+ is required' >&2; return 1
  fi
  (( BASH_REMATCH[1] >= 21 )) || { echo 'Java 21+ is required' >&2; return 1; }
}
