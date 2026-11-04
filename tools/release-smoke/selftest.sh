#!/usr/bin/env bash
# Regression tests for run.sh's argument parsing and candidate preflight. Everything runs in
# subshells against disposable fixtures (JMLX_SMOKE_TUPLE_FILE, JMLX_SMOKE_PRODUCER_VERSIONS,
# JMLX_SMOKE_REPO_DIR), so the real smoke repository, producer versions and tuple file are never
# touched. Run it with ./tools/release-smoke/selftest.sh; CI's Java job runs it as a step.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

# expect <description> <expected-status> <command...>
expect() {
  local desc="$1" want="$2" status=0
  shift 2
  "$@" >/dev/null 2>&1 || status=$?
  if [ "$status" -eq "$want" ]; then
    echo "ok: $desc"
  else
    echo "FAIL: $desc (got $status, want $want)"
    fail=1
  fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Source run.sh in a subshell and run a body against it. The body may export the test hooks first.
body() {
  bash -c "source '$here/run.sh'; smoke_paths; $1"
}

# --- parse_args: valid combinations ---
expect "mode only defaults to all" 0 body 'parse_args ci; [ "$MODE" = ci ] && [ "$PROFILE" = all ] && [ "$RECORD" = false ]'
expect "explicit all profile" 0 body 'parse_args candidate --profile all; [ "$MODE" = candidate ] && [ "$PROFILE" = all ]'
expect "pure profile" 0 body 'parse_args ci --profile pure-java; [ "$PROFILE" = pure-java ]'
expect "equals-form profile" 0 body 'parse_args central --profile=pure-java; [ "$MODE" = central ] && [ "$PROFILE" = pure-java ]'
expect "record on ci all" 0 body 'parse_args ci --record; [ "$RECORD" = true ]'

# --- parse_args: rejections ---
expect "missing mode" 2 body 'parse_args'
expect "unknown mode" 2 body 'parse_args bogus'
expect "duplicate mode" 2 body 'parse_args ci candidate'
expect "unknown profile" 2 body 'parse_args ci --profile bogus'
expect "profile without value" 2 body 'parse_args ci --profile'
expect "duplicate profile" 2 body 'parse_args ci --profile all --profile pure-java'
expect "unexpected trailing argument" 2 body 'parse_args ci extra'
expect "duplicate record" 2 body 'parse_args ci --record --record'
expect "record on candidate" 2 body 'parse_args candidate --record'
expect "record on central" 2 body 'parse_args central --record'
expect "record on pure profile" 2 body 'parse_args ci --profile pure-java --record'

# --- preflight_candidate: pure profile ---
tuple="$tmp/released.properties"
producers="$tmp/producers"
printf 'jmlxJinjaVersion=0.6.0\njmlxTokenizerVersion=0.1.0\njmlxVisionVersion=0.1.0\n' > "$tuple"
printf 'jmlx-jinja=0.6.0-SNAPSHOT\njmlx-tokenizer=0.1.0-SNAPSHOT\njmlx-vision=0.1.0-SNAPSHOT\n' > "$producers"

expect "pure preflight passes (snapshot producers strip to the tuple)" 0 body \
  "export JMLX_SMOKE_TUPLE_FILE='$tuple' JMLX_SMOKE_PRODUCER_VERSIONS='$producers'; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate"
expect "pure preflight passes with a repeating override" 0 body \
  "export JMLX_SMOKE_TUPLE_FILE='$tuple' JMLX_SMOKE_PRODUCER_VERSIONS='$producers' JMLX_JINJA_VERSION=0.6.0; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate"
expect "pure preflight rejects a missing tuple file" 2 body \
  "export JMLX_SMOKE_TUPLE_FILE='$tmp/absent.properties' JMLX_SMOKE_PRODUCER_VERSIONS='$producers'; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate"
expect "pure preflight rejects a missing tuple entry" 2 body \
  "printf 'jmlxJinjaVersion=0.6.0\njmlxTokenizerVersion=0.1.0\n' > '$tmp/missing-entry.properties'; export JMLX_SMOKE_TUPLE_FILE='$tmp/missing-entry.properties' JMLX_SMOKE_PRODUCER_VERSIONS='$producers'; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate"
expect "pure preflight rejects a snapshot tuple entry" 2 body \
  "printf 'jmlxJinjaVersion=0.6.0-SNAPSHOT\njmlxTokenizerVersion=0.1.0\njmlxVisionVersion=0.1.0\n' > '$tmp/snapshot-tuple.properties'; export JMLX_SMOKE_TUPLE_FILE='$tmp/snapshot-tuple.properties' JMLX_SMOKE_PRODUCER_VERSIONS='$producers'; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate"
expect "pure preflight rejects a mismatched producer" 2 body \
  "printf 'jmlx-jinja=0.7.0-SNAPSHOT\njmlx-tokenizer=0.1.0-SNAPSHOT\njmlx-vision=0.1.0-SNAPSHOT\n' > '$tmp/mismatched.properties'; export JMLX_SMOKE_TUPLE_FILE='$tuple' JMLX_SMOKE_PRODUCER_VERSIONS='$tmp/mismatched.properties'; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate"
expect "pure preflight rejects a non-repeating override" 2 body \
  "export JMLX_SMOKE_TUPLE_FILE='$tuple' JMLX_SMOKE_PRODUCER_VERSIONS='$producers' JMLX_JINJA_VERSION=0.7.0; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate"

# --- preflight_candidate: all profile (consumer defaults are read from the real gradle.properties) ---
all_producers="$tmp/all-producers"
printf 'jmlx-models=0.1.0\njmlx-native-macos-arm64=0.1.0\njmlx-core=0.5.0\njmlx-ffi=0.5.0\njmlx-tokenizer=0.1.0\njmlx-jinja=0.6.0\njmlx-vision=0.1.0\n' > "$all_producers"

expect "all preflight passes (release producers match stripped defaults)" 0 body \
  "export JMLX_SMOKE_PRODUCER_VERSIONS='$all_producers'; MODE=candidate PROFILE=all RECORD=false; preflight_candidate"
expect "all preflight passes with a matching override" 0 body \
  "export JMLX_SMOKE_PRODUCER_VERSIONS='$all_producers' JMLX_CORE_VERSION=0.5.0; MODE=candidate PROFILE=all RECORD=false; preflight_candidate"
expect "all preflight rejects a snapshot producer" 2 body \
  "printf 'jmlx-models=0.1.0\njmlx-native-macos-arm64=0.1.0\njmlx-core=0.5.0\njmlx-ffi=0.5.0\njmlx-tokenizer=0.1.0\njmlx-jinja=0.6.0-SNAPSHOT\njmlx-vision=0.1.0\n' > '$tmp/all-snapshot.properties'; export JMLX_SMOKE_PRODUCER_VERSIONS='$tmp/all-snapshot.properties'; MODE=candidate PROFILE=all RECORD=false; preflight_candidate"
expect "all preflight rejects an override that moves the expectation" 2 body \
  "export JMLX_SMOKE_PRODUCER_VERSIONS='$all_producers' JMLX_CORE_VERSION=0.4.0; MODE=candidate PROFILE=all RECORD=false; preflight_candidate"

# --- a failing preflight is side-effect-free: the smoke repository is untouched ---
repo="$tmp/repo"
mkdir -p "$repo"
touch "$repo/marker"
body "export JMLX_SMOKE_REPO_DIR='$repo' JMLX_SMOKE_TUPLE_FILE='$tmp/absent.properties' JMLX_SMOKE_PRODUCER_VERSIONS='$producers'; MODE=candidate PROFILE=pure-java RECORD=false; preflight_candidate" >/dev/null 2>&1 || true
if [ -f "$repo/marker" ]; then
  echo "ok: failing preflight leaves the smoke repository untouched"
else
  echo "FAIL: preflight touched the smoke repository"
  fail=1
fi

# --- task input isolation: the verification tasks declare mode, profile, versions, tuple and repo ---
for needle in "inputs.property('mode'" "inputs.property('profile'" "inputs.property('versions'" \
  "inputs.property('repo'" "inputs.file(tupleFile)"; do
  if grep -qF "$needle" "$here/smoke-conventions.gradle"; then
    echo "ok: verification task inputs declare $needle"
  else
    echo "FAIL: smoke-conventions.gradle does not declare $needle"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "selftest FAILED" >&2
  exit 1
fi
echo "selftest passed"
