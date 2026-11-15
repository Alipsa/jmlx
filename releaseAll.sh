#!/usr/bin/env bash
# Publishes every eligible, unreleased, non-SNAPSHOT Gradle publication in this repository as
# one Maven-layout ZIP and one automatic Central Portal deployment. See
# tools/release-all/README.md and req/phase8-1-release-checklist.md. The per-module release.sh
# scripts remain available for individual releases. No version is rewritten, no baseline is
# advanced, and no Git tag is created by this script.
set -euo pipefail
RELEASE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$RELEASE_ROOT"
source "$RELEASE_ROOT/tools/release-all/preflight.sh"
usage() {
  echo 'usage: ./releaseAll.sh [--dry-run|--bundle-only|--publish-verified receipt|--resume deployment-id]' >&2
  exit 2
}
mode=release
resume=
verified_receipt=
case "${1:-}" in
  '') [[ $# == 0 ]] || usage ;;
  --dry-run) mode=plan; [[ $# == 1 ]] || usage ;;
  --bundle-only) mode=bundle; [[ $# == 1 ]] || usage ;;
  --resume) mode=resume; [[ $# == 2 && -n "$2" ]] || usage; resume=$2 ;;
  --publish-verified) mode=verified; [[ $# == 2 && -n "$2" ]] || usage; verified_receipt=$2 ;;
  *) usage ;;
esac
release_tools_preflight "$mode"
gradle=("$RELEASE_ROOT/gradlew" -p "$RELEASE_ROOT" -PreleaseAllMode=true)
if [[ "$mode" != resume ]]; then
  gradle+=("-Dorg.gradle.java.installations.paths=$RELEASE_JAVA_HOME")
fi
if [[ "$mode" == resume ]]; then
  "${gradle[@]}" releaseAll "-PreleaseAllResume=$resume"
  exit
fi
if [[ "$mode" == release ]]; then
  echo 'Use --publish-verified <final-receipt> after exact-bundle review and publication authorization.' >&2
  exit 1
fi
if [[ "$mode" == verified ]]; then
  [[ -z "$(git status --porcelain)" ]] || { echo 'Verified publication requires a clean tree' >&2; exit 1; }
  python3 "$RELEASE_ROOT/tools/release-all/final_receipt.py" --root "$RELEASE_ROOT" --verify "$verified_receipt"
  "${gradle[@]}" -PreleaseAllUseExistingBundle=true "-PreleaseAllVerifiedReceipt=$verified_receipt" releaseAll
  exit
fi
# The plan records HEAD and the guard rechecks it before any staging: a dirty tree (including
# untracked files) or a new commit makes the manifest stale.
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit or stash changes before release planning' >&2; exit 1; }
"${gradle[@]}" releaseAllPlan
manifest="$RELEASE_ROOT/build/releaseAll/manifest.json"
# The expected selection comes from the same reviewed manifest as candidate smoke. Reused
# Central coordinates must never enter staging or spend a deployment slot.
selected=$(groovy -e 'println new groovy.json.JsonSlurper().parse(new File(args[0])).selected.collect { "${it.artifactId}:${it.version}" }.sort().join(",")' "$manifest")
EXPECTED_SELECTION=$(python3 "$RELEASE_ROOT/tools/release-all/release_inputs.py" select \
  "$RELEASE_ROOT/tools/release-smoke/released-8.4.json" | \
  awk -F '\t' '$3 == "local" {print $1 ":" $2}' | sort | paste -sd, -)
[[ -n "$EXPECTED_SELECTION" ]] || { echo 'Refusing an empty release selection' >&2; exit 1; }
[[ "$selected" == "$EXPECTED_SELECTION" ]] || {
  echo "Selected components '$selected' differ from expected '$EXPECTED_SELECTION'; see build/releaseAll/report.txt" >&2
  exit 1
}
if [[ "$mode" == plan ]]; then
  # Report mode: missing signing configuration is a warning, not a failure.
  "${gradle[@]}" -PreleaseAllSigningReport=true releaseAllSigningCheck
  echo 'Dry run completed; signing configuration will be checked during preparation.'
  exit
fi
# Signing runs before native/model/adapter verification; all selected modules participate.
"${gradle[@]}" releaseAllSigningCheck
if [[ "$mode" == bundle ]]; then
  [[ -n "${JMLX_RELEASE_CANDIDATE_TAG:-}" ]] || {
    echo 'Set JMLX_RELEASE_CANDIDATE_TAG to the reviewed annotated candidate tag' >&2; exit 1;
  }
  invocation=$(python3 -c 'import uuid; print(uuid.uuid4())')
  evidence="$RELEASE_ROOT/build/releaseAll/task-evidence-$invocation.json"
  required=$(python3 "$RELEASE_ROOT/tools/release-all/verify_native_tests.py" --tasks)
  test_args=()
  IFS=, read -r -a test_paths <<< "$required"
  for task in "${test_paths[@]}"; do
    # Required paths are from the committed inventory, never user-supplied cleanup targets.
    module="${task#:}"; module="${module%%:*}"
    task_name="${task##*:}"
    rm -rf "$RELEASE_ROOT/$module/build/test-results/$task_name"
    test_args+=("$task" --rerun)
  done
  before=$(shasum -a 256 "$manifest" | cut -d' ' -f1)
  "${gradle[@]}" "${test_args[@]}" bundleAll --no-build-cache \
    --init-script "$RELEASE_ROOT/tools/release-all/task-evidence.init.gradle" \
    "-PreleaseRequiredTasks=$required" "-PreleaseInvocationId=$invocation" "-PreleaseEvidenceFile=$evidence"
  after=$(shasum -a 256 "$manifest" | cut -d' ' -f1)
  [[ "$before" == "$after" ]] || { echo 'Plan manifest changed during bundling' >&2; exit 1; }
  python3 "$RELEASE_ROOT/tools/release-all/preliminary_record.py" \
    --root "$RELEASE_ROOT" --tag "$JMLX_RELEASE_CANDIDATE_TAG" \
    --events "$evidence" --invocation "$invocation" \
    --output "$RELEASE_ROOT/build/releaseAll/preliminary-$invocation.json"
fi
