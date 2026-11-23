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
  echo 'usage: ./releaseAll.sh [--dry-run|--bundle-only|--publish-verified receipt --acceptance record --acceptance-sha256 digest|--resume deployment-id]' >&2
  exit 2
}
mode=release
resume=
verified_receipt=
acceptance_record=
acceptance_sha=
case "${1:-}" in
  '') [[ $# == 0 ]] || usage ;;
  --dry-run) mode=plan; [[ $# == 1 ]] || usage ;;
  --bundle-only) mode=bundle; [[ $# == 1 ]] || usage ;;
  --resume) mode=resume; [[ $# == 2 && -n "$2" ]] || usage; resume=$2 ;;
  --publish-verified)
    mode=verified
    [[ $# == 6 && -n "$2" && "$3" == --acceptance && -n "$4" &&
       "$5" == --acceptance-sha256 && "$6" =~ ^[0-9a-f]{64}$ ]] || usage
    verified_receipt=$2; acceptance_record=$4; acceptance_sha=$6 ;;
  *) usage ;;
esac
release_tools_preflight "$mode"
gradle=("$RELEASE_ROOT/gradlew" -p "$RELEASE_ROOT" -PreleaseAllMode=true)
if [[ "$mode" != resume ]]; then
  gradle=(env "JAVA_HOME=$RELEASE_JAVA_HOME" "${gradle[@]}")
  gradle+=("-Dorg.gradle.java.installations.paths=$RELEASE_JAVA_HOME,$RELEASE_JAVA21_HOME"
    -Dorg.gradle.java.installations.auto-detect=false
    -Dorg.gradle.java.installations.auto-download=false
    -Dorg.gradle.java.installations.fromEnv=)
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
  verified_hashes=$(python3 "$RELEASE_ROOT/tools/release-all/final_receipt.py" \
    --root "$RELEASE_ROOT" --verify "$verified_receipt" --plugin-digests \
    --acceptance "$acceptance_record" --acceptance-sha256 "$acceptance_sha")
  IFS=$'\t' read -r verified_bundle_sha verified_manifest_sha <<< "$verified_hashes"
  [[ "$verified_bundle_sha" =~ ^[0-9a-f]{64}$ && "$verified_manifest_sha" =~ ^[0-9a-f]{64}$ &&
     "$verified_hashes" == "$verified_bundle_sha"$'\t'"$verified_manifest_sha" ]] || {
    echo 'Receipt verifier returned malformed digests' >&2; exit 1;
  }
  "${gradle[@]}" -PreleaseAllUseExistingBundle=true "-PreleaseAllVerifiedReceipt=$verified_receipt" \
    "-PreleaseAllOwnerAcceptance=$acceptance_record" \
    "-PreleaseAllOwnerAcceptanceSha256=$acceptance_sha" \
    "-PreleaseAllExpectedBundleSha256=$verified_bundle_sha" \
    "-PreleaseAllExpectedManifestSha256=$verified_manifest_sha" releaseAll
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
  required_args=()
  IFS=, read -r -a required_paths <<< "$required"
  for task in "${required_paths[@]}"; do
    # The committed list includes native tests and payload verification, never user-supplied
    # cleanup targets. Non-test tasks have no result directory, so removal is a no-op.
    module="${task#:}"; module="${module%%:*}"
    task_name="${task##*:}"
    rm -rf "$RELEASE_ROOT/$module/build/test-results/$task_name"
    required_args+=("$task" --rerun)
  done
  before=$(shasum -a 256 "$manifest" | cut -d' ' -f1)
  bundle_command=("${gradle[@]}" "${required_args[@]}" bundleAll --no-build-cache
    --init-script "$RELEASE_ROOT/tools/release-all/task-evidence.init.gradle"
    "-PreleaseRequiredTasks=$required" "-PreleaseInvocationId=$invocation" "-PreleaseEvidenceFile=$evidence")
  argv_file="$RELEASE_ROOT/build/releaseAll/gradle-argv-$invocation.json"
  python3 - "$argv_file" "${bundle_command[@]}" <<'PY'
import json, sys
with open(sys.argv[1], 'x') as stream:
    json.dump(sys.argv[2:], stream, indent=2)
    stream.write('\n')
PY
  "${bundle_command[@]}"
  after=$(shasum -a 256 "$manifest" | cut -d' ' -f1)
  [[ "$before" == "$after" ]] || { echo 'Plan manifest changed during bundling' >&2; exit 1; }
  python3 "$RELEASE_ROOT/tools/release-all/preliminary_record.py" \
    --root "$RELEASE_ROOT" --tag "$JMLX_RELEASE_CANDIDATE_TAG" \
    --events "$evidence" --invocation "$invocation" \
    --argv-json "$argv_file" --java-home "$RELEASE_JAVA_HOME" --java21-home "$RELEASE_JAVA21_HOME" \
    --output "$RELEASE_ROOT/build/releaseAll/preliminary-$invocation.json"
fi
