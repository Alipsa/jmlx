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
  echo 'usage: ./releaseAll.sh [--dry-run|--bundle-only|--resume deployment-id]' >&2
  exit 2
}
mode=release
resume=
case "${1:-}" in
  '') [[ $# == 0 ]] || usage ;;
  --dry-run) mode=plan; [[ $# == 1 ]] || usage ;;
  --bundle-only) mode=bundle; [[ $# == 1 ]] || usage ;;
  --resume) mode=resume; [[ $# == 2 && -n "$2" ]] || usage; resume=$2 ;;
  *) usage ;;
esac
release_tools_preflight
gradle=("$RELEASE_ROOT/gradlew" -p "$RELEASE_ROOT" -PreleaseAllMode=true)
if [[ "$mode" == resume ]]; then
  "${gradle[@]}" releaseAll "-PreleaseAllResume=$resume"
  exit
fi
# The plan records HEAD and the guard rechecks it before any staging: a dirty tree (including
# untracked files) or a new commit makes the manifest stale.
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit or stash changes before release planning' >&2; exit 1; }
"${gradle[@]}" releaseAllPlan
manifest="$RELEASE_ROOT/build/releaseAll/manifest.json"
selected=$(groovy -e 'println new groovy.json.JsonSlurper().parse(new File(args[0])).selected.size()' "$manifest")
# Exact eligibility gate: the three pure-Java modules are the only selected candidates; the
# four native-chain SNAPSHOT modules are reported excluded with their SNAPSHOT reasons in
# report.txt (a SNAPSHOT dependency would exclude the depending module from the plan itself).
[[ "$selected" == 3 ]] || {
  echo "Expected 3 selected components, plan selected $selected; see build/releaseAll/report.txt" >&2
  exit 1
}
if [[ "$mode" == plan ]]; then
  # Report mode: missing signing configuration is a warning, not a failure.
  "${gradle[@]}" -PreleaseAllSigningReport=true releaseAllSigningCheck
  echo 'Dry run completed; signing configuration will be checked during preparation.'
  exit
fi
# Strict signing preflight, before the selected modules' checks run: bundleAll's staging
# wiring makes each selected module's publish/sign tasks depend on this check and its check
# task depend on the source guard (the plugin wires both). The SNAPSHOT consumer
# (:jmlx-models) is not part of this flow; the runbook's root ./gradlew check covers it.
"${gradle[@]}" releaseAllSigningCheck
"${gradle[@]}" bundleAll
[[ "$mode" == bundle ]] || "${gradle[@]}" releaseAll
