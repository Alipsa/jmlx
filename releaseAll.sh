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
# Optional local testing of nexus-release-plugin changes. The release-mode settings gate
# resolves the plugin exclusively from the Gradle Plugin Portal unless this override is set.
if [[ -n "${RELEASE_ALL_PLUGIN_DIR:-}" ]]; then
  [[ -f "$RELEASE_ALL_PLUGIN_DIR/src/main/groovy/se/alipsa/gradle/plugin/release/ReleaseAllPlanTask.groovy" ]] || {
    echo 'Invalid nexus-release-plugin checkout: RELEASE_ALL_PLUGIN_DIR' >&2; exit 1
  }
  gradle+=("-PreleaseAllPluginDir=$RELEASE_ALL_PLUGIN_DIR")
fi
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
if [[ "$selected" == 0 ]]; then echo 'No eligible unreleased components'; exit; fi
if [[ "$mode" == plan ]]; then
  # Report mode: missing signing configuration is a warning, not a failure.
  "${gradle[@]}" -PreleaseAllSigningReport=true releaseAllSigningCheck
  echo 'Dry run completed; signing configuration will be checked during preparation.'
  exit
fi
"${gradle[@]}" releaseAllSigningCheck
# jmlx stages only Gradle-built components, so no GPG/Maven preflight is needed (unlike
# consumers with static Maven components).
scope=$(groovy -e 'new groovy.json.JsonSlurper().parse(new File(args[0])).verificationScope.each { println it }' "$manifest")
selectedModules=$(groovy -e 'new groovy.json.JsonSlurper().parse(new File(args[0])).selected.findAll { it.projectPath }.each { println it.projectPath }' "$manifest")
checks=()
# verificationScope is the selected modules plus every candidate that (transitively) depends
# on one: for jmlx that means :jmlx-models (SNAPSHOT) is check-verified against the released
# tokenizer/vision versions. Only the selected modules get the SNAPSHOT self-version guard --
# running it on a SNAPSHOT consumer would fail by design.
while IFS= read -r module; do
  [[ -n "$module" ]] || continue
  checks+=("$module:check")
done <<< "$scope"
while IFS= read -r module; do
  [[ -n "$module" ]] || continue
  checks+=("$module:verifyNoSnapshotDependencies")
done <<< "$selectedModules"
if (( ${#checks[@]} > 0 )); then
  "${gradle[@]}" "${checks[@]}"
fi
"${gradle[@]}" -PreleaseAllPrepare=true releaseAllStage
"${gradle[@]}" releaseAllPrepareMaven
"${gradle[@]}" bundleAll
[[ "$mode" == bundle ]] || "${gradle[@]}" releaseAll
