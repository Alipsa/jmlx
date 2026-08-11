#!/usr/bin/env zsh
#
# Repins scripts/bootstrap-native.sh's MLX_C_COMMIT to a new mlx-c commit,
# then re-runs bootstrap-native.sh and regen-bindings.sh so the staged
# native/ tree and the committed jextract bindings both reflect it.
#
# Deliberately does NOT touch MLX_METAL_VERSION/URL/SHA256: mlx-c is
# versioned independently of MLX, and bootstrap-native.sh's own
# version-pairing assertion (req/initial-plan.md Decision 9) already checks
# the new commit's CMakeLists.txt GIT_TAG against the currently pinned
# mlx-metal version and fails loudly on a mismatch -- this script does not
# need to duplicate that check. scripts/checkDependencies.zsh's mlx-metal
# check exists specifically to hand you a commit that already pairs
# correctly with a given mlx-metal version.
#
# Usage: ./scripts/updateMlx.zsh <mlx-c-commit-sha>
#   <mlx-c-commit-sha>  Full 40-character commit hash from
#                       https://github.com/ml-explore/mlx-c. A short hash is
#                       rejected rather than guessed at: bootstrap-native.sh
#                       compares it against `git rev-parse HEAD` (always
#                       full-length) to decide whether a re-checkout is
#                       needed, so a short pin would defeat that idempotency
#                       check on every future run.
set -euo pipefail

REPO_ROOT="${0:A:h}/.."
REPO_ROOT="${REPO_ROOT:A}"
BOOTSTRAP_SCRIPT="$REPO_ROOT/scripts/bootstrap-native.sh"

log() { print -r -- ">>> $*"; }
die() { print -r -- "ERROR: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: $0 <mlx-c-commit-sha>"

NEW_COMMIT="${1:l}"
[[ "$NEW_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "'$1' is not a full 40-character hex commit hash -- pass the complete SHA, not an abbreviated one (see the usage comment at the top of this script for why)"

OLD_COMMIT="$(grep -oE 'MLX_C_COMMIT="[^"]+"' "$BOOTSTRAP_SCRIPT" | grep -oE '"[^"]+"' | tr -d '"')"
[[ -n "$OLD_COMMIT" ]] || die "could not find a MLX_C_COMMIT=\"...\" line in $BOOTSTRAP_SCRIPT -- has it been restructured?"

if [[ "$OLD_COMMIT" == "$NEW_COMMIT" ]]; then
  log "MLX_C_COMMIT is already pinned to $NEW_COMMIT -- nothing to repin, running bootstrap/regen anyway to refresh the staged native/ tree"
else
  log "Repinning MLX_C_COMMIT: $OLD_COMMIT -> $NEW_COMMIT"
  # BSD sed (macOS): -i '' for an in-place edit with no backup file.
  sed -i '' -E "s/MLX_C_COMMIT=\"[0-9a-f]+\"/MLX_C_COMMIT=\"${NEW_COMMIT}\"/" "$BOOTSTRAP_SCRIPT"
  grep -q "MLX_C_COMMIT=\"${NEW_COMMIT}\"" "$BOOTSTRAP_SCRIPT" || die "repin did not take -- $BOOTSTRAP_SCRIPT unchanged, aborting before running anything against it"
fi

log "Running bootstrap-native.sh (this asserts the new commit's CMakeLists.txt GIT_TAG still pairs with the pinned mlx-metal version -- see the header comment above)"
"$REPO_ROOT/scripts/bootstrap-native.sh"

log "Running regen-bindings.sh"
"$REPO_ROOT/scripts/regen-bindings.sh"

log "Done. Next steps:"
log "  1. ./gradlew build -- confirm everything still compiles and passes against the new bindings."
log "  2. git diff --exit-code jmlx-ffi/src/main/generated/java -- if this is non-empty, the bindings actually changed; review the diff."
log "  3. git diff scripts/bootstrap-native.sh -- review the repinned commit."
log "  4. req/initial-plan.md's Decision 9 and its research findings still name the old commit ($OLD_COMMIT) and version in prose -- this script does not edit markdown; update that text by hand if this repin is meant to stick."
