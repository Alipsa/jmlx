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
warn() { print -r -- "WARNING: $*" >&2; }
die() { print -r -- "ERROR: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: $0 <mlx-c-commit-sha>"

NEW_COMMIT="${1:l}"
[[ "$NEW_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "'$1' is not a full 40-character hex commit hash -- pass the complete SHA, not an abbreviated one (see the usage comment at the top of this script for why)"

# Split from the die, not chained through `||` on the following line: under
# set -e, a failing pipeline inside a plain assignment kills the script right
# here -- before a die on the next line ever gets to run -- so the one
# scenario this guard exists for (bootstrap-native.sh restructured) would
# otherwise surface as a bare, undiagnosed exit 1.
OLD_COMMIT="$(grep -oE 'MLX_C_COMMIT="[^"]+"' "$BOOTSTRAP_SCRIPT" | grep -oE '"[^"]+"' | tr -d '"')" \
  || die "could not find a MLX_C_COMMIT=\"...\" line in $BOOTSTRAP_SCRIPT -- has it been restructured?"

# Backed up before any edit (even in the "already pinned" branch below, so
# the same rollback logic covers a bootstrap/regen failure regardless of
# whether this run actually changed the pin) and restored by this trap if
# anything after this point fails -- otherwise a version-pairing assertion
# failure inside bootstrap-native.sh (the documented safety net for a bad
# repin) leaves the repo sitting on a repinned commit without regenerated
# bindings and no hint to revert.
BOOTSTRAP_BACKUP="$(mktemp)"
cp "$BOOTSTRAP_SCRIPT" "$BOOTSTRAP_BACKUP"
# Set only once bootstrap-native.sh itself has exited zero (see below). A
# pairing-assertion failure INSIDE bootstrap-native.sh aborts before it
# touches native/install/lib, so reverting the pin is the whole story in
# that case. But if bootstrap succeeds and regen-bindings.sh fails after
# it, native/ has already been rebuilt against $NEW_COMMIT and reverting
# the pin does NOT revert that -- native/ is untracked, ./gradlew build
# doesn't re-run bootstrap, and the mismatch only self-heals on the next
# manual bootstrap run. This flag is how rollback() tells those two cases
# apart.
BOOTSTRAP_COMPLETED=""
rollback() {
  # Not "status": zsh reserves that name as a read-only synonym for $? and
  # refuses `local status=...`.
  local exit_status=$?
  if [[ $exit_status -ne 0 ]]; then
    cp "$BOOTSTRAP_BACKUP" "$BOOTSTRAP_SCRIPT"
    warn "aborting -- reverted scripts/bootstrap-native.sh to its state from before this run (git diff scripts/bootstrap-native.sh should now be clean if it was clean going in)"
    if [[ -n "$BOOTSTRAP_COMPLETED" ]]; then
      warn "note: native/, jmlx-ffi/src/main/generated/java, and req/mlx-api-inventory.md may already reflect $NEW_COMMIT and are NOT reverted -- re-run ./scripts/bootstrap-native.sh and ./scripts/regen-bindings.sh (now back on $OLD_COMMIT), then ./gradlew generateMlxApiInventory before ./gradlew build"
    fi
  fi
  rm -f "$BOOTSTRAP_BACKUP"
}
trap rollback EXIT

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
BOOTSTRAP_COMPLETED=1

log "Running regen-bindings.sh"
"$REPO_ROOT/scripts/regen-bindings.sh"

log "Regenerating the checked-in MLX API inventory"
"$REPO_ROOT/gradlew" -p "$REPO_ROOT" generateMlxApiInventory

log "Done. Next steps:"
log "  1. ./gradlew build -- confirm everything still compiles and passes against the new bindings."
log "  2. git diff --exit-code jmlx-ffi/src/main/generated/java -- if this is non-empty, the bindings actually changed; review the diff."
log "  3. git diff req/mlx-api-inventory.md -- review the regenerated inventory, since a bindings diff changes the symbol set it renders."
log "  4. git diff scripts/bootstrap-native.sh -- review the repinned commit."
log "  5. req/initial-plan.md's Decision 9 and its research findings still name the old commit ($OLD_COMMIT) and version in prose -- this script does not edit markdown; update that text by hand if this repin is meant to stick."
