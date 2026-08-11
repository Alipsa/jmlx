#!/usr/bin/env bash
#
# Regenerates jmlx-ffi's jextract bindings from the headers staged by
# bootstrap-native.sh (native/install/include/mlx/c/mlx.h). Idempotent and
# meant to be re-run whenever mlx-c's pinned commit changes; see
# req/initial-plan.md §3 and Decisions 8/10 for the rationale behind each
# non-obvious step below.
#
# jextract build pinned: 25-jextract+2-4 (2025-11-25), resolved and verified
# by scripts/bootstrap-native.sh's trust-on-first-use check.
#
# Usage: ./scripts/regen-bindings.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="$REPO_ROOT/native/install"
INCLUDE_DIR="$INSTALL_DIR/include"
UMBRELLA_HEADER="$INCLUDE_DIR/mlx/c/mlx.h"
HALF_H="$INCLUDE_DIR/mlx/c/half.h"
HALF_H_OVERRIDE="$REPO_ROOT/scripts/jextract-overrides/mlx/c/half.h"

JEXTRACT_BIN="$REPO_ROOT/native/scratch/jextract/bin/jextract"
SCRATCH_DIR="$REPO_ROOT/native/scratch/regen-bindings"
DUMP_FILE="$SCRATCH_DIR/discovered-includes.args"
FILTERED_ARGS_FILE="$SCRATCH_DIR/filtered.args"
UNFILTERED_SCRATCH_OUT="$SCRATCH_DIR/pass1-unfiltered"

OUTPUT_DIR="$REPO_ROOT/jmlx-ffi/src/main/generated/java"
TARGET_PACKAGE="se.alipsa.jmlx.ffi"

log() { printf '>>> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$JEXTRACT_BIN" ]] || die "jextract not found at $JEXTRACT_BIN -- run ./scripts/bootstrap-native.sh first"
[[ -f "$UMBRELLA_HEADER" ]] || die "$UMBRELLA_HEADER not found -- run ./scripts/bootstrap-native.sh first"
[[ -f "$HALF_H_OVERRIDE" ]] || die "missing $HALF_H_OVERRIDE"

rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR" "$UNFILTERED_SCRATCH_OUT"

# Decision 10: mlx/c/half.h's __bf16 typedef is a hard parse error for
# jextract on this target (v0.1 doesn't support bfloat16 anyway). Patch it out
# for the duration of this script, then restore the real header -- so
# native/install/include stays byte-for-byte what bootstrap-native.sh staged
# once we're done here.
cp "$HALF_H" "$SCRATCH_DIR/half.h.real"
restore_half_h() { cp "$SCRATCH_DIR/half.h.real" "$HALF_H"; }
# INT/TERM too, not just EXIT: a Ctrl-C mid-jextract must not leave the
# patched (bf16-typedef-stripped) half.h sitting in native/install/include,
# silently diverging from what bootstrap-native.sh staged there.
trap restore_half_h EXIT INT TERM
cp "$HALF_H_OVERRIDE" "$HALF_H"

# --- Pass 1: discover every symbol jextract would emit, unfiltered ----------
# We don't use this pass's generated Java at all -- only --dump-includes'
# output, which lists every discovered symbol together with the header it
# came from. No -l flag (Decision 8): NativeLoader loads via
# System.load(absolutePath), which only composes with the default
# loaderLookup()-based SYMBOL_LOOKUP jextract emits without -l.
log "Pass 1/2: discovering symbols (this run's generated Java is discarded; only --dump-includes matters)"
"$JEXTRACT_BIN" \
  --output "$UNFILTERED_SCRATCH_OUT" \
  -t "$TARGET_PACKAGE" \
  --dump-includes "$DUMP_FILE" \
  --include-dir "$INCLUDE_DIR" \
  "$UMBRELLA_HEADER"

[[ -s "$DUMP_FILE" ]] || die "--dump-includes produced an empty file -- did jextract fail silently?"

# --- Filter: keep only symbols that came from mlx-c's own headers ----------
# Two independent reasons this filter exists (req/initial-plan.md Decision
# 10): it trims the Darwin system-header spillover jextract pulls in
# transitively (pthread/rusage/signal/etc. -- irrelevant noise in a generated
# blob nobody should be reviewing line-by-line), and as a side effect it also
# fixes a separate jextract bug where mlx_array_new/free/new_data/
# new_data_managed are silently dropped from an *unfiltered* whole-umbrella
# emission despite being correctly discovered in pass 1 above.
grep -E '# header: .*/mlx/c/' "$DUMP_FILE" | sed 's/ *#.*//' > "$FILTERED_ARGS_FILE"
FILTERED_COUNT="$(wc -l < "$FILTERED_ARGS_FILE" | tr -d ' ')"
[[ "$FILTERED_COUNT" -gt 0 ]] || die "filtering --dump-includes down to mlx/c/ headers produced zero symbols -- did the header layout change?"
log "Filtered to $FILTERED_COUNT symbols under mlx/c/ (from $(wc -l < "$DUMP_FILE" | tr -d ' ') discovered total)"

# --- Pass 2: the real generation, restricted to the filtered symbol set ----
# Generated into a scratch dir first, not straight into OUTPUT_DIR: a pass-2
# failure (or a mid-run Ctrl-C) must never leave the committed bindings
# deleted with nothing valid to replace them.
PASS2_OUTPUT_DIR="$SCRATCH_DIR/pass2-output"
mkdir -p "$PASS2_OUTPUT_DIR"
log "Pass 2/2: generating bindings into $PASS2_OUTPUT_DIR"
"$JEXTRACT_BIN" \
  --output "$PASS2_OUTPUT_DIR" \
  -t "$TARGET_PACKAGE" \
  --include-dir "$INCLUDE_DIR" \
  "$UMBRELLA_HEADER" \
  "@$FILTERED_ARGS_FILE"

GENERATED_COUNT="$(find "$PASS2_OUTPUT_DIR" -name '*.java' | wc -l | tr -d ' ')"
[[ "$GENERATED_COUNT" -gt 0 ]] || die "pass 2 produced no Java files"
log "Generated $GENERATED_COUNT Java files into $PASS2_OUTPUT_DIR"

# Sanity-check the two independent bugs this script works around, so a
# jextract/mlx-c upgrade that silently changes behavior fails loudly here
# instead of surfacing later as a confusing NoSuchElementException.
for fn in mlx_array_new mlx_array_free mlx_array_new_data mlx_get_default_device mlx_add; do
  grep -rq "findOrThrow(\"$fn\")" "$PASS2_OUTPUT_DIR" || die "expected generated binding for $fn is missing -- see Decision 10 in req/initial-plan.md"
done
log "Confirmed mlx_array_new/free/new_data and mlx_get_default_device/mlx_add are present."

# Only now, with pass 2 fully generated and verified, replace the committed
# bindings -- the window in which OUTPUT_DIR doesn't exist is a single mv.
rm -rf "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"
mv "$PASS2_OUTPUT_DIR" "$OUTPUT_DIR"

log "Done. Re-run and 'git diff --exit-code $OUTPUT_DIR' to check for bindings drift."
