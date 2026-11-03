# Phase 7.6 implementation checkpoint — 2026-11-03

The task remains: implement the full reviewed plan, verify it, then open a PR. Branch:
`phase7-6-implementation`. No PR is open. This checkpoint does not claim phase completion.

## Current boundary

Reference work and core DeltaNet prototypes are committed and locally verified. The 2026-11-03 user review selects chunked prefill, rejects custom Metal for this milestone
and requires an optimization pass before WP2; see [the benchmark report](phase7-6-wp1-benchmark.md). No production
prefill default, new native facade or custom Metal kernel has been selected. The optimization target is about 600 ms at 1,280 tokens / 48 calls, with the unchanged
1e-4 numerical bound. No further default-choice approval is needed after this gate is met.

At 1,280 tokens / 48 independent layer calls, the clean HF-qualified Java medians are about
2.55 s chunked and 13.85 s recurrent. The reference custom Metal path is about 115 ms, while
the built-in op fails the present numerical gate. Measured C API downcalls are 3,159,936 versus
11,676,096; these are not GPU dispatch counts. All raw results and limits are in the report.

## Completed local work

- Separate HF/MLX hash-locked venvs, installed-source/wheel provenance, complete pinned real
  artifact inventories, config/rotary/dispatch/normalization probes and actual converter exports.
- Saved HF, mlx-vlm float/affine and mlx-lm float/affine fixtures with a supported 128-token BPE
  tokenizer. Float saved-weight remaps/axes/norm offsets compare bit-exactly. mlx-lm observation
  hooks now record its own delta and precise gated-norm paths.
- Both large 27B captures were recreated after temporary storage disappeared. Their binary
  checksums and complete records reproduce the originals exactly, using the archived executed
  probe. The captures now live in persistent ignored workspace storage.
- Core `GatedDeltaRule`, `GatedDeltaNet`, `RecurrentCache`; explicit prefill methods, evaluated
  retained state, causal convolution history and HF Q/K epsilon convention.
- Independent HF strict component fixtures, split-prefill/cache/failure tests, repeated request
  cleanup, exact parameter-scope handle stability and a 4,128-position bounded graph test.
- Both precision modes pass `:jmlx-core:check`. Four new native suites are required by CI;
  hosted CI has not run this branch. The pure Gradle reference guard passes and rejects a
  deliberately incorrect hash. Python remains opt-in.
- Clean-source benchmarks at `7f315e02c2dc8e16dca67f2cb196c8923160ee54`, with direct HF output/state
  qualification on all 12 prepared-input cases (maximum error about 1.85e-6, unchanged 1e-4
  bound), first per-shape calls, warmed samples, host observations and separate native-call counts.

Known limits: HF's own real 0.8B cached/fresh text discrepancy reaches 3.19e-4 and remains
recorded as a failed 1e-4 measurement; strict tiny bounds were not changed. Stock/diagnostic
27B normalization contributions reach about 0.10, with recorded greedy choices agreeing.
Diagnostics never replace stock acceptance/goldens. Java full-model remainder is still unmeasured.

## Resume after the WP1 review

1. Select the reviewed prefill default, or amend the plan before adding a custom Metal path.
2. WP2: gated full attention, partial MRoPE differential tests and hybrid blocks.
3. WP3: strict pure descriptor/layout/header preflight; actual float and affine fixtures, with
   declared stock-reference bounds and separate normalization attribution before Java runs.
4. WP4: record old-path exact bits in both modes before generation-loop extraction; commit that
   extraction separately, verify it, then implement hybrid text loading/generation. Keep scheduler
   rejection and the model/session/cache ownership boundaries from the plan.
5. WP5: vision composition/prompt expansion; shared tower accepts empty DeepStack, with the
   nonempty guard retained in every Qwen3-VL loader/descriptor path and its tests moved correctly.
6. WP6: full-session ownership, atomic positions/poisoning, cancellation/failure and memory evidence.
7. WP7: Java 0.8B Tier-B and manual 27B stock text/image acceptance, exact IDs/output and per-step
   error/margins. Declare independent real-artifact tolerances before Java acceptance. Preserve
   failed/ineligible cases and the normalization contribution; no silent epsilon inflation.
8. WP8: compatibility/docs, required native suites and full final CI. Open a PR only when the
   complete task is finished and verified. No merge is authorized by the implementation request.

## Persistent local resources

All paths below are under ignored `.tier-b/phase7-6/`:

- `artifacts/{hf-08b,mlx-27b}`: full pinned checkpoints.
- `captures/{27b-stock,27b-diagnostic}.npz`: about 2.6 GiB each, exact original hashes.
- `benchmark-inputs/`: identical prepared data and independent HF expected output/state files.
- `final-evidence/`: benchmark outputs copied into the committed reference evidence directory.
- The clean benchmark checkout was retired after measurement. The repository call-site scanner
  also scans nested worktrees; create future benchmark worktrees outside the repository (for
  example under `/private/tmp`), while keeping data/captures in this persistent ignored cache.
- `logs/`: saved task logs. Older `/tmp/jmlx-phase76` resources no longer exist.

Venvs remain under `tools/qwen35-reference/{hf,mlx}/.venv/`. The tool README documents regeneration.
The unrelated untracked user file `resume.sh` must remain untouched and out of commits.
Native pins, generated bindings, publication and versions are unchanged. No approval reviewer
rejection occurred. No subagents were used.

Final root-check note: its first run encountered the task-owned nested benchmark checkout and
reported duplicate generated-binding sites from that checkout. The clean temporary worktree
was removed after all measurements; no inventory exemption or binding mapping was changed.

Final verification: `./gradlew :check :jmlx-core:check` passes after retiring the benchmark
checkout. JUnit XML confirms all four new suites actually executed (11 tests total, zero skips,
failures or errors). Provenance verification covers 110 local files plus the shared image.

## WP1 optimization in progress

- Replaced the repeated full-RHS solve with a stable 64×64 inverse batched across up to 16
  chunks; all state-independent quantities are batched. State scan evaluation is bounded to
  eight chunks per window. Temporary handles are retired promptly.
- Added a dedicated lazy decode recurrence and deferred cache retention. The existing counter
  measures zero internal `mlx_eval` calls in complete DeltaNet decode and one caller boundary
  evaluation for output plus both state roots.
- Added independent HF correlated/near-correlated, strong-decay, multi-batch and long-boundary
  stress fixtures; existing strict and memory bounds remain unchanged. The doubling-inverse
  experiment fails the correlated FLOAT32 case and is not used.
- Initial optimized aggregate median: about 512 ms versus 2,551 ms before restructuring.
  Clean-source benchmark/counter refresh and final guard verification remain before WP2.
