# Phase 7.6 implementation checkpoint — 2026-11-03

The task remains: implement the full reviewed plan, verify it, then open a PR. Branch:
`phase7-6-implementation`. No PR is open. This checkpoint does not claim phase completion.

## Current boundary

The user chose chunked prefill, deferred custom Metal and required structural optimization
before WP2. That gate is now met: clean-source median 474.313 ms for 48 calls / 1,280 tokens,
versus 2,550.557 ms before optimization and 304.105 ms for mlx-vlm ordinary chunked ops.
The target was about 600 ms; all HF comparisons remain at 1e-4. No further default-choice
approval is needed. See [the optimization report](phase7-6-wp1-optimization.md).

The implementation batches preprocessing across at most 16 chunks, uses a stable 64×64 inverse,
bounds state scan evaluation to eight-chunk windows, and has a dedicated lazy decode step.
The complete DeltaNet decode counter measures zero internal evaluations and one caller boundary.
The unchanged memory ceiling passes. The built-in native op remains excluded; no Metal facade
or kernel is added. Proceed to WP2 and the remaining original implementation/PR objective.

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

## Remaining implementation packages

1. WP1 is complete; retain chunked prefill and the dedicated decode step as the chosen paths.
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

## Optimized WP1 gate completed

Clean commit `2c3f9da4b30c1147cae11d6072320683bb9511f1`: 474.313 ms at 48 calls / 1,280
tokens, 1.56× the ordinary mlx-vlm chunked path, within the 600 ms target. Full matrix, strict
HF stress fixtures and unchanged memory checks pass. Calls drop to 430,032; decode measures
zero internal evaluations and one caller boundary. Raw records are under `evidence/optimized/`
and the optimization report. The user has selected chunked ops; proceed to WP2 without another
default-choice permission request. Custom Metal and the built-in op remain excluded.

## WP2 components qualified

Added `GatedDecoderAttention` and `HybridDecoderBlock`, preserving the existing decoder classes.
Attention splits query/gate within each head, uses shifted ordinary Q/K RMS norms, partial
MRoPE and sigmoid gating; its output width comes from configured heads/head dimension.
Independent pinned HF fixtures cover hidden widths 128 and 96 with attention width 128,
unequal multimodal positions, both block variants, prefill and cached decode. All four strict
tests execute and pass at the unchanged 1e-4 bound; rotary tails compare exactly. Split
prefill and invalid mask/rotation preflight are covered. New native suite is required in CI.
The partial-rotary Javadoc now distinguishes qualified Qwen3.5 behavior from the unchanged
Qwen3-VL full-head requirement. Full model loading and generation remain pending.

## WP3 preflight and storage checks

Added pure `Qwen35Descriptor`/`Qwen35Checkpoint` and complete header shape/type/byte-range
validation. Actual five tiny converter artifacts and both recorded real-artifact inventories
classify successfully. Malformed schedules, rotary metadata, capabilities, prefixes, axes,
MTP combinations, partial vision, unknown keys and invalid byte ranges fail before loading.
Typed metadata does not decide capability. Original real configs are copied under evidence;
their hashes match the existing original Hub file inventory.

`Qwen35Weights` selectively retains validated parameters, applies HF-only norm shifts and
convolution transposes once, selects patch axes by shape and preserves packed U32 values.
The new `MLXIO` overload releases unselected lazy native file entries without target-scope
handles or evaluation. Thus recognized MTP data never becomes evaluated model parameters.
Actual float converter parameters compare bit-for-bit with normalized HF parameters; packed
weights retain every bit and floating sidecars become FLOAT32. Both new native tests execute
and pass; required CI coverage is wired. `:check :jmlx-core:check :jmlx-models:check` passes.

Actual mlx-lm config serialization requires the explicit `rope_parameters.type: default`
alias after converted-layout classification; this dated finding is recorded in the living
plan. Missing types and conflicting aliases remain rejected. Full-model float/packed output
acceptance is still pending the WP4 stack; storage checks alone do not complete WP3's
numerical gate. Next record pre-change exact bits in both modes, then extract the generation
loop in its own commit before hybrid generation integration.

## Generation-loop extraction verified

Recorded all 14 existing family/quantized variants in both TF32 modes before editing
`DecoderModel`, at clean source commit `167eb382823f38b70de5b0b40ee822fd055d04c1` (the
unrelated untracked user file still makes the recorder's global dirtiness flag true).
Earlier recordings from `128cd1e` were preserved under ignored
`.tier-b/phase7-6/prior-exact-bits-128cd1e/`; the fresh pre-change captures are also preserved
under `.tier-b/phase7-6/pre-generation-extraction-bits/`. The recorder correctly refused to
overwrite the earlier baseline before it was moved; no baseline was replaced silently.

Extracted package-private `GenerationLoop` with an architecture state adapter. Existing
decoders/composing models keep the same preflight, mutation guard, retained-array evaluation,
sampler, cancellation, events, result IDs and terminal-listener logging category. The generic
loop adds a post-evaluation commit hook for hybrid sessions without changing existing math.
`exactBitsVerify` passes for all 14 variants in both modes, and `:jmlx-models:check` passes.
Baseline metadata/capture digests and extracted source hashes are recorded in
`tools/qwen35-reference/evidence/generation-loop-extraction.json`. This extraction is committed
separately before hybrid stack integration.
