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

## Hybrid text stack and packed numerical gate

Added internal `Qwen35Decoder` with full/recurrent per-layer state, lazy per-step construction,
one all-roots execution boundary and a separate global-position commit. Split prefill covers
convolution/chunk boundaries; late evaluation failure poisons every cache without advancing
the committed session position. Independent CPU HF goldens cover all float layouts, tied/equal
linear heads and an eight-layer schedule, lengths 4/65/129, two cached steps, fresh full
forwards and complete retained state. Strict 1e-4 bounds pass unchanged.

Before Java packed-output comparison, `generate_packed.py` declared 1e-4 arithmetic bounds
against independently dequantized HF weights, plus separately measured normalization-only
contributions for stock mlx-vlm envelopes. The diagnostic never supplies acceptance goldens:
the exported file contains independent HF and unmodified stock MLX logits only. Case selection
and gap eligibility precede Java; all 16 candidate prompts per family remain recorded.
12/16 VLM and 16/16 LM cases satisfy gap > 2*epsilon + 1e-5 at every recorded step; exact
IDs agree on every eligible case. The four ineligible VLM cases are not counted as passing ID
proof. Maximum Java/HF error is 7.1525574e-7 for both layouts; maximum Java/stock VLM error
is 0.024449743, separately accounted for by its measured normalization convention. LM stock
maximum error is 6.854534e-7. This completes WP3's text packed numerical gate; image composition,
public generation, default-precision and Tier-B acceptance remain pending.

## Public generation, vision and ownership qualified

Added `Qwen35Model` implementing `TextGenerationModel` directly, dispatching only dense
`qwen3_5`, with capability selected from the full vision inventory. All five storage layouts
load; both LM layouts report TEXT and reject images. Scheduler startup rejects all five
because the hybrid model is not a `DecoderModel`. Zero-token requests validate expanded
geometry/capacity without inference. Public sampling, penalties, token/terminal events,
EOS/stop precedence, abort cleanup and tokenizer output pass.

The shared tower permits empty DeepStack levels; Qwen3-VL's descriptor guard remains and
its loader explicitly checks before construction. The old tower rejection test is replaced
with positive empty-feature-level and extra-subtree rejection checks, plus descriptor/loader
family rejection. Independent HF vision cases pass for HF and actual VLM float layout,
including resized/multiple images and both 8×8/48×48 learned grids. Pixels (1e-6), positions
(exact), patch/interpolation/block/merger and full logits (1e-4), cached/fresh and public greedy
IDs pass. Initial vision validation caught an incorrect fixture whitespace recipe: original
IDs were assumed rather than tokenized. Preserved rejected recipe/artifact under ignored
`.tier-b/phase7-6/rejected-vision-whitespace-recipe/`, then regenerated from tokenizer IDs.
No Java math or numerical bound was changed to accommodate that recipe correction.

Actual converted processor serialization needs a family-specific adapter documented in the
living plan. Its saved files remain unchanged; source/pixel alias probe passes at 1.20e-7.
Cancellation is cooperative between images, layers/chunks and decode steps; request reuse
after cancellation passes. History uses indexed gathering to avoid retaining a larger
contiguous-slice/donated buffer. Exact physical memory checks control allocator pool reuse:
fixed recurrent state, K/V-only growth, borrowed handle retirement and exact cleanup pass.
Default-mode HF error maximum is 0.0014960468 under its separately predeclared 0.005 bound.
Strict FLOAT32 remains unchanged at 1e-4. Public-decode counter passes: no internal mlx_eval
or mlx_array_eval before the boundary, one explicit all-roots mlx_eval, with readbacks recorded
separately. Full regression refresh and real Tier-B acceptance are next; no PR is open.

## Real-artifact acceptance declarations (before Java)

Full regression refresh and both-mode exact-bit checks pass. Added a compile/lint-verified
opt-in `TierBQwen35Test`, recording actual vectors, per-step errors/margins, exact IDs/text,
first/warm request timings, native/process peaks and host observations. Bounds are declared
before any Java real-model output: HF 0.8B uses 0.001 (independent HF cached/fresh reached
0.000319, preserved failures at 1e-4); 27B keeps an independently fixed arithmetic allowance
of 0.001 plus the measured stock normalization convention. Strict tiny remains 1e-4.

Recorded full FLOAT32 stock and separately labeled diagnostic 27B probes before Java. The
original prompt pair fails a global margin envelope because late text continuation has a
0.29576 normalization shift, making that envelope too wide for other phases. It is preserved
as ineligible. Added independently rendered candidates and stock/diagnostic probes with
logits-only capture (same math/evaluations/path observations, omitting large captured states).
`27b-reference-selection.json` preserves all failed candidates and selects the first eligible
pair, `text_copy,image`, without Java measurements. Epsilon is 0.10062797164916992, comprising
the unchanged 0.001 arithmetic allowance and a measured 0.09962797164916992 normalization
contribution. Minimum gap is 0.3372325897216797, greater than 2*epsilon+0.0001. Diagnostic
arrays never enter acceptance goldens. All nine source outputs (prefill plus eight decode
outputs) remain included, including the post-budget ninth prediction. Public timing/ID scope
is eight generated tokens with EOS stopping disabled, explicitly recorded.

Both manifests pin original file digests, CPU-HF-derived positions, rendered prompts, expected
decoded output, selected stock-only logits and predeclared resource bounds. They are covered
by the pure reference guard as shared files. The old probe source before logits-only capture
is archived under executed-sources to retain exact executed hashes. Persistent FLOAT32 and
candidate captures are under `.tier-b/phase7-6/captures/`. Next execute the two Java suites
sequentially on committed source; retain failed runs and diagnose without loosening bounds.

## WP7 real-artifact acceptance executed (2026-11-03)

Both Java suites ran sequentially on committed source `ee822c9` (M5 Max, macOS 27.2, MLX
0.32.4 / mlx-c a26f2d5, TF32 off), the 0.8B Tier-B first and the 27B manual run second.
Both passed every numerical, stability, exact-ID/decoded-output and resource gate on the
first attempt; no bound was loosened and no golden was regenerated:

- 0.8B (`TierBQwen35Test`, task `qwen35-generation`): maximum Java/HF absolute logit error
  8.320808410644531e-5 (text) and 4.0340423583984375e-4 (image), under the predeclared
  ε = 0.001 / margin = 1e-4; exact greedy IDs and decoded output on both cases in every
  warmed repetition; native peak 5,118,123,572 B and sampled footprint 6,648,795,264 B,
  both under the predeclared 8e9/12e9 `resource_limits`. Record:
  `tools/qwen35-reference/evidence/08b-java-acceptance.json`.
- 27B (manual run against `tools/tier-b/qwen38-27b-4bit.json`): maximum Java/stock error
  0.044699668884277344 (text) and 0.09969949722290039 (image), under ε =
  0.10062797164916992 (the unchanged 0.001 arithmetic allowance plus the measured
  0.09962797164916992 stock normalization contribution) / margin 1e-4, minimum top-two gap
  0.3372325897216797; exact greedy IDs and decoded output on both cases; native peak
  19,261,831,808 B, sampled footprint 22,680,715,080 B, load 853 ms. Record:
  `27b-java-acceptance.json` plus `27b-java-normalization-attribution.json`; the
  independently labeled diagnostic arrays never enter the acceptance goldens.
- Default-precision comparison only (27B, mlx-vlm's default BF16 parameters against the
  qualified FLOAT32 teacher histories): maximum logit difference 1.1851611137390137 (image
  prefill), all recorded greedy choices agree. Record:
  `27b-default-precision-comparison.json`; qualified acceptance remains FLOAT32/TF32 off.

The 0.8B manifest gained its `local_evidence` block (measured maxima, peaks, walls);
provenance was re-recorded and verified (190 local + 3 shared files, both manifests among
the shared set).

## Qwen35MemoryTest allocator over-charge resolved (2026-11-03)

The intermittent +16384/+32768 activeBytes failure at position 128 is the pinned MLX 0.32.4
Metal allocator's pool reuse, not a jmlx leak and not caused by per-step `clearCache`:
`BufferCache::reuse_from_cache(n)` accepts any pooled buffer in `[n, min(2n, n + 2 * page))`,
and `MetalAllocator::malloc` charges `active_memory_ += buf->length()` (the physical length,
not the requested size). The fixture's 49152-byte GDN state buffers sit inside only the
32768-byte KV tier's window (positions 65–128; the 16384 and 65536 tiers' windows exclude
it), so K and/or V can be physically charged 49152 instead of 32768: 0, 16384 or 32768
bytes over. The over-charge is the live K/V buffer itself — it dies at the 129 tier crossing
and at session end (the exact baseline assertion always passed), and a KV-only probe without
the GDN buffers is byte-exact clean at every step. Whether a pooled 49152 old-state buffer
(handed back by the previous step's late Metal completion-handler teardown) is in the pool at
the KV malloc races the dispatch thread, which makes the failure intermittent; `clearCache`
only modulates the frequency because the pool refills intra-eval. Pre-fix failure values were
always 247808 (+16384) or 264192 (+32768) against expected 231424, and never anything else
across ~20 runs. The test now asserts that exact set of physically possible deltas at 128
(with a comment citing the mechanism) and keeps exact equality at 64, 256 and the session-end
baseline — the real leak check. Verified stable: 10 consecutive full-suite runs pass. The
PR #46 rule (memory 030d45fb) "clear the cache before the endpoint and assert exact equality"
holds only while endpoint requests are below 16 KiB or their windows contain no other
retained size; this fixture is the first counterexample geometry.

## WP8 closeout (2026-11-03)

- `.github/workflows/tier-b.yml`: new `qwen35` arm on `self-macos-xl` (task
  `qwen35-generation`, suite `TierBQwen35Test`); the header and step comments now cover the
  four self-hosted arms.
- Docs updated: `req/phase6-compatibility.md`, `req/phase6-inference-report.md`,
  `req/phase6-tier-b-artifacts.md`, `req/phase6-golden-precision.md`,
  `tools/tier-b/README.md`, `jmlx-models/README.md`, the parent plan's delivery-order table
  and §7.6 status, and AGENTS.md.
- The three scratch memory-diagnostic tests were deleted; the throwaway benchmark worktree
  was removed.
- Full root `./gradlew check` re-run green on the final tree (post memory-test fix); PR
  opened against `main`. No merge is authorized; human review and the hosted CI runs
  (macOS 26 native job; the qwen35 arm's first self-hosted execution) remain pending.
