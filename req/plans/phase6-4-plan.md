# Phase 6.4 implementation plan — decode performance and cache management

**Goal:** Make prefill and single-token decode explicit, give KV caches bounded and testable
lifecycle semantics, and publish reproducible performance and native-memory measurements for every
supported decoder family.

**Sources:** `req/full-roadmap.md` §6.4 and `req/plans/phase6-plan.md` §6.4. Phase 6.4.1 owns
packed-KV attention and quantized retention after this milestone; Phase 6.5 owns the request
scheduler. This milestone provides the batch-safe cache and mask primitives both need.

**Implementation locations:** `jmlx-core/src/main/java/se/alipsa/jmlx/nn/` (`KVCachePolicy`, `KVCache`,
`AttentionMask`, `CachedAttention`, `DecoderBlock`, `DecoderAttention`,
`GroupedQueryAttention`, `MultiHeadAttention`, `RopeSpec`),
`jmlx-models/src/main/java/se/alipsa/jmlx/models/` (`DecoderModel`, `GenerationRequest`,
`GenerationCachePolicy`, `SamplingPipeline`), a new opt-in runner under
`jmlx-examples/src/main/java/se/alipsa/jmlx/examples/`,
`jmlx-examples/build.gradle`, and `.github/workflows/ci.yml`.
Keep the corresponding native tests in each module's `src/test` tree.

**Starting point (2026-09-30):** `DecoderModel.generate` already feeds the full prompt once and
then one token per loop, but uses one generic path. `KVCache` is append-only: `offset()` is both
absolute next position and retained sequence length. `DecoderAttention` assumes
`keyLength == offset + sequence`, and `AttentionMask.slidingWindow` assumes retained keys begin at
position zero. Those assumptions fail as soon as old keys are evicted. `KVCache.append` closes the
superseded handles it owns; forks cannot share those handles. Supported families in
`req/phase6-compatibility.md` are Llama, Qwen2, Mistral, Gemma v1, Phi-3, and Mixtral. The current
runtime is Java 25 on macOS Apple Silicon, mlx-c `fba4470` and `mlx-metal==0.31.2`.

## Contracts to settle first

- **Positions:** keep an absolute `nextPosition` (current `offset()` compatibility meaning), a
  `startPosition` for the first retained key, and `length = nextPosition - startPosition`. RoPE and
  dynamic frequency calculation use absolute positions. Masks compare absolute query/key positions.
  A cache's arrays have sequence axis length `length`, not `offset` after eviction. Every layer in
  one decoder request must have the same position range across layers before and after a forward
  call. This is a deliberate 6.4 limit: mixed per-layer windows (`max_window_layers`, mixed
  `layer_types`) remain unsupported. For a batch, store these values **per row**. Left-pad
  retained keys/values to the longest retained row: the
  newest key is at the right edge for every row, so single-token append remains one uniform
  concatenate. Trim and reorder restore left padding; masks exclude each row's padded prefix.
  Keep `offset()` for a single row or uniform batch and throw a named error for unequal positions,
  so an old scalar call cannot silently rotate every row at the wrong position.
- **Capacity:** `new KVCache(scope)` remains unbounded `FULL` for source and behavior compatibility.
  An explicit `FULL(capacity)` policy rejects a request *before* an append that would exceed
  capacity; it never silently evicts needed context. For a left-padded batch, capacity bounds the
  physical padded sequence width (the maximum retained row length). Preflight every row and the
  resulting padded width before a batch append; for `FULL`, an absolute `nextPosition` beyond
  capacity is invalid as well. The single-request generation formula below does not substitute
  for this per-batch preflight. Sliding attention retains at
  most `window - 1` prior keys between calls; this is its steady-state retention limit, not an
  additional user-set maximum. The next single-token query sees exactly its
  permitted predecessors. A window of one therefore retains zero rows. The attention operation
  may transiently need more than the steady-state limit during multi-token prefill; benchmark
  peak memory separately from retained memory.
- **Policy split and eviction trigger:** define `GenerationCachePolicy` in `jmlx-models` for
  requests; `GenerationRequest.withCachePolicy(GenerationCachePolicy.slidingWindowFromModel())`
  passes an unresolved value. Resolve it once against
  `descriptor.attention().slidingWindow()` before constructing caches. Define a separate,
  fully resolved `KVCachePolicy` in `jmlx-core`; its public `KVCache` constructor accepts only
  `FULL` or a numeric `SLIDING_WINDOW(window)`, never a model-derived placeholder. Only that
  resolved sliding policy evicts, and its window must equal the descriptor's non-null window.
  Reject either request-level sliding variant by name whenever the loaded descriptor has
  `slidingWindow() == null`, regardless of family. `FULL` on a windowed layer retains all keys and still
  applies the layer's sliding attention mask, matching current `DecoderModel.forward` behavior.
  Existing request factories and `new KVCache(scope)` default to `FULL` for **all six families**.
  The current tiny Mistral fixture has window 4; tiny Phi-3 and Mixtral fixtures have no window,
  so only Mistral supplies synthetic eviction evidence. An explicit request policy opts into
  eviction on a checkpoint with a window. Keep existing FULL
  goldens; compare SLIDING output to new independent references with numerical tolerance because
  shorter SDPA key lengths may change reduction order and token ties.
- **Prefill semantics:** a multi-token prefill must compute attention using all keys needed by
  *each* query in that call. The existing `KVCache` ownership contract records that an MLX
  `ArrayDesc` owns lazy graph inputs by value; do not synchronize at each layer. Build the final
  logits and, only for `SLIDING_WINDOW`, trimmed cache tensors. Pass those tensors into
  `SamplingPipeline.select` so its existing single `MLX.eval` call on either sampling path evaluates
  selection and cache tensors together. A standalone public `DecoderModel.forward` evaluates its
  logits and changed cache tensors once before returning, so callers do not inherit an
  unobservable pending cache mutation.
  Probe whether the evaluated trim frees its old backing buffer. Prefill length may exceed the
  window. A following decode uses absolute
  positions from the untrimmed total. For a chunked prefill, mask construction must account for
  a nonzero `startPosition`.
- **Ownership:** reset closes owned handles and returns positions to zero; closing an old handle
  never invalidates the caller's input. `fork` creates independent owned key/value storage in the
  destination scope (materialize a copy, not a view or merely another wrapper); `reorder` gathers
  and materializes independent storage along batch axis, supports repeated indices, and validates
  their bounds. `fork` and `reorder` each evaluate their new key/value storage once before returning;
  they run outside the generation sampler, and the result must survive source-scope close without
  retaining the source buffer. Append/reset/evict on one fork must not change another. Cache methods
  remain on the scope-owning thread. Validation failure leaves the prior state intact; graph-construction
  failures are atomic only where both replacement handles have not yet been installed.
- **Cache-set failure:** validate all layer shapes, positions, policies and capacity before a
  forward mutates any layer. From the first cache mutation until successful step-boundary eval,
  **any** exception, including lazy Metal or OOM failures surfaced by `MLX.eval` or sampling,
  poisons every cache in the set. Subsequent forward/append rejects the set until every cache is
  reset; generation closes its request scope on failure. Do not promise rollback after installed
  handles have replaced and closed prior handles. For bounded `FULL` generation, preflight the
  exact required positions with long arithmetic:
  `maxNewTokens == 0 ? 0 : (long) prompt.length + maxNewTokens - 1`. Zero tokens does not prefill;
  no known capacity failure may occur after tokens have been emitted.
- **Batch validity:** reject zero-row batches and zero-length prefill/decode inputs at public
  boundaries. Reorder may produce a one-row or repeated-row batch, but an empty index list is an
  error. For padded rows, queries outside each row's valid length must be excluded from output
  selection; do not send a fully masked query row to native attention without proving its result
  is finite and unused.
- **Compatibility:** keep the signatures of `DecoderModel.forward`, legacy `generate`, and the 12-component
  `GenerationConfig` record unchanged. Add an immutable cache-policy field to the final
  `GenerationRequest` class, defaulted by every existing constructor/factory; expose a
  `withCachePolicy(...)` copy method for opt-in calls. Keeping `GenerationConfig` unchanged
  preserves its record patterns and equality; `GenerationRequest` keeps identity equality.
  The public `DecoderModel.forward` now synchronizes before returning to detect lazy native
  failures and commit or poison its cache set; document this behavioral change in its Javadoc.
  Generation's internal prefill/decode path does not call the public forward and joins cache
  evaluation to the sampler's existing sync. Default full-attention behavior and greedy/sampling
  goldens stay unchanged. A windowed architecture cannot select a policy that changes its
  descriptor's numerical window. Reject contradictory or unsupported settings by name.
- **Cancellation:** poll before prefill and between decode steps. An in-progress prefill is not
  interruptible. Preserve event order, EOS/stop-token behavior, seeded RNG sequence, text deltas,
  and `GenerationAbortedException` partial results.
- **Other public attention consumers:** `GroupedQueryAttention` and `MultiHeadAttention` use
  scalar offsets and implicit causal masks. Keep their existing behavior by rejecting a cache
  unless it is `FULL`, uniform across rows, and has `startPosition == 0`, before any projection
  or cache mutation. Add native tests for each rejection path.
- **Dynamic NTK:** `RopeSpec.stepFrequencies` yields one frequency vector for one ending length.
  Reject unequal-position batches for `DynamicNtk` before mutation in 6.4; a later per-row
  frequency implementation can lift this restriction. Other RoPE modes may use per-row offsets
  after the native probe below.

## Work sequence

### 0. Record baseline and probe native capabilities

- [ ] Add a small, opt-in `jmlx-examples` benchmark CLI before performance edits. It loads a
  local checkpoint, runs cold load, prefill, one-token decode, and sustained generation as distinct
  phases, and writes machine-readable JSON plus a short Markdown report. Do not download models
  or run it as a required CI performance gate. The harness source, exact commands and results are
  committed and documented (the roadmap's "published benchmark harness"); `jmlx-examples` itself
  stays unpublished. Add `implementation project(':jmlx-models')` to
  `jmlx-examples/build.gradle`, plus `implementation libs.jackson.databind` for JSON output
  (or an explicitly tested manual JSON writer). Add a separate `JavaExec` task with
  `jmlx.library.path`; keep `HelloMLX` as the application main class. Run the tiny-fixture harness
  in CI as a smoke test that validates the JSON schema and exit status, with no speed thresholds.
- [ ] Include commit, Java version, OS/device, mlx-c and `mlx-metal` pins, model revision and hashes,
  architecture, dtype, generation/cache policy, batch, prompt/context lengths, warm-up count,
  sample count, median and distribution (for example p10/p90), tokens/s, and peak *active* native
  bytes, plus native memory growth: active bytes sampled before and after each phase and at fixed
  token intervals during sustained and repeated generation. Synchronize/evaluate before stopping
  each timer. Define cold load as the first load in a fresh JVM with the OS page cache in its
  observed state, not a claimed flushed page cache. Hash checkpoints **after** the timed load
  (or in a separate process before launching the measurement JVM) and record the method used.
  Record cold load separately from warm
  inference and specify timer boundaries. Report allocator cached bytes and RSS
  separately if collected; neither substitutes for active native memory.
- [ ] Add a documented public `jmlx-core` memory facade for active bytes, peak bytes, peak
  reset and optional allocator cached bytes, based on the `jmlx-ffi` test-fixture
  `NativeMemoryProbe`. Probe the already bound `mlx_get_peak_memory`,
  `mlx_reset_peak_memory`, and `mlx_get_cache_memory` on the pinned runtime and define counter
  scope, thread behavior and failure handling in Javadoc. If reset is ineffective, run phases in
  separate processes or label peaks process-wide. Mark the facade's calls implemented in the MLX
  API inventory and pass the call-site guard.
- [ ] Record the pre-change baseline for one tiny deterministic checkpoint and at least one
  accessible Tier-B checkpoint. Save command lines, raw JSON, environment and interpretation under
  a Phase 6.4 benchmark report; do not set a speedup target before those numbers exist, and land
  no performance optimization before this baseline is committed.
- [x] Probe cache quantization separately on the pinned MLX runtime: representation, supported
  bits/group sizes and head shapes, round trip, direct packed SDPA, and the dequantization memory
  path. The probe found no useful attention-memory path; measurements and the deliberately skipped
  append/slice/reorder and scope-ownership probes are recorded in `req/phase6-4-benchmark.md`.
  Those lifecycle probes become required if a future fused packed-KV attention path makes the mode
  viable.

### 1. Split prefill and decode without changing results

- [ ] In `DecoderModel`, introduce internal prefill and single-token decode methods. Prefill
  accepts `[B,T]` with `T > 0`; decode accepts `[B,1]` and a populated cache. Only prefill selects
  the final hidden-state row from a multi-token output. Keep one request-owned cache set and
  per-step activation child scopes. Avoid copying or rebuilding the prompt on decode, and never
  recompute cached keys/values; construct only the next token's `[B,1]` input. Keep the public
  `forward` wrapper separate: it evaluates once before return, while generation calls the internal
  methods directly so the sampler remains its only step-boundary sync. Update forward's Javadoc.
- [ ] Assert all layer caches have matching absolute/retained positions before and after each
  call. Preflight every layer and poison the set for any failure before successful step eval after
  mutation, including an eval-time failure. Add a package-private injectable step-boundary
  evaluator shared by the public `DecoderModel.forward` boundary and the generation sampler
  boundary; production delegates to the existing `MLX.eval` call, while tests can throw at that
  boundary after cache mutation. Pass the evaluator from the model instance into each
  per-request `SamplingPipeline` constructor; both the greedy and sampled branches of `select`
  use it for their existing eval calls. Do not use a static hook or add a second evaluation to
  generation. Test a non-finite-logits failure separately through `generate` by replacing a
  test fixture's output-projection weights with NaNs so `requireFinite` throws after eval. The
  public `forward` path does not gain a finite-logits check. Label this case separately from
  the injected eval failure.
  Add a named failure
  for a mismatched cache list or batch shape. Preserve `forward` as the general public entry point,
  delegating through the same internal attention logic.
- [ ] Lock existing Llama/Qwen greedy and seeded goldens, plus family prefill/decode Hugging Face
  goldens. Extend the existing cancellation tests for prefill and decode boundaries and add
  listener-failure cleanup cases. Run the baseline harness again after this refactor and record
  the delta.

### 2. Implement capacity, reset, eviction, fork and reorder in `KVCache`

- [ ] Add the pure-Java validated, resolved `KVCachePolicy` in `jmlx-core`, plus the request-level
  `GenerationCachePolicy` and descriptor-based resolver in `jmlx-models`. Add explicit `startPosition`, `nextPosition`,
  `length`, batch shape and dtype invariants, with per-row metadata for batched caches. Keep
  `offset()` as an alias for uniform `nextPosition` for existing callers. Provide a batched append
  overload with per-row valid input lengths and the left-padded retained layout above. Update
  `keys()`/`values()` Javadoc from `offset()` to retained `length()` and describe padded batches.
  Reject negative/overflowing positions, incompatible append shapes, and full capacity overrun
  before replacing handles.
- [ ] Implement `reset`, post-attention `trimToLast(n)`/sliding eviction, `fork(destinationScope)`,
  and batch `reorder(indices, destinationScope)` using `MLXShape` operations plus an explicit
  materialization strategy proven by a native ownership test. Handle empty retained windows,
  duplicated reorder indices, and one-row batches; reject an empty index list. Return a new cache
  for reorder so the source remains valid if either tensor operation fails. Evaluate both new
  tensors before returning from fork or reorder; close temporary handles on every failure path
  and document destination-scope lifetime requirements.
- [ ] Evaluate append and trim arrays through the sampler's existing eval, without a second sync
  or per-layer sync. Fork and reorder use their own one-time eval before returning. Do not assume
  `copy` or `contiguous` allocates independent storage. Probe evaluated data pointers where
  available and the active-byte delta after releasing the source; include `B * H == 1` tail slices.
  Require independent storage for fork/reorder ownership. For sliding decode, permit one previous
  backing buffer until the next step; require a steady active-memory plateau bounded by
  `3 * logicalRetainedCacheBytes + 8 MiB` above the warmed model baseline. Start plateau samples
  only after at least two decode steps following prefill, so the first retained view of a long
  prefill buffer is excluded; report that transient peak separately. Confirm no upward trend
  across two equal-length token intervals. Document any fixture-specific adjustment.
- [ ] Add native-gated `KVCacheTest` cases for shape/position invariants, reset and reuse, copy
  independence (including closing the source scope), repeated-index reorder, failure atomicity,
  and window one. Put the measured active-memory plateau in a named
  `KVCacheMemoryPlateauTest`. Pure-Java policy validation and null-descriptor-window rejection
  run on Ubuntu.

### 3. Make attention and masks correct after eviction and for batches

- [ ] Update `DecoderAttention` and `DecoderModel.normalizedHiddenStates` to derive the attention
  key range from retained cache start/length and the new chunk, while deriving RoPE offsets and
  dynamic frequencies from absolute next position. Only an explicit `SLIDING_WINDOW` cache trims;
  a `FULL` cache on a windowed layer remains untrimmed and masked. The attention graph must retain
  complete just-appended keys before the trimmed cache is built; never trim before a multi-token
  prefill query has used those keys. Evaluate with sampler selection once per generation step. Update
  `AttentionMask` to take absolute query/key starts and lengths.
- [ ] Add compatible public overloads: `DecoderModel.forward(tokenIds, caches, validLengths)` for
  left-padded ragged batches; corresponding overloads on `DecoderBlock`, `AttentionMask`, and
  `RopeSpec` for batch position/mask data. Keep the current `forward(tokenIds, caches)` uniform-only.
  Add **default**, nonabstract methods to `CachedAttention` for batch metadata, preserving external
  subclasses; its old `[sequence,keyLength]` mask contract remains on the old overload. Update
  Javadocs and test that old implementations still compile and reject unsupported ragged input.
- [ ] Guard `GroupedQueryAttention` and `MultiHeadAttention` against windowed, evicted or
  nonuniform cache positions before their current scalar-RoPE/implicit-causal paths. Test rejection
  for each class, including a padded batch, while preserving legacy full-cache outputs.
- [ ] Define a batch position/mask input for `[B,T]` that represents each row's absolute next
  position and valid key span. Build BOOL masks of shape `[B,1,T,S]` (or a proved equivalent MLX
  broadcast shape), including causal, window, and per-row valid-length rules. Reject ragged input
  without supplied lengths; never treat padding as a real key. Keep the single-batch fast path.
  Apply RoPE at each row's absolute position; a scalar offset is valid only for uniform rows. Define
  the new core facade for the already bound `mlx_fast_rope_dynamic` array-offset call, and probe
  its offset broadcasting with the left-padded layout. If that call cannot express per-row offsets,
  use per-row slicing/concatenation and benchmark its cost. Reject unequal positions for Dynamic NTK.
  This is a primitive for Phase 6.5; no scheduler is added here.
- [ ] Confirm `MLXFast.scaledDotProductAttention` accepts the proposed batched mask shape and
  causal flag combination on the pinned runtime. Add a focused native probe if the contract is
  unclear. Probe nonzero key starts, unequal row lengths, batch reorder, and all-masked invalid
  positions before using the mask in production. Name the suites `BatchAttentionMaskTest` and
  `BatchRopeProbeTest` so CI can assert they ran.
- [ ] After the baseline, profile concatenate bytes per step and evaluate a preallocated capacity
  buffer updated through the already bound `mlx_slice_update`; for sliding windows, compare a ring
  buffer with materialized eviction. Add a core facade and inventory entry only for a measured
  improvement that preserves ownership, masks and goldens.
- [ ] Preserve the current `mistral.json` byte-for-byte and extend `tools/hf-reference/` to generate
  new independent Mistral window-4 goldens for prompts shorter than, equal to, and longer than the
  window, chunked prefill, and multiple decode steps. Commit new goldens and the updated
  `provenance.json` file entries so `verifyHfReferenceGoldens` checks them. Include a sliding-window
  older-token-independence test and an explicit test
  at the `p - j == window` boundary. Respect the tiny fixture's
  `max_position_embeddings = 128`; any test beyond that bound needs a new checkpoint under
  `tools/hf-reference/goldens/checkpoints/` and its files listed in `provenance.json` for
  `verifyHfReferenceGoldens`. Dynamic-NTK chunking remains governed by its documented 6.3 behavior;
  do not claim chunk-invariance for that scaling mode.

### 4. Add cache quantization only on a successful probe

The pinned native probe failed the attention-memory gate. The two unchecked implementation items
below move to **6.4.1** (`req/plans/phase6-4-1-plan.md`), after this 6.4 PR closes and before 6.5.
The explicit unsupported policy factories remain in place until 6.4.1 passes its own gate.

- [ ] If Task 0 proves a usable representation, add an opt-in quantized cache policy with bits,
  group size and an explicit accuracy envelope. Keep model weights unchanged. Quantized keys and
  values, scales, and any metadata must obey the same reset/fork/reorder/evict ownership rules.
  Reject unsupported dtype, head size, group size or runtime capability before generation starts.
- [ ] Compare float and quantized prefill/decode on deterministic tiny fixtures for every family,
  reporting token IDs, max/mean logits error, memory savings and decode throughput. Exact token
  equality is required only for fixtures whose reference margin supports it; otherwise state the
  measured tolerance and any token divergence. The default float path retains its existing greedy
  contract; the opt-in quantized mode gets a separately documented accuracy contract.
- [x] If the probe fails, record the reason and keep quantized cache creation as a named unsupported
  capability. Do not silently accept a request and use float cache.

### 5. Finish the benchmark and acceptance evidence

- [ ] Benchmark cold load, prefill, one-token decode, sustained tokens/s and peak native active
  memory for **each supported family** at documented prompt/context lengths and batch sizes.
  Use the committed deterministic tiny checkpoints for all six family rows as the mandatory
  reproducible comparison, explicitly labeling these numbers **synthetic fixture performance**.
  Add separate pinned Tier-B results for accessible Llama, Qwen2 and Mistral. Gemma requires
  terms acceptance; public MIT-licensed Phi-3 is held by hosted-runner RAM; a runnable Mixtral
  artifact/runner is pending. Keep their numbers synthetic until Tier-B runs are feasible. The
  pinned Tier-B Mistral's window metadata must be inspected and recorded before claiming any
  real-artifact eviction evidence; its current local manifest does not establish a window. Do not
  present tiny Mixtral numbers as representative of full-scale MoE weights. Tier-B runs are manual
  or scheduled; save exact commands, model hashes and device details. Run deterministic tiny
  fixtures and memory regression tests in required native CI.
- [ ] In `jmlx-models`, test repeated generation on one loaded model after warm-up, cache reset
  and reuse, long-context full-attention generation up to capacity (bounded memory and matching
  reference output), capacity rejection just beyond it, and sliding-window generation past the
  window. Base single-request capacity boundary cases on
  `maxNewTokens == 0 ? 0 : (long) prompt.length + maxNewTokens - 1`; add a ragged-batch case
  where padded width, rather than a shorter row's length, reaches capacity.
  Sample active native bytes after scope cleanup and during a sustained capped decode. Set a
  documented tolerance based on fixture tensor sizes and allocator noise; assert the numeric
  plateau above rather than a fixed process RSS. Add named native suites:
  `DecoderCacheLifecycleTest` (repeated generation/reset), `DecoderCacheCapacityTest` (single and
  ragged-batch boundaries), `DecoderSlidingWindowTest` (Mistral fixture eviction), and
  `DecoderForwardPoisoningTest` (construction failure, injected step-boundary eval failure after
  mutation, and a generation-only non-finite-logits case using NaN output-projection fixture
  weights). Rename `.github/workflows/ci.yml`'s
  "Assert Phase 6.3 native suites executed" step to "Assert required native suites executed"
  and extend it to require `KVCacheTest`,
  `KVCacheMemoryPlateauTest`, `AttentionMaskTest`, `DecoderAttentionTest`,
  `BatchAttentionMaskTest`, and `BatchRopeProbeTest` under `jmlx-core`, plus all four new
  `Decoder*Test` suites under `jmlx-models`. Add a separate CI step that runs the
  tiny-fixture benchmark once, parses its JSON, and checks required fields without performance
  thresholds.
- [ ] Update `req/phase6-compatibility.md`, `req/phase6-tier-a-fixtures.md`, model README and
  benchmark instructions/results with the implemented cache modes and any quantization outcome.
  Keep family verification status tied to existing Tier-A/Tier-B evidence.

## Verification and exit gate

Run pure-Java tests and formatting on Ubuntu; run native-gated tests and the benchmark on a
bootstrapped macOS Apple Silicon host. Before each commit run `./gradlew spotlessApply`,
`./gradlew :check :jmlx-core:check :jmlx-models:check :jmlx-examples:check`, relevant Javadocs,
`./gradlew verifyMlxApiCallSites verifyMlxApiHeaderCoverage`, and `git diff --check`. Use
`@EnabledIfNativeAvailable` for native tests. Any new native facade gets an inventory override and
regenerated `req/mlx-api-inventory.md`. Keep `jmlx-models` free of direct `jmlx-ffi` calls.

Accept 6.4 when existing generation goldens are unchanged; cache ownership tests prove independent
forks and reorder; repeated generation and capped long decode show bounded active native memory;
long-context full-attention generation is correct up to capacity and rejected beyond it;
sliding-window masks and outputs match the new Mistral window-4 synthetic references across
prefill/decode and nonzero starts;
batch masks handle unequal positions without padding leakage; both other public attention classes
reject unsupported cache states; and a reproducible synthetic-fixture report contains tokens/s and
peak native memory for all six supported families, with separately labeled Tier-B runs for the
three currently runnable Tier-B families. The named unsupported quantized policy is the 6.4
outcome; enabling quantized retention requires the separate 6.4.1 gate.

## Source coverage

Every requirement from both sources, and where this plan covers it.

| Requirement | Source | Covered by |
|---|---|---|
| Prefill and decode as first-class paths | roadmap; phase6-plan 1 | Task 1 |
| No prompt rebuild or cached K/V recomputation during decode | phase6-plan 1 | Task 1 |
| Cancellation before prefill and between decode steps, not mid-prefill | phase6-plan 1 | Contracts (Cancellation); Task 1; Deferred |
| Cache capacity policies | roadmap; phase6-plan 2 | Contracts (Capacity); Task 2 |
| Sliding-window eviction | roadmap; phase6-plan 2 | Contracts (Capacity, Policy split and eviction trigger, Prefill semantics); Tasks 2-3 |
| Reset, fork, reorder for batched decoding | roadmap; phase6-plan 2 | Contracts (Ownership); Task 2 |
| Fork/reorder by copying; sharing only after a reviewed refcount design | phase6-plan 2 | Contracts (Ownership); Task 2; Deferred |
| Batch-safe positions and masks | roadmap; phase6-plan 3 | Contracts (Positions, Batch validity); Task 3 |
| Cache quantization only after a probe of representation, accuracy and ownership | roadmap; phase6-plan 3 | Task 0 probe; Task 4 failed-probe outcome; Phase 6.4.1 follow-up |
| Benchmark: cold load, prefill, one-token decode, sustained generation | roadmap; phase6-plan 4 | Tasks 0 and 5 |
| Benchmark: memory growth and peak native memory | roadmap; phase6-plan 4 | Tasks 0 and 5 |
| Report metadata: commit, Java, macOS/device, both native pins, model revision/hash, config, batch, context, warm-up, samples | phase6-plan 4 | Task 0 |
| Published (committed, documented) benchmark harness | roadmap | Tasks 0 and 5 |
| Optimize only after the baseline is recorded | roadmap | Task 0 (baseline gate); Task 3 (post-baseline buffer optimization) |
| Bounded-memory native tests: repeated generation, reset, long context, sliding eviction | phase6-plan 5 | Tasks 2 and 5 |
| Tiny fixtures in required CI; Tier-B manual/scheduled | phase6-plan 5 | Task 5 |
| Gate: bounded native memory in repeated generation | roadmap; phase6-plan gate | Exit gate |
| Gate: long-context and sliding-window behavior correct | roadmap; phase6-plan gate | Exit gate |
| Gate: reproducible tokens/s and peak memory for every supported family | roadmap; phase6-plan gate | Exit gate |

## Explicitly deferred

- Request batching, queueing, back-pressure and worker scheduling (6.5).
- Shared cache buffers or copy-on-write forks until a separately reviewed refcount/ownership design
  proves that append, eviction and close in one branch cannot invalidate another.
- Mid-prefill cancellation and interruption of native kernels until native support is reviewed.
- Performance claims across changed MLX pins, devices, model revisions or batch shapes.
