# Phase 6.1 implementation plan — generation and sampling

**Roadmap:** `req/plans/phase6-plan.md` §6.1
**Prerequisite:** Phase 6.0b, merged in PR #22 (`c4809f6`)

## Objective

Extend the common Llama/Qwen generation path from greedy-only decoding to deterministic,
request-local sampling. Implement the policy already represented by `GenerationConfig`: repetition,
frequency, and presence penalties; temperature; top-k, top-p, and min-p filtering; categorical
selection; and optional selected-token log probabilities. Preserve the existing synchronous event,
cancellation, stop-token, scope-ownership, and greedy-output contracts.

Phase 6.1 does not add batching, an asynchronous publisher, tokenizer-aware text deltas, model
families, or general array/random parity. “Streaming” in this milestone means the existing
one-token-at-a-time synchronous `GenerationEvent` callback, now carrying an optional selected-token
log probability.

## Accepted baseline

- PR #22's Java and native jobs passed. The native job ran the 6.0b selection/RNG probes on the
  pinned macOS/Apple-Silicon tuple and verified the inventory, header report, and Python oracle.
- `mlx_argmax_axis` retains first-index tie-breaking and is the greedy fast path.
- `mlx_random_key` plus `mlx_random_split_num` provides stable explicit request-local key streams.
  Global `MLXRandom.seed` is not permitted in generation.
- Axis-aware categorical selection is repeatable for a fixed explicit key. Selection outputs remain
  valid after input scopes close when the result belongs to the longer-lived scope.
- The inventory/call-site guard remains mandatory: model code must use `jmlx-core` facades and must
  not import `se.alipsa.jmlx.ffi`.

The first documentation change should amend all three provisional statements in
`req/plans/phase6-0b-probe-findings.md`: the “compiled on Linux” paragraph and Decisions bullets 2
and 4, which respectively retain “native execution remains the acceptance gate” and “native
execution remains required.” Replace them with PR #22's successful native job details. That is
evidence closure, not a new probe.

## Policy contract

Define these rules before implementation so native details cannot accidentally become public
semantics.

### Greedy and sampled modes

- `temperature == 0` is deterministic greedy mode. Apply repetition/frequency/presence penalties in
  FLOAT32, then call `argmaxAxis`; do not create or split a random key.
- In greedy mode, require the filtering defaults (`topK == 0`, `topP == 1`, `minP == 0`) and an empty
  seed. Reject non-default sampling-only fields rather than silently ignoring them; in particular,
  a present greedy seed would advertise reproducibility state that the no-RNG path never consumes.
- `temperature > 0` is sampled mode and requires an explicit seed. Enforce this cross-field rule in
  `GenerationConfig`'s compact constructor: `OptionalLong.empty()` throws
  `IllegalArgumentException` at configuration construction, before a model or native scope is
  involved. Do not generate an unrecoverable implicit seed and never use MLX's global RNG.
- `logProbabilities` is valid in either mode. Greedy reports `0.0` by explicit API convention, not as
  the `temperature → 0+` limit: a k-way tied maximum has limit `log(1/k)`. Sampled mode reports the
  selected token's natural-log probability after all filters and final renormalization.

Add a convenience factory for the common sampled case without replacing the existing record
constructor, for example `GenerationConfig.samplingDefaults(maxNewTokens, seed, temperature,
eosTokenIds)`. Require its temperature to be positive—`samplingDefaults(..., 0, ...)` is a policy
error, not an implicit switch to greedy—and keep `greedyDefaults` source-compatible.

### Ordered logits pipeline

For each decode step, process the final-token logits along the vocabulary axis in this order:

```text
model logits
  → cast to FLOAT32 and record a finiteness flag checked after the single evaluation
  → repetition penalty over prompt + previously emitted tokens
  → frequency penalty over prompt + previously emitted tokens
  → presence penalty over prompt + previously emitted tokens
  → greedy argmax, when temperature == 0
  → divide by temperature
  → descending vocabulary sort
  → top-k mask
  → top-p mask (preserve the boundary token)
  → min-p mask
  → scatter filtered logits back to original vocabulary order
  → categorical draw with this step's explicit child key
  → selected-token post-filter log probability, when requested
```

Penalty semantics are fixed as follows:

- repetition: for every seen token, divide a non-negative logit by `repetitionPenalty`, otherwise
  multiply it; apply once per distinct token;
- frequency: subtract `frequencyPenalty * occurrenceCount`;
- presence: subtract `presencePenalty` once when the occurrence count is nonzero;
- history includes the full prompt and every emitted token, including EOS; an explicit stop token is
  selected but neither emitted nor added to history because generation terminates immediately.

Filtering uses stable token identity, not an assumed tie order from `topk`. The Python binding
docstring for the exact pinned MLX `v0.31.2` shared core `argsort` operation—which mlx-c's
`mlx_argsort_axis` wraps—documents stability in
[`python/src/ops.cpp` lines 2852–2855](https://github.com/ml-explore/mlx/blob/v0.31.2/python/src/ops.cpp#L2852-L2855).
Apply ascending argsort to negated logits in original vocabulary order so the result is descending
by logit with token ID ascending as the secondary order. Add a native tie regression for the pinned
runtime, but treat the cited shared-core contract—not the observation—as the semantic basis. The
same docstring says NaNs sort last; explicit finite-logit rejection remains belt-and-braces validation
that provides a jmlx policy error instead of making sort placement load-bearing.

`topK == 0`, `topP == 1`, and `minP == 0` disable their stages. Top-p uses cumulative probability
after the top-k mask and keeps the first token whose inclusion reaches the threshold. Min-p then
keeps candidates whose current probability is at least `minP * maxProbability`. Every filter must
preserve at least the highest-ranked candidate. Compute masks in sorted order, then form the inverse
mapping by scattering the sorted filtered logits back with
`putAlongAxis(destination, sortedTokenIds, filteredSortedLogits, vocabularyAxis)`. The destination is
a vocabulary-shaped zero tensor, but its initial value is deliberately irrelevant: `sortedTokenIds`
is a full permutation and every destination position is overwritten. The scatter restores original
vocabulary order in O(V) without a second O(V log V) argsort. If a later optimization scatters only a
retained prefix, it must change the fill to negative infinity and add a regression for excluded
positions rather than reusing this full-permutation assumption. Call categorical only after the
restoration. Thus a fixed key is applied to the same index order as direct Python MLX/`mlx_lm`; the
oracle must not reproduce a jmlx-only sorted-domain draw.

This index-order parity is not a claim of complete sampler parity with `mlx_lm`. Phase 6.1 follows
the roadmap/Hugging Face stage order—temperature before top-k, top-p, and min-p—whereas current
`mlx_lm` applies temperature inside categorical after filtering. The Python oracle implements this
plan's declared stage order; matching `mlx_lm` output for `temperature != 1` with an active filter is
out of scope.

### Validation and terminal behavior

- Retain the current finite/range checks and add cross-field greedy validation. At generation entry,
  reject negative or out-of-vocabulary prompt/EOS/stop IDs and `topK > vocabSize` before model
  native work. Reject rather than clamp top-k because silently changing a policy across models hides
  a configuration error; document that callers must adapt a shared policy to each vocabulary.
- The public `Set<Integer>` fields cannot represent duplicate IDs. Preserve immutable-set copying,
  explicitly test EOS/stop overlap, and retain EOS precedence. Do not claim to reject duplicates
  that the API has already collapsed.
- Build the finite flag and selection from the same original-logit lazy graph rather than
  synchronizing before selection. Do not sanitize with `where`: no selected entry point in the
  pinned argmax/categorical path reports NaN as a native status failure, and an extra vocabulary-
  sized replacement kernel would not improve attribution. Cast the BOOL finite scalar to INT32,
  then call `MLX.eval(finiteFlagInt32, selectedIndex, selectedLogprob)` once (omitting the logprob
  output when disabled). Read and inspect the already-evaluated finite flag before returning or
  emitting the selected token. This preserves fail-loud attribution without a pre-selection GPU
  submission/host round-trip. An all-masked/empty distribution is an internal error with the stage
  named in the message, never a native categorical failure.
- EOS remains in generated IDs and receives a token event/log probability. An explicit stop token
  remains excluded from generated IDs/events/log probabilities. When log probabilities are
  requested, `GenerationResult.logProbabilities().size()` equals `generatedTokenIds().size()` and
  each token event carries the matching value; otherwise the result list is empty and token-event
  values are null. With zero generated tokens both states are represented by an empty list, so the
  result constructor alone cannot distinguish them; the generation path enforces the policy-specific
  invariant.
- Cancellation before prefill or between steps consumes no new key. Once a categorical draw begins,
  that step's split is consumed even if the selected ID terminates generation. Listener-failure and
  terminal-event behavior remain unchanged.

## Implementation slices

### 1. Narrow `jmlx-core` facade additions

Add only the operations required by the pipeline, following existing `MLXScope`, `NativeOps`, axis,
dtype-conversion, and error-reporting patterns:

- `MLXOps.argsortAxis` for canonical vocabulary order;
- `MLXOps.cumulativeSumAxis(a, axis, reverse, inclusive)` for nucleus filtering, exposing all
  parameters of `mlx_cumsum` and following the existing axis-suffixed facade naming;
- `MLXOps.isFinite` plus scalar `MLXOps.all` for fail-loud logit validation;
- `MLXOps.logSumExpAxis` for stable selected-token log probabilities;
- `MLXShape.takeAlongAxis` and `putAlongAxis` for sorted lookup, vocabulary-order scatter, and sparse
  penalty updates;
- `MLXRandom.key(scope, seed)`, `MLXRandom.split(key, count, targetScope)`, and
  `MLXRandom.categorical(logits, axis, key)`.

The random contracts expose keys as scope-owned UINT32 arrays: `key` has shape `[2]`, `split` has
shape `[count, 2]`, and `categorical` converts native UINT32 indices to jmlx's INT32 index dtype.
Require `count > 0`, a live key with dtype UINT32 and shape `[2]`, a valid axis, FLOAT32 logits, and
scope compatibility before invoking native code. The explicit split target must be the key scope or
a descendant, allowing per-step split temporaries to live in the activation scope. Document that
split does not mutate its parent.

Document `takeAlongAxis` beside the existing `takeAxis` with an explicit contrast: `takeAxis`
inserts the full index-array shape at one axis, while `takeAlongAxis` performs elementwise aligned
gather and requires its index shape to be broadcast-compatible with the non-selected dimensions.
Give `putAlongAxis` the corresponding aligned-update contract and keep it lazy like `take` and
`takeAxis`; do not read back an index tensor inside a general facade. Its Javadoc must warn that
native scatter does not bounds-check index values. The sampling path validates history token IDs in
Java before constructing their index array, while argsort-produced permutation indices are bounded
by construction. Test both source validations without turning the public operation into an eager
synchronization point.

Do not expose the flat sort/top-k/partition variants. The sampling pipeline does not need them, and
the 6.0b probe proved they flatten decoder logits. Keep their exact-source probe inventory records;
split selected identities into implemented facade records while leaving non-selected candidates
explicitly probe-only/planned for later parity work. Regenerate and verify both inventory artifacts
as applicable.

As part of slice 1—not as a later cleanup—add explicit non-`unplanned` override records for every
new call site: `mlx_argsort_axis`, `mlx_all`, `mlx_cumsum`, `mlx_isfinite`,
`mlx_logsumexp_axis`, `mlx_take_along_axis`, `mlx_put_along_axis`, `mlx_random_key`,
`mlx_random_split_num`, and `mlx_random_categorical`. Point each implemented record at its facade
and tests; keep the existing exact-source records only for direct probe calls. Run
`generateMlxApiInventory` before the call-site guard.

Add native core tests for shapes/dtypes, negative axes where supported, invalid axes/count/key
shapes, stable split bytes, parent-key immutability, categorical fixed-key output, lazy ownership,
and cleanup on native failure.

### 2. Internal sampling engine

Create a package-private `SamplingPipeline` in `jmlx-models`, independent of Llama/Qwen classes. It
accepts final-token logits, immutable policy, vocabulary size, token-frequency state, and an optional
request RNG state, and returns a small immutable selection containing token ID and optional log
probability.

Keep `SamplingPipeline`, `PenaltyInputs`, and their test-accessed helpers package-private and test
them from the same Java package. They do not appear in generated Javadocs or the published public API;
do not add public/reflective diagnostic bridges for tests.

Keep token counts in a Java insertion-ordered map keyed only by distinct prompt/emitted token IDs and
update it once per emitted token. Use a sparse native update, not a dense `float[vocabSize]` upload:
materialize `[distinctSeen]` token IDs and counts in the per-step activation scope, gather only those
logits with `takeAlongAxis`, apply repetition/frequency/presence math to that compact vector, and
write the adjusted values back with `putAlongAxis`. The host and transfer work is therefore
O(distinct history), not O(vocabulary), while the logits remain on-device. Represent these actual
inputs with a package-private immutable `PenaltyInputs` value used by production code, not a
test-only diagnostic hook. A pure-Java test with a 151,936-token vocabulary and short history asserts
that its ID/count arrays are distinct-history-sized without allocating a dense vocabulary vector.

For sampled mode, hold exactly one current key in the generation scope. Per step, create the split
result and child views in the activation scope: child 0 is hoisted as the next generation-scope key
and child 1 is the categorical draw key. After those result graphs have captured their inputs, call
`currentKey.close()`—which routes through `MLXScope.free`—before assigning the successor. The
activation close releases the split array, draw-key view, and every other temporary. At all times the
generation scope therefore owns at most one live current key plus the newly hoisted successor during
replacement; it does not accumulate one key per token. Pin lazy safety with a test that closes the
old key before first evaluation of the categorical result. This fixed child ordering and early-
release sequence are part of the reproducibility tuple.

Implement filtering on sorted logits. Use argsort plus token-ID secondary ordering, gather logits,
and apply positional/cumulative masks with negative infinity. Scatter the filtered sorted logits into
a vocabulary-shaped zero destination at the full `sortedTokenIds` permutation; every element is
overwritten, so the fill value is not part of the result. Do not sort the permutation a second time.
Call categorical only on that vocabulary-ordered tensor, so fixed-seed behavior matches direct
Python MLX's categorical index domain rather than a permuted domain.

For an optional log probability, gather the selected filtered logit and subtract
`logSumExpAxis(vocabularyOrderedFilteredLogits, vocabularyAxis, true)`. Do not calculate
`log(softmax(...))`, which is less stable at masked `-infinity` entries, and do not expose
alternative-token log probabilities in 6.1.

Build the finite flag, categorical/argmax result, and optional logprob as one lazy graph and evaluate
their terminal scalars together. The implementation invariant is exactly one explicit `MLX.eval`
per step and no `toIntArray`/`toFloatArray` readback before it; enforce this in code review and keep
the terminal readbacks adjacent to that evaluation. Do not add evaluation counters or test-only
diagnostic seams to shipped classes merely to restate this visible control-flow rule.

### 3. Decoder integration and public results

Replace `DecoderModel.requireGreedy` with policy validation and a single call to
`SamplingPipeline` after the existing final-token logits slice. Keep cache construction, prefill,
single-token decode, cancellation polling, and activation-scope boundaries unchanged.

- Initialize token counts and request RNG once per `generate` call.
- Preserve current token output for finite logits under the default greedy policy and keep its direct
  argmax fast path. Deliberate behavior changes are: non-finite logits now fail before argmax;
  selected log probabilities become legal; and invalid seed/temperature/filter combinations fail at
  `GenerationConfig` construction rather than later in `DecoderModel.requireGreedy`. In particular,
  the current `LlamaModelTest.rejectsNonGreedyPolicy` temperature-1/empty-seed fixture changes from an
  `UnsupportedOperationException` at `generate` to an `IllegalArgumentException` at construction.
  Rename/update that test and add separate valid sampled-generation coverage; this observable timing
  and exception-type change is accepted because `jmlx-models` remains unpublished
  `0.1.0-SNAPSHOT`.
- Populate `GenerationEvent.logProbability` and `GenerationResult.logProbabilities` only when
  requested; otherwise retain null event values and an empty result list.
- Leave `textDelta` null. Tokenizer-aware incremental Unicode decoding belongs to Phase 6.2.
- Preserve the legacy `DecoderModel.generate(int[], ...)` overload as greedy-only.

Strengthen constructors so `GenerationEvent` rejects log probabilities on terminal events and
`GenerationResult` permits only an empty logprob list or one whose cardinality equals generated
tokens. The generation path—not the result value alone—distinguishes “not requested” from “requested
with zero generated tokens.” Update Javadocs on `GenerationConfig`, `GenerationEvent`,
`GenerationResult`, and `TextGenerationModel` with the exact distribution and stop/EOS rules.

### 4. Oracle and deterministic fixtures

Add a separate canonical sampling fixture under `tools/mlx-oracle/fixtures`. It records:

- input FLOAT32 logits and history counts;
- every post-policy distribution stage for hand-review;
- explicit seed/key split convention;
- selected token and selected-token post-filter log probability;
- at least greedy, penalties-only, top-k, top-p boundary, min-p boundary, and combined-policy cases.

Extend `runner.py` with a fixtures-directory mode and generalize the Gradle declarations rather than
adding a second hard-coded invocation. Define the `*.input.json` file tree as generation inputs, map
every input basename to a declared `*.expected.json` output, and declare both file trees as verify
inputs. Generation must reject an orphan expected file and verification must reject either an input
without expected output or an expected output without input. This makes adding/removing/renaming any
fixture invalidate Gradle up-to-date state while retaining explicit generation and read-only
verification.

Do not change the existing array fixture's meaning. Unlike Phase 6.0b, close the sampling
differential loop: a native Java test reads the committed Python sampling JSON and compares Java
stage distributions/selections on the same pinned tuple. The Python runner computes masks in sorted
order, scatters them back to vocabulary order, and calls `mx.random.categorical` there exactly as
the Java design does. It must also apply temperature before filtering even though current `mlx_lm`
orders those stages differently. Keep ordinary Java tests independent of Python execution; only
committed JSON crosses the boundary.

Add a small native subprocess probe that runs the same seeded request in two fresh JVMs and compares
canonical token/logprob output. Use committed exact outputs for two deliberately chosen seeds to
prove both same-seed repeatability and different-seed divergence without a probabilistic assertion.

### 5. Tests, documentation, and CI

Test in layers:

- pure Java: config cross-field validation, token-ID/vocabulary checks, immutable inputs, result/event
  cardinality, EOS/stop precedence, and history-count updates;
- native pipeline: one focused test per ordered stage, combined policy, finite-logit rejection,
  top-p/min-p boundaries, ties, no empty candidate set, selected log probability, and key ownership;
- model integration: Llama and Qwen tiny checkpoints, unchanged greedy goldens, seeded repeatability,
  logprob event/result alignment, cancellation, listener failure, and every terminal path;
- resource regression: repeated sampled generation and failures remain within the existing bounded
  native-memory allowance;
- oracle/subprocess: committed Python differential cases and fresh-JVM reproducibility.

Update `jmlx-models/README.md` with greedy and seeded examples, the reproducibility tuple (jmlx
commit, MLX pins, macOS/device, model/checkpoint, prompt IDs, complete policy, seed, and batch shape),
post-truncation logprob semantics, synchronous callback threading, and explicit unsupported
alternative logprobs/text deltas. Update `req/phase6-tier-a-fixtures.md` so the Python sampling row
is added and names its Java consumer. Replace the current blanket statement that Java-side JSON
comparison belongs to a later milestone: the original array-oracle row remains a Python environment
self-check, while the new sampling row is consumed by Java in 6.1.

The existing macOS ARM64 job remains the required native gate. Add the sampling oracle fixture and
fresh-JVM probe to that job if they are not already reached through module `check`; the Ubuntu job
continues to run pure-Java contract tests and repository inventory/call-site verification without
installing MLX Python.

## Change map

| Area | Planned change |
| --- | --- |
| `jmlx-core` | Axis sorting/gather/cumulative/finite facades and explicit key/split/categorical APIs. |
| `jmlx-models` | Ordered sampling pipeline, request-local RNG state, decoder integration, selected logprobs. |
| `tools/mlx-oracle` | Canonical sampling-stage fixture and multi-fixture generation/verification. |
| `req/` | Inventory classifications, Tier-A differential evidence, closed 6.0b native-run note. |
| CI/docs | Native reproducibility probe and public sampling/reproducibility contract. |

## Verification and acceptance gate

On every platform:

```text
./gradlew -p buildSrc check
./gradlew :check
./gradlew :jmlx-ffi:check
./gradlew :jmlx-core:check
./gradlew :jmlx-models:check
./gradlew :jmlx-tokenizer:check
```

On Apple Silicon after native/oracle setup:

```text
./scripts/bootstrap-native.sh
./tools/mlx-oracle/install.sh
./gradlew verifyMlxApiHeaderCoverage verifyMlxOracle verifyMlxOracleFixtures
./gradlew --no-build-cache :jmlx-core:check :jmlx-models:check
```

Accept Phase 6.1 only when:

- every non-default policy stage has an isolated golden and a combined-policy golden;
- the legacy and common greedy paths retain their existing token outputs and consume no RNG key;
- explicit-seed tokens and selected log probabilities repeat in-process and across fresh JVMs on the
  recorded tuple;
- cancellation or completion of one independently executed request cannot perturb another request's
  seeded sequence;
- stop/EOS/logprob cardinality and listener behavior match the documented contract;
- repeated success, cancellation, stop, EOS, and failure paths release native memory;
- Python oracle verification and Java consumption of its committed sampling fixture both pass; and
- inventory rendering, native-call guarding, formatting, Checkstyle, Javadocs, and all CI jobs pass.

## Non-goals

- Batched generation, schedulers, key splitting across a batch, beam search, speculative decoding,
  or cache fork/reorder; these belong to Phase 6.4/6.5.
- Tokenizer-aware `textDelta`, chat-message request types, or prompt-template policy; Phase 6.2 owns
  them.
- Alternative/top-N log probabilities, logits return, custom processor plugins, grammar-constrained
  decoding, or user-supplied native keys.
- General exposure of flat selection/partition/random functions merely because bindings exist.
- Reproducibility claims across changed native pins, OS/device, model revision, prompt, policy, or
  batch shape.
