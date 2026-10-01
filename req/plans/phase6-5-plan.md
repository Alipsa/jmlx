# Phase 6.5 implementation plan — batching and first Central release

**Sources:** `req/full-roadmap.md` §6.5 and `req/plans/phase6-plan.md` §6.5. The
6.4.1 packed-KV prototype recorded a stop; it does not block this milestone and
compressed-cache requests continue to fail explicitly. The first release covers local
text generation on macOS Apple Silicon with Java 25 and the pinned MLX runtime. It does
not add an HTTP server or claim arbitrary Hugging Face checkpoint compatibility.

**Goal:** A clean Java consumer can resolve the six intended Central modules, render a
supported chat prompt, stream deterministic sampled output, run bounded concurrent
requests through an in-process scheduler, and close all native resources. The scheduler
has one MLX-owning worker; callers can submit and cancel from other threads.

## Starting point and prerequisite evidence

- `DecoderModel.generate` is a synchronous, single-request loop. It already defines
  event order, token/stop semantics, cancellation polling, partial results on listener
  failure, per-request sampling, and scope cleanup. Keep it as the direct API.
- `DecoderModel.forward(ids, caches, validLengths)` and `KVCache` have left-padded
  batch positions, masks, capacity checks, and reorder/fork primitives. The scheduler
  must use a shared internal batched prefill/decode path; invoking `generate` once per
  queued request would not satisfy the batching gate.
- `SamplingPipeline` currently selects one `[1,1,V]` logits row with one RNG key.
  Extract a request-local selection/decoder state so batched logits can be sliced by
  row without sharing a random stream. Preserve the direct path's selection order and
  event semantics.
- `req/phase6-compatibility.md` and `req/phase6-tier-b-artifacts.md` record real-artifact
  evidence for Llama, Qwen2, and Mistral. Gemma v1, Phi-3, and Mixtral currently have
  synthetic evidence only. The release report must say so; a new real-artifact claim
  requires the Phase 6 family gate (pinned hashes, license/access, tokenizer and model
  goldens, and a recorded run). Do not hold the scheduler work for unavailable gated or
  oversized artifacts.
- Before release, confirm 6.4's bounded-memory, long-context, and per-family benchmark
  evidence in `req/phase6-4-benchmark.md`. Fill any missing evidence or explicitly
  narrow the release claim; a checked box in an older plan is not a substitute for a
  measured result.

## 1. Settle the public scheduler contract

Add `BatchGenerationScheduler` and a small immutable `BatchSchedulerConfig` in
`jmlx-models/src/main/java/se/alipsa/jmlx/models/`. Expose one asynchronous
`submit(GenerationRequest, Consumer<GenerationEvent>)` returning a request handle with
a read-only completion stage and a thread-safe `cancel()` signal. The handle owns no
native object. Document these behaviors in Javadoc before implementing the worker:

- Make a worker-owned factory the only constructor path:
  `BatchGenerationScheduler.start(config, modelFactory)`, where the checked factory
  receives the root `MLXScope` and returns `TextGenerationModel`. The worker creates
  that root, invokes the factory there, verifies the result is a `DecoderModel` whose
  model scope is the root or its descendant, then publishes a ready scheduler. Add a
  package-private, worker-thread-only scope accessor on `DecoderModel` because its
  inherited `Module.scope()` is protected in `se.alipsa.jmlx.nn`; check
  `root.isAncestorOf(model.modelScope())` before accepting the factory result. A
  factory that returns a model from an unrelated scope fails startup with a named
  error. Startup failures close the factory result where possible and the root, then
  propagate to the caller. `TextGenerationModels.load(scope, path)` already returns
  the interface, so this accepts the existing loader without changing its public
  return type. Do not accept a caller-loaded model or scope: `MLXScope.newChild()` and
  every weight access check the owning thread. The worker alone closes the model and
  root when the scheduler closes. Submission and cancellation never touch MLX arrays
  or caches.
- Complete request stages on a dedicated, bounded completion dispatcher so synchronous
  `CompletionStage` dependents never execute on the MLX worker. Reserve one completion
  slot for every accepted request at submission; retain it through queued and active
  states and release it only after the dispatcher has run that request's completion.
  The dispatcher queue plus running completions is therefore bounded by the total
  admission permits (`maxQueuedRequests + maxBatchSize`), and worker handoff never
  waits for queue space. Exhausted permits reject new submissions immediately. Expose
  a read-only `CompletionStage` facade; `toCompletableFuture()` must return a defensive
  copy, not the internally completed future. The handle's `cancel()` atomically ORs
  its signal with `GenerationRequest.cancellationToken()`. Stage cancellation is not
  a request cancellation API; a caller must use the handle. Document that completion
  observers must return promptly and must not block on another request's stage; doing
  so can deadlock a bounded dispatcher. Reject `close()` from the worker or completion
  dispatcher with `IllegalStateException` to prevent self-join.
- Configure positive `maxBatchSize`, `maxQueuedRequests`, and a bounded prompt-token
  budget for a batch. Count prefill work as `rows × maximum padded prompt width`,
  including left padding, rather than summing valid prompt lengths. Reject a prompt
  exceeding the budget even alone. A full queue causes immediate, named rejection
  with no request side effects; it never grows implicitly.
  An active batch holds at most `maxBatchSize` requests; waiting requests count against
  `maxQueuedRequests` and every accepted request also holds one total admission permit
  through completion dispatch. The oldest waiting request determines the next
  cohort's compatibility group; scan later requests only to fill that cohort without
  exceeding either bound.
  Add a scheduler `maxNewTokensPerRequest` cap and reject larger requests before
  enqueue. With no mid-decode joins, the next cohort waits at most that many active
  decode steps plus callback time; document this step-count bound and verify that the
  oldest waiting request runs next. There is no wall-clock bound while callbacks may
  block.
- A callback runs synchronously on the worker, receives only immutable Java events,
  and must return promptly. It must not call a blocking scheduler operation or close the
  scheduler from that worker. `submit` remains nonblocking even when invoked by a
  callback; `close()` from that worker throws. No unbounded output event queue is added.
  A slow callback delays the worker; document that operational limit and measure it
  in the benchmark. Never wait for request completion on the worker.
- Cancellation before admission completes with `CANCELLED` and one terminal event;
  cancellation in flight is observed before prefill or between decode steps, never in
  the middle of a native call. At **every** active step boundary, the worker also
  re-checks all waiting requests' own cancellation tokens and handle signals. It
  removes cancelled requests from the queue immediately, emits their one `CANCELLED`
  terminal event on the worker, and hands completion to the dispatcher; their queue
  slots are then free, while admission permits remain held until completion dispatch
  finishes. Removing one request never cancels or changes another.
  After a token-listener exception, close that request's state and complete its future
  exceptionally with the existing `GenerationAbortedException` partial result; emit no
  terminal event for it. A terminal-listener exception follows the direct API's logged,
  successful-result rule. `close()` atomically stops admission; a racing or later
  `submit` gets a named `SchedulerClosedException`, distinct from the named
  `BatchQueueFullException` for capacity rejection. The worker treats every remaining
  queued and active request as cancelled at its next safe boundary: each gets a
  `GenerationResult` with `FinishReason.CANCELLED`, its partial generated IDs, and
  exactly one terminal event delivered on the worker, including tokenizer flush if
  applicable. Then the worker closes all request scopes, model, and root. `close()`
  joins the worker and waits for the completion dispatcher to deliver every accepted
  request's stage before returning; it never abandons an accepted stage. Document that
  a callback that never returns can stall worker shutdown and an observer that never
  returns can stall dispatcher drain and `close()`; there is no forced timeout.
- Match direct `GenerationResult` and `GenerationEvent` semantics: EOS token included,
  explicit stop token excluded, one terminal event for normal completion, tokenizer
  flush delta, optional post-filter log probability, and `maxNewTokens == 0` without
  prefill. Do not change `GenerationConfig`'s record shape or direct `generate` signature.

## 2. Build the batched execution path

Implement a worker state machine in `jmlx-models` with one request context per admitted
request: prompt IDs, cache policy, token-frequency map, generation count, independent
`SamplingPipeline`/RNG key, incremental text decoder, event listener, result builder,
and cancellation signal. Validate vocabulary IDs, tokenizer size, cache policy, and
bounded-FULL worst-case capacity before admission. A bad request fails its own handle
without poisoning the queue.

**Scope layout:** On the worker, create one model root; create each active cohort's
scope `C` as a child of the validated `DecoderModel.modelScope()`, so weight scopes
remain ancestors of every forward activation even when the factory used a child of
the root for the model. Cache sets live in `C`. Each request owns a child of
`C` containing its sampler key and rank indices. For **every forward step**, create
`S = C.newChild()`, allocate the batched token IDs in `S`, and run the forward there.
Activations, masks, RoPE frequencies and logits therefore live in `S`, not `C`;
`KVCache.append` hoists retained K/V into `C`. For each row, create a short-lived
row-step scope under its request scope and use the explicit-destination
`MLXShape.takeAxis(logitsInS, indexInRowStep, rowStep, axis)` overload to copy that
row's logits. This overload has no scope-relation check; the resulting row logits,
sampler key and rank indices have one request-scope ancestry chain. Never combine
operands from sibling requests. The joint evaluator may receive arrays from these
different scopes on the same owning thread. After joint evaluation **and readBack**,
close row-step scopes, then `S`; close a request scope only after row removal. Reorder
into `C`. Add a native test for this full scope chain, including filtered sampling,
key hoist, forward and cache retention, before relying on it in the scheduler. Make
the cross-scope `takeAxis` rule explicit in that overload's `jmlx-core` Javadoc:
source and destination may be unrelated, provided the result is evaluated before
the source scope closes. Add a `jmlx-core` native test with unrelated sibling scopes
that proves the copied result survives source-scope close; require it in native CI.

1. Extract a package-private decoder step API shared with `DecoderModel.generate`:
   batched prefill accepts left-padded `[B,T]` IDs and `validLengths`, returns the last
   valid logits for each row; decode accepts `[B,1]`. Use one cache set per compatible
   active group with cache row order matching request-context order. Keep the model's
   cache-set preflight and poisoning rules. Before scheduler integration, add one
   native model test per supported family comparing left-padded batched prefill and
   decode logits to independent single-row logits within a stated numerical tolerance.
   Include per-row masks and RoPE positions, Gemma embedding scaling, Phi-3 fused
   projections, and Mixtral expert routing at `B > 1`. Require these suites in CI.
   Avoid reimplementing attention or sampling in the scheduler.
2. Form a prompt batch within `maxBatchSize` and the prompt-token budget. Group by
   resolved cache policy and any architecture/runtime restriction that cannot safely
   share a cache (including Dynamic-NTK unequal-position limits). Left-pad variable
   prompt lengths only when the native batch path supports them; otherwise split into
   equal-length groups and record this as a batching limit. Never silently replace a
   requested policy. Do not introduce mid-decode joins unless cache assembly and
   position semantics are proved; finishing the current cohort first is the initial
   single-worker policy.
3. Split `SamplingPipeline.select` into lazy `build`, one joint `evaluate`, and
   `readBack` phases, retaining its direct-path wrapper. At each step run one model
   forward, build each live row's selection with its own policy, penalty history and
   RNG key, then evaluate **all** selected IDs, requested log probabilities, finite
   checks, and changed cohort cache arrays in one call through the model's existing
   `StepBoundaryEvaluator` (native default delegates to `MLX.eval`). Expose that
   package-private hook to the scheduler so `DecoderForwardPoisoningTest` can inject
   a joint-evaluation failure; do not call `MLX.eval` directly here. Read back each
   row only after the joint call succeeds. This gives one joint synchronization for
   `B` rows; a subsequent compaction may require its own copy evaluation. Attribute
   a joint evaluation failure to the shared cohort. Greedy selection consumes no RNG.
   Test that cancellation, reorder, and a different companion request do not advance
   another row's key or step count.
4. Classify failures at the step boundary. Request validation, a non-finite row's
   logits discovered after a successful joint evaluation, tokenizer output-decoder
   failure (including final flush), and token-listener failure affect only that row;
   use the direct path's `GenerationAbortedException` stage/partial-result conventions
   for decoder and listener failures. Forward construction, selection-graph/native
   errors, or joint evaluation failure poison the whole cohort cache set and fail
   every active row with a named cause. Do not infer a row-only native failure from
   the order in which arrays were passed to `MLX.eval`. Queued requests may start
   with fresh caches only if the model and worker remain valid.
5. After selection, emit events in stable row order. Collect all finished, cancelled,
   and row-failed indices from one step, then perform **one** `KVCache.reorder` per
   layer for the surviving rows; never compact once per departing row. Reorder
   materializes a copy while source caches remain live, so reset all superseded
   caches immediately after the new cache set has evaluated and been installed. If
   layer `k` reorder fails, reset every partially created new cache from layers
   `0..k-1`, keep the old set only long enough to report a named cohort-wide
   compaction failure, then reset/close it and fail all surviving rows. Do not
   continue decode with a partly reordered cache set. If no rows survive,
   reset/close the old set without reorder. Check that retained cache data are
   independent of closed request scopes. Put the per-layer reorder call behind a
   package-private scheduler seam so an injected failure at an interior layer can
   assert cleanup and cohort failure without relying on a real Metal fault.
   Benchmark peak active native bytes during compaction, including the temporary
   old-plus-new cache sets.

**Native tests:** mixed prompt lengths, uneven output lengths, EOS/stop/max-token paths,
FULL and sliding policies, capacity rejection, Dynamic-NTK grouping, cancellation
before/after prefill, queued cancellation polled during an active cohort, callback
failure, distinct full-queue/closed rejection races, close with queued/active requests
and drained completion stages, one-step multi-row compaction, and repeated batches
with a stable active-memory plateau. The plateau assertion must sample active bytes
**within one long cohort** across many decode steps, as well as after repeated cohorts, so a
per-step activation leak cannot hide behind end-of-cohort cleanup. Exercise a worker
thread after `NativeOps.DEFAULT_STREAM` was initialized on a different thread, then
load and generate entirely on that worker; this is the required pinned-runtime
cross-thread stream probe, not an assumption of
Metal command-encoder safety. If it fails, change the native stream ownership or
initialization design and repeat the probe before enabling the scheduler.
Assert required native test XML in `.github/workflows/ci.yml` so an unbootstrapped run
cannot appear green through skipped tests. After cancelling a row, assert the other
row's RNG key sequence and decode-step count are unchanged; compare its events
exactly with a reference run using the **same cancellation schedule and batch shapes**.
Compare its logits with an uncancelled run only within a documented numerical tolerance,
since compaction changes batch shape. Retain the roadmap's no-cross-batch-shape token
identity rule. Include a measurable proof that a multi-row cohort uses fewer model
forward calls than one forward per request.

## 3. Consumer example, benchmark, and support report

- Add an opt-in `jmlx-examples` program using a local checkpoint and paired tokenizer:
  construct the scheduler with
  `start(config, scope -> TextGenerationModels.load(scope, checkpoint))`, render a
  chat request, stream sampled text, submit two or more requests, cancel one, await
  all results, and close the scheduler. The worker closes its model and scope. Take
  paths and limits as arguments or environment configuration; no checkpoint download
  during required PR checks.
- Extend `DecodeBenchmark` to record batch size, queue depth, prompt lengths,
  heterogeneous completion lengths, throughput, per-request first-token and total
  latency, and active/peak native memory. Report direct single-request and batched
  runs with exact commands, hashes, pins, device, warmup, and sample distribution in
  `req/phase6-4-benchmark.md` or a linked Phase 6.5 report. No hard speed threshold
  without baseline evidence; enforce bounds and correctness in CI.
- Add model download/cache guidance to `jmlx-models/README.md`: user-managed local
  directories, revision pinning, hashes, license acceptance, size/eviction policy,
  pairing checkpoint and tokenizer, and the Tier-B manifest. A resolver is optional
  and separately scoped; examples should work with already-downloaded artifacts.
- Produce `req/phase6-inference-report.md` from the compatibility matrix, Tier-A/Tier-B
  results, benchmark method/results, package smoke, native pins, notices/licenses, and
  explicit unsupported features (including packed KV, GGUF variants not implemented,
  gated/pending real artifacts, and unsupported model families). Review every
  `planned`/`unsupported` row before release; do not upgrade a synthetic result to
  real-artifact verification.

## 4. Package and release verification

Create a small independent Gradle consumer under `tools/release-smoke/` with no
`project(...)` dependencies or checkout classpath. Its dependencies are the published
`se.alipsa:jmlx-models` coordinate and runtime-only
`se.alipsa:jmlx-native-macos-arm64`; transitive resolution must bring in
`jmlx-core`, `jmlx-ffi`, `jmlx-tokenizer`, and `jmlx-jinja`. Compile and run on a fresh
Gradle home on macOS ARM64 with Java 25 and `--enable-native-access=ALL-UNNAMED`, with
no `jmlx.library.path` or `JMLX_LIBRARY_PATH` override. Use the committed synthetic
Mistral checkpoint at `tools/hf-reference/goldens/checkpoints/mistral/` and its paired
chat tokenizer at `jmlx-tokenizer/src/test/resources/families/mistral/` (tokenizer
vocabulary fits the checkpoint's 128-token vocabulary). A smoke setup task copies
`config.json`, `generation_config.json`, and `model.safetensors` from the checkpoint,
plus `tokenizer.json` and `tokenizer_config.json` from the tokenizer fixture, into
one disposable model directory. The consumer reads that directory from a filesystem
path; neither fixture is supplied through the consumer classpath. Commit a seeded
smoke golden under `tools/release-smoke/goldens/` containing the chat prompt IDs,
generated IDs, decoded deltas, seed, sampling policy, and native pin/batch shape used
for exact comparison. The smoke test loads the assembled directory through the
scheduler's worker-owned factory, renders its chat template, streams the golden
completion, executes a two-request bounded batch, and closes the scheduler. Assert
the packaged native pin and extracted-runtime path, resolved module versions and POM
graph, and that no checkout artifact appears in
the dependency report. Verify the license/NOTICE/Javadoc artifacts and the native
jar's required dylibs/metallib.

Add an opt-in `smoke` Maven repository to each published module's Gradle publication,
located under a disposable build directory, with `publishMavenPublicationToSmokeRepository`
tasks. These tasks publish jars, POMs and Gradle metadata without invoking `release`,
Central credentials, or `publishToMavenLocal`. In PR/macOS CI, publish all six current
SNAPSHOT modules there, then run the consumer in **CI mode** against that repository;
CI mode permits SNAPSHOT versions but asserts all six module coordinates came from the
disposable repository, not a checkout project or `mavenLocal()`. Use repository content
filters so jmlx modules resolve only from the selected smoke repo, while external
libraries may resolve from Central. This catches package-shape problems with the
current build. For a release candidate, set the intended non-SNAPSHOT versions and
run the same isolated smoke in **release mode**, which rejects every SNAPSHOT in the
resolved graph. Finally run release mode with Maven Central as the sole jmlx source
after all six modules are visible. Give each consumer run a fresh `GRADLE_USER_HOME`
and no substitution rules or `mavenLocal()` repository. The Central-resolved smoke
is a pre-release-completion gate, not a PR check that requires publication.

Publish manually on real macOS ARM64 using the existing per-module `release.sh` and
root `verifyNoSnapshotDependencies`, credential, signing, and release-script guards.
Do not automate credentialed publishing in CI. Set each module to its intended first
release version and leave already-published prerequisites at that non-SNAPSHOT version
until their dependents have released and the Central consumer smoke passes:

| Order | Module | First Central version | Published dependencies |
| --- | --- | --- | --- |
| 1 | `jmlx-jinja` | 0.6.0 | none |
| 2 | `jmlx-tokenizer` | 0.1.0 | `jmlx-jinja` |
| 3 | `jmlx-native-macos-arm64` | 0.1.0 | none; runtime opt-in |
| 4 | `jmlx-ffi` | 0.5.0 | none |
| 5 | `jmlx-core` | 0.5.0 | `jmlx-ffi` |
| 6 | `jmlx-models` | 0.1.0 | `jmlx-core`, `jmlx-tokenizer` |

For each module, run its ordinary and release verification checks, inspect generated
POM/module metadata, signatures, licenses and native resources where applicable,
publish, and confirm Central availability before publishing a dependent. Record the
coordinates and verification result in the inference report. Only after the final
Central-resolved clean-consumer smoke passes should each module move to its next
`-SNAPSHOT` version. If Central publication cannot occur during implementation, leave
this gate open and report the exact unpublished module; the first-release milestone is
not complete on local-repository smoke alone.

## Exit gate

1. The documented consumer example renders chat, streams seeded output, runs a real
   multi-row bounded batch, cancels one request while the other keeps its RNG sequence
   and step count, and closes the worker-owned model and native scope by closing the
   scheduler.
2. Required Java and macOS native CI checks pass, including non-skipped scheduler
   suites, repeated-batch memory bounds, and package smoke. Benchmarks and support
   claims have reproducible evidence for each claimed family.
3. All six release coordinates are available from Central, the exact published graph
   resolves in a clean macOS ARM64 consumer, and the release report records supported
   and unsupported capabilities with their actual verification level.
