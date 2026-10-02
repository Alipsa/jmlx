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
- **Benchmark evidence does not exist yet.** `req/phase6-4-benchmark.md` records no native
  throughput or memory results (it was written on Linux; performance comparison is left as
  an explicit native-run acceptance item). The only measured native data from 6.4 is the
  quantization probe (CI runs 125/126). Task, before release: on macOS Apple Silicon run the
  per-family `DecodeBenchmark` decode loop (bounded-memory and long-context cases included)
  for the claimed families, and record raw JSON, commands, hashes, pins and device in
  `req/phase6-4-benchmark.md` or the linked Phase 6.5 report. Any family or claim without a
  recorded run is explicitly narrowed out of the release claim; a checked box in an older
  plan is not a substitute for a measured result.

## 1. Settle the public scheduler contract

Add `BatchGenerationScheduler` and a small immutable `BatchSchedulerConfig` in
`jmlx-models/src/main/java/se/alipsa/jmlx/models/`. Expose one asynchronous
`submit(GenerationRequest, Consumer<GenerationEvent>)` returning a request handle with
a read-only completion stage and a thread-safe `cancel()` signal. The handle owns no
native object. Document these behaviors in Javadoc before implementing the worker:

- Make a worker-owned factory the only constructor path: `BatchGenerationScheduler.start(config,
  modelFactory)`, where the checked factory receives the root `MLXScope` and returns
  `TextGenerationModel`. The worker creates that root, invokes the factory there, verifies the
  result is a `DecoderModel` whose model scope is the root or its descendant, then publishes a ready
  scheduler. Add a package-private, worker-thread-only scope accessor on `DecoderModel` because its
  inherited `Module.scope()` is protected in `se.alipsa.jmlx.nn`; check
  `root.isAncestorOf(model.modelScope())` before accepting the factory result. A factory that
  returns a model from an unrelated scope fails startup with a named error. Startup failures close
  the factory result where possible and the root, and `start()` then joins the worker thread (see
  the exit sequence) before propagating the error to the caller. `TextGenerationModels.load(scope,
  path)` already returns the interface, so this accepts the existing loader without changing its
  public return type. Do not accept a caller-loaded model or scope: `MLXScope.newChild()` and every
  weight access check the owning thread. The worker alone closes the model and root when the
  scheduler closes. Submission and cancellation never touch MLX arrays or caches.
- Complete request stages on a dedicated, bounded completion dispatcher so the MLX worker never runs
  a request's completion itself. This guarantees where completion is dispatched, not where every
  observer runs (see the registering-thread limit below). Reserve one completion slot for every
  accepted request at submission; retain it through queued and active states and release it only
  after the dispatcher has run that request's completion. The dispatcher queue plus running
  completions is therefore bounded by the total admission permits (`maxQueuedRequests +
  maxBatchSize`), and worker handoff never waits for queue space (the exit sequence adds no queued
  task). Exhausted permits reject new submissions immediately. Expose the handle's stage as
  `internalFuture.minimalCompletionStage()`: it is read-only and its `toCompletableFuture()` returns
  a new future, so no hand-written facade is needed. Dependents attached to that stage see a failed
  request as a `CompletionException` whose **cause** is the real exception (checked in jshell:
  `minimalCompletionStage().handle`/`exceptionally` receive the wrapper, while handlers on the
  internal future get the raw exception). Every "completes exceptionally with X" statement in this
  plan therefore means X is the cause of what a stage dependent receives. The Javadoc tells callers
  to unwrap `CompletionException` (`join()` throws it; `get()` throws `ExecutionException`), and one
  test pins this behavior. The handle's `cancel()` atomically ORs its signal with
  `GenerationRequest.cancellationToken()`. Stage cancellation is not a request cancellation API; a
  caller must use the handle. Document that completion observers must return promptly and must not
  block on another request's stage; doing so can deadlock a bounded dispatcher. Document the
  registering-thread limit: a non-async dependent (`thenAccept`, `thenApply`, `thenRun`, ...)
  attached to an **already completed** stage runs immediately on the attaching thread, and a
  read-only `minimalCompletionStage()` does not change that. A token listener or event callback
  (which runs on the worker) that calls `otherHandle.stage().thenAccept(...)` on a finished request
  therefore runs that observer on the MLX worker, so such an observer must not block or use MLX.
  Callbacks must use the `*Async` forms or avoid attaching dependents. A `close()` from the worker
  or completion dispatcher cannot join itself, and throwing there would be swallowed by a dependent
  stage (`handle.stage().thenRun(scheduler::close)` would silently never close). Such a call instead
  stops admission and returns immediately without waiting; the worker's exit sequence (see the
  shutdown rule below) completes the shutdown, so nothing outside needs to call `close()` again.
  Document that it does not wait, and test the pattern, including that the JVM can then exit.
- Configure positive `maxBatchSize`, `maxQueuedRequests`, and a bounded prompt-token budget for a
  batch. Count prefill work as `rows × maximum padded prompt width`, including left padding, rather
  than summing valid prompt lengths. Reject a prompt exceeding the budget even alone, and a
  `maxNewTokens` above `maxNewTokensPerRequest`, with `IllegalArgumentException` at `submit` (they
  can never succeed on retry); `BatchAdmissionRejectedException` is reserved for transient capacity
  ("retry later"). Exhausted admission capacity causes immediate, named rejection with no request
  side effects; it never grows implicitly. Name that exception `BatchAdmissionRejectedException`,
  not "queue full": cancelled requests and requests awaiting completion dispatch free their queue
  slot but keep their admission permit, so a slow dispatcher can reject a submission while the
  waiting queue is empty. Document this. An active batch holds at most `maxBatchSize` requests;
  waiting requests count against `maxQueuedRequests` and every accepted request also holds one total
  admission permit through completion dispatch. The oldest waiting request determines the next
  cohort's compatibility group; scan later requests only to fill that cohort without exceeding
  either bound. Add a scheduler `maxNewTokensPerRequest` cap and reject larger requests before
  enqueue. With no mid-decode joins, the next cohort waits at most that many active decode steps
  plus callback time; document this step-count bound and verify that the oldest waiting request runs
  next. There is no wall-clock bound while callbacks may block.
- A callback runs synchronously on the worker, receives only immutable Java events,
  and must return promptly. It must not call a blocking scheduler operation. `submit` remains
  nonblocking even when invoked by a callback; `close()` from that worker only stops admission
  and returns without waiting (see the dispatcher rule above). No unbounded output event queue
  is added.
  A slow callback delays the worker; document that operational limit and measure it
  in the benchmark. Never wait for request completion on the worker.
- Cancellation before admission completes with `CANCELLED` and one terminal event; cancellation in
  flight is observed before prefill or between decode steps, never in the middle of a native call.
  At **every** active step boundary, the worker also re-checks all waiting requests' own
  cancellation tokens and handle signals. It removes cancelled requests from the queue immediately,
  emits their one `CANCELLED` terminal event on the worker, and hands completion to the dispatcher;
  their queue slots are then free, while admission permits remain held until completion dispatch
  finishes. Removing one request never cancels or changes another. Document the cost: this is up to
  `maxQueuedRequests` user `isCancelled()` calls per decode step on the MLX worker, and a token that
  blocks stalls the worker like a slow callback. A token whose `isCancelled()` throws fails only its
  own request: it is removed like a cancelled request, with no terminal event, and its stage
  completes exceptionally with `GenerationAbortedException` carrying the cause and its partial
  generated IDs (empty if it never started), matching the token-listener failure rule below. It
  never affects another row or the worker. After a token-listener exception, close that request's
  state and complete its future exceptionally with the existing `GenerationAbortedException` partial
  result; emit no terminal event for it. A terminal-listener exception follows the direct API's
  logged, successful-result rule. `close()` atomically stops admission; a racing or later `submit`
  gets a named `SchedulerClosedException`, distinct from the named `BatchAdmissionRejectedException`
  for capacity rejection. The worker treats every remaining queued and active request as cancelled
  at its next safe boundary: each gets a `GenerationResult` with `FinishReason.CANCELLED`, its
  partial generated IDs, and exactly one terminal event delivered on the worker, including tokenizer
  flush if applicable. Then the worker closes all row state, cohort scopes, model, and root.
  `close()` joins the worker and waits for the completion dispatcher to deliver every accepted
  request's stage before returning; it never abandons an accepted stage. Document that a callback
  that never returns can stall worker shutdown and an observer that never returns can stall
  dispatcher drain and `close()`; there is no forced timeout. **Exit sequence.** The dispatcher is a
  `ThreadPoolExecutor` subclass whose bounded queue holds exactly the admission-permit count; the
  exit sequence submits **no** task, so it needs no extra slot and cannot be rejected or block the
  worker. Every worker exit path (orderly close, `FAILED`, and every startup failure after the
  dispatcher exists) runs the same three steps **in this order**: (1) close the model and root (best
  effort), (2) release the one-scheduler guard, (3) call the executor's `shutdown()` last, which
  still runs every already-queued completion. The order matters: `terminated()` may run on the
  worker (only if the dispatcher never started a thread; checked in jshell) or on a dispatcher
  thread after `shutdown()` has returned, so the plan relies on neither. Because the guard is
  released before `shutdown()` is even called, `CLOSED` can never be visible while the guard is
  still held, wherever `terminated()` runs, and a caller who sees `CLOSED` and calls `start()` is
  not rejected. Releasing the guard before the thread literally ends is safe because the worker
  makes no MLX calls after step (1). `CLOSED` is set (unless the state is `FAILED`) from the
  executor's `terminated()` hook, which runs only after every queued and running completion has
  finished, so it holds for a dispatcher of any size and never precedes the drain. Shutting the
  dispatcher down is the worker's job, so a `close()` made from the worker or dispatcher leaves no
  live non-daemon thread and no state stuck at `CLOSING`. If `Thread.start()` for the worker itself
  throws (for example `OutOfMemoryError`: unable to create native thread), no worker `finally` will
  ever run, so `start()` releases the guard and shuts the dispatcher down itself before rethrowing;
  test this with an injected thread factory. An outside `close()` joins the worker, then calls
  `awaitTermination` on the dispatcher (no timeout, like the drain above) and returns.
  **Interruption:** these waits are uninterruptible: `close()` keeps waiting if interrupted,
  restores the thread's interrupt flag before returning, and never throws `InterruptedException`, so
  a try-with-resources close stays quiet; test it. **Startup wait:** `start()`'s wait for the worker
  to report ready is uninterruptible too: if the calling thread is interrupted while a slow factory
  is loading, `start()` keeps waiting, restores the interrupt flag before returning (or before
  rethrowing a startup failure), and never throws `InterruptedException`. Otherwise it could return
  while the worker still loads, leaving a live non-daemon worker that holds the guard and the model
  and that no caller references; test this by interrupting the thread that calls `start()` during a
  slow factory. **Startup failure:** `start()` joins the worker thread (same uninterruptible wait)
  before rethrowing, so the guard is already released and an immediate retry is accepted.
  **Combining calls:** every `close()` from an outside thread waits for the drain, including one
  made after a non-waiting `close()` from the worker or dispatcher; repeated `close()` calls are
  safe; the non-waiting rule above is the only exception to "`close()` joins". The worker and the
  dispatcher are named **non-daemon** threads: a daemon worker could be killed mid-native call at
  JVM exit, which is worse than a forgotten `close()` keeping the JVM alive. Document that `close()`
  is required (use try-with-resources).
- **Failed-worker state.** A failure that leaves the model or worker unusable moves the scheduler to
  `FAILED`: a thrown `Error`, an exception outside the §2 step-failure classification, a failed
  model, or a cleanup step (closing a cache, scope, or the model) that itself throws. Cohort-wide
  failures with a named cause (forward, selection-graph, joint-evaluation, compaction) poison that
  cohort's caches and fail its rows, but keep the worker `RUNNING` because the model is still valid.
  Each such row's stage completes with `GenerationAbortedException` carrying that cause and the
  row's partial IDs, and no listener gets a terminal event, as in the other failure rules. On
  entering `FAILED` the worker atomically stops admission, completes **every** queued and active
  stage exceptionally with the existing `GenerationAbortedException` (partial generated IDs, empty
  for queued requests) whose cause is a named `SchedulerFailedException`, so consumers handle one
  partial-result type. No listener receives a terminal event on failure, matching the token-listener
  failure rule. The worker then makes a best-effort close of the model and root and exits, so the
  "never abandons an accepted stage" promise above still holds. Later `submit` calls throw
  `SchedulerFailedException` itself, distinct from `SchedulerClosedException` (orderly close) and
  `BatchAdmissionRejectedException`. `close()` after a failure still returns normally: it joins the
  already-exited worker, drains the dispatcher, and does not rethrow the failure (try-with-resources
  stays quiet, and the cause has already reached every stage and the next `submit`). A caller with
  no requests in flight would otherwise only see the failure by calling `submit`, so add a
  non-blocking `state()` accessor (`RUNNING`, `CLOSING`, `FAILED`, `CLOSED`) and
  `Optional<Throwable> failure()`. Transitions: `RUNNING → CLOSING` when `close()` stops admission,
  and `CLOSING → CLOSED` only after the worker has finished its cleanup and released the guard and
  the dispatcher has drained (set by the executor's `terminated()` hook, see the exit sequence);
  `RUNNING → FAILED` when the worker fails. `FAILED` is terminal and sticky: a later `close()` still
  drains and returns but leaves the state `FAILED`, so `failure()` stays meaningful. A failure
  during `CLOSING` (for example a throwing cleanup) also ends in `FAILED`. An `Error` is recorded as
  the cause, not swallowed, and may be rethrown from the worker thread after stages complete.
- Match direct `GenerationResult` and `GenerationEvent` semantics: EOS token included,
  explicit stop token excluded, one terminal event for normal completion, tokenizer
  flush delta, optional post-filter log probability, and `maxNewTokens == 0` without
  prefill. Do not change `GenerationConfig`'s record shape or direct `generate` signature.

## 2. Build the batched execution path

Implement a worker state machine in `jmlx-models` with one request context per admitted
request: prompt IDs, cache policy, token-frequency map, generation count, independent
`SamplingPipeline`/RNG key, incremental text decoder, event listener, result builder,
and cancellation signal. Validate vocabulary IDs, tokenizer size, cache policy, and
bounded-FULL worst-case capacity in `submit` on the **caller thread**, before taking an
admission permit and without calling into `DecoderModel`: `start()` snapshots the
immutable model facts needed (vocabulary size, sliding-window size, RoPE kind, capacity
limits) on the worker once the model is accepted, and publishes them with the ready
scheduler. A bad request is rejected by `submit` itself with an `IllegalArgumentException`:
it takes no admission permit and returns no handle, so it cannot poison the queue. All
request validation happens at submit; none is deferred to the step boundary.

**Tokenizer thread safety.** Under the scheduler one `HfTokenizer` (carried in each
`GenerationRequest`) is used concurrently: submitting threads encode and size-check with it
while the worker creates and drives incremental decoders, where the direct API only ever
used it from one thread. Thread safety of `HfTokenizer`, `AddedTokenMatcher` (which has a
`HashMap` field), `ChatTemplateRenderer` and the Jinja `Template` after construction is
unverified. Before the scheduler ships, require either a stated Javadoc guarantee of
concurrent use after construction or a concurrency test (several threads encode, render, and
decode against single-threaded reference output), and fix any shared mutable state found.

**Gate 0 — native thread probes (before any scheduler code).** (a) Exercise a worker thread after
`NativeOps.DEFAULT_STREAM` was initialized on a different thread, then load and generate entirely on
that worker. (b) Run two threads that each load and generate concurrently on the shared default
stream, which is what a user calling the public direct `generate` while a scheduler runs, or
starting two schedulers, would do. These are pinned-runtime probes, not an assumption about Metal
command-encoder or MLX stream thread safety (MLX 0.31.2's behavior here is unverified). Give each
probe its own forked JVM, the same pattern as `loaderGuardTest`: (a) needs a fresh JVM in which
nothing else has touched `DEFAULT_STREAM`, which is initialized once per JVM, and (b) races two MLX
threads on a stream this plan treats as unsafe, so it can crash its JVM and must not share one with
(a), with `:jmlx-models:test`, or with `check`. Register `threadProbeTest` (`include
'**/WorkerStreamProbeTest.class'`) for (a) only, excluded from `test` and wired into `check`: (a) is
a required test. Register `concurrentStreamProbe` (`include '**/ConcurrentStreamProbeTest.class'`)
for (b), excluded from `test` and **not** wired into `check`; run it as its own CI step with
`continue-on-error: true`, give the step an `id`, and record `steps.<id>.outcome` in the job summary
so a crash is captured even when no XML exists. Upload the result XML with `if-no-files-found:
warn`, because a crashed JVM may write none. Do not rely on `ignoreFailures` for (b): it covers
failing tests, and whether it covers a crashed test process is unverified. (b) is an evidence probe
and never gates, since a pass proves nothing. If (a) fails, the native stream ownership or
initialization design changes and §1–§2's worker design must be revisited. A passing (b) is **not**
permission: a race test can pass on CI and still crash later. "At most one MLX-using thread at a
time per process" is therefore the documented default and stays so unless upstream MLX documents
that sharing a stream across threads is safe. Because the constraint covers all of `jmlx-core`, not
just the scheduler, it is a public-contract change to `jmlx-core` and goes in its package Javadoc
and README. State one **known exception**: the `Cleaner` backstops in `MLXScope` (`mlx_array_free`)
and `MLXGrad.Fn` (`mlx_closure_free`/`mlx_closure_value_and_grad_free`, plus its upcall arena) run
on the JVM Cleaner thread when a scope or function is never closed. The array free is the unresolved
cross-thread-free question in `req/initial-plan.md` (Open questions); the closure frees are a
related but separate case that question does not cover. The Javadoc and README name both paths and
link the question rather than claim the rule holds absolutely; callers avoid them by always closing
scopes and functions, which the scheduler's worker does. If probe (b) shows any instability, add
probe (c), freeing arrays (and closures) from a second thread while the worker generates, which also
resolves that open question. Enforce the rule where possible by rejecting a second `start()` while
another scheduler's worker has not exited. The guard is static, so it enforces the rule only within
one jmlx classloader; the rule itself stays per process, because MLX's native state (the default
stream and the Metal device) is process-wide. Say both in the Javadoc. (Whether the JDK already
refuses to load the native library into a second classloader on the `System.load` path is
unverified; the documented rule says "per process" either way.) The guard is released by the
worker's exit sequence, in a `finally`, whatever happened: after an orderly close, after a runtime
`FAILED` (including one whose best-effort close of the model or root threw), and on every
startup-failure path in which the worker exits (a throwing factory, a failed scope check). It is
never released while the worker can still make MLX calls; it is released after the model and root
are closed and before the dispatcher is shut down, so `start()` after a failed `start()` or after
`state() == CLOSED` is not rejected. If the worker thread cannot be started, `start()` itself
releases it (see the exit sequence). `start()` has no startup timeout: a factory that never returns
makes `start()` block, even if its caller is interrupted, and that is documented rather than
reported as a failure, because a timeout would throw while a live worker still holds the guard and
the model. The rule bounds concurrent MLX threads and an exited worker is not one, so a failed
native cleanup (leaked handles; state `FAILED`, recorded by `failure()`) does not block a restart,
and one failed `start()` cannot block later ones until the JVM exits; test the failure paths,
including a throwing root close. Close-then-restart (common in tests and the example) works. The
public direct `generate` API cannot be blocked, so for that path the rule is documentation only. Run
the probes before writing the scheduler, and keep (a) as a required native CI test and (b) as an
evidence probe afterwards.

**Scope layout:** On the worker, create one model root; create each active cohort's scope
`C` as a child of the validated `DecoderModel.modelScope()`, so weight scopes remain
ancestors of every forward activation even when the factory used a child of the root for the
model. Cache sets live in `C`. Each request's sampler key also lives in `C` and is released
with `MLXArray.close()` when the row leaves the cohort. The `rankIndices` array (and the
`arange` it is built from) is identical for every row because every row shares the model's
vocabulary size V, so the cohort builds **one** in `C`, shares it across rows, and frees it
at cohort end. `SamplingPipeline.close()` today only closes `currentKey`, so per-row copies
would stay in `C` until the cohort ends; sharing avoids that leak (the alternative is
extending `close()` to free them). Sharing needs a refactor: the constructor currently
builds `rankIndices` itself, and only for top-k or `topP == 0` policies, so add a
constructor that accepts the shared array (null when the policy needs none) and keep the
existing constructors for the direct path and its tests. Every operand sits on one `C → S`
chain, so no request-scope tree or cross-scope copy is needed. For **every forward step**,
create `S = C.newChild()`, allocate the batched token IDs in `S`, and run the forward there.
Activations, masks, RoPE frequencies and logits therefore live in `S`, not `C`;
`KVCache.append` hoists retained K/V into `C`. Per-row logits slices and selection graphs
are built in `S` (an op's result goes to the innermost scope among its operands, and `S`
descends from `C`); `advanceKey` hoists each row's successor key from `S` to `C` and closes
the old key. After joint evaluation **and readBack**, close `S`; close a row's key only when
the row leaves. Reorder into `C`. Add a native test for this full scope chain, including
filtered sampling, key hoist, forward and cache retention, before relying on it in the
scheduler. No change to `MLXShape.takeAxis` or its Javadoc is required.

1. Extract a package-private decoder step API shared with `DecoderModel.generate`. It must
   be **lazy**: it does not call the `StepBoundaryEvaluator`, and prefill slices the last
   column **before** `embedding.project` / `lmHead.forward`, mirroring the private
   `prefill`/`decode`. With left padding the last valid column of every row is always column
   `T-1` (`preflightBatch` requires the padded width to equal the longest valid row), so a
   single slice suffices; do not write a per-row gather. Do not build it on the public
   `forward(ids, caches, validLengths)`: that method already evaluates logits plus every
   cache array before returning and projects all `[B,T,V]` positions, which would sync twice
   per step and compute T× more logits in prefill. Batched prefill accepts left-padded
   `[B,T]` IDs and `validLengths`, returns the last valid logits for each row; decode
   accepts `[B,1]`. Use one cache set per compatible active group with cache row order
   matching request-context order. Keep the model's cache-set preflight and poisoning rules.
   Before scheduler integration, add one native model test per supported family comparing
   left-padded batched prefill and decode logits to independent single-row logits within a
   stated numerical tolerance. Include per-row masks and RoPE positions, Gemma embedding
   scaling, Phi-3 fused projections, and Mixtral expert routing at `B > 1`. Require these
   suites in CI. Avoid reimplementing attention or sampling in the scheduler.
2. Form a prompt batch within `maxBatchSize` and the prompt-token budget. Group by
   resolved cache policy and any architecture/runtime restriction that cannot safely
   share a cache (including Dynamic-NTK unequal-position limits). Left-pad variable
   prompt lengths only when the native batch path supports them; otherwise split into
   equal-length groups and record this as a batching limit. Never silently replace a
   requested policy. Do not introduce mid-decode joins unless cache assembly and
   position semantics are proved; finishing the current cohort first is the initial
   single-worker policy.
3. Split `SamplingPipeline.select` into lazy `build`, one joint `evaluate`, and `readBack`
   phases, retaining its direct-path wrapper. At each step run one model forward, build each
   live row's selection with its own policy, penalty history and RNG key, then evaluate
   **all** selected IDs, requested log probabilities, finite checks, and changed cohort
   cache arrays in one call through the model's existing `StepBoundaryEvaluator` (native
   default delegates to `MLX.eval`). Expose that package-private hook to the scheduler so
   `DecoderForwardPoisoningTest` can inject a joint-evaluation failure; do not call
   `MLX.eval` directly here. Read back only after the joint call succeeds. Prefer batched
   readback over ~3–4 native calls per row (each `toIntArray()` is a contiguous copy plus
   eval): `build` stacks the cohort's per-row results into two lazy arrays **before** the
   joint evaluate and passes them into it, so reading them back does not trigger a second
   sync. One is an int32 `[B,3]` array of selected ID, finite flag, and tempered-finite flag
   (a constant 1 for greedy rows, which have no tempered check); the other is a float32
   `[B]` array of post-filter log probabilities, built only when some row requests them.
   Benchmark this against per-row readback. With the lazy step API this gives one joint
   synchronization for `B` rows. A subsequent compaction is **not** free: today
   `KVCache.reorder` calls `MLX.eval` per layer, so one compaction step costs `L` extra
   syncs. Record this in the benchmark, and add a deferred-eval reorder variant only if the
   measurement justifies it. Attribute a joint evaluation failure to the shared cohort.
   Greedy selection consumes no RNG. Test that cancellation, reorder, and a different
   companion request do not advance another row's key or step count.
4. Classify failures at the step boundary. A non-finite row's logits discovered after a
   successful joint evaluation, tokenizer output-decoder failure (including final flush),
   and token-listener failure affect only that row; use the direct path's
   `GenerationAbortedException` stage/partial-result conventions for decoder and listener
   failures. A non-finite row (the direct path throws a plain `IllegalStateException` with
   no partial result, and poisons its caches) completes its own stage exceptionally with
   `GenerationAbortedException` carrying the finite-validation cause and the IDs generated
   so far, emits no terminal event, and leaves the cohort via the step's single compaction.
   Row isolation is plausible for dense attention but unverified for Mixtral's gather-based
   expert routing, so it is tested rather than assumed (see the native tests); if a family
   fails that test, its non-finite rows fail the whole cohort instead. Forward construction,
   selection-graph/native errors, or joint evaluation failure poison the whole cohort cache
   set and fail every active row as in §1 (`GenerationAbortedException` with the named cause,
   partial IDs, no terminal event). Do not infer a row-only native failure
   from the order in which arrays were passed to `MLX.eval`. Queued requests may start with
   fresh caches only while the worker is `RUNNING`; otherwise the failed-worker rule in §1
   applies.
5. After selection, emit events in stable row order. Collect all finished, cancelled,
   and row-failed indices from one step, then perform **one** `KVCache.reorder` per
   layer for the surviving rows; never compact once per departing row. Compact layer by
   layer: `new = reorder(old); install(new); old.reset()`. `reorder` already evaluates
   its copy, and nothing reads the old set after a failure because every surviving row
   fails anyway, so there is no reason to hold a full second cache set; peak extra
   memory is one layer's K/V rather than a whole cache. If layer `k` reorder fails,
   reset the old layers `k..L-1` and the installed new layers `0..k-1`, report a named
   cohort-wide compaction failure, and fail all surviving rows. Do not continue decode
   with a partly reordered cache set. If no rows survive, reset/close the old set
   without reorder. Check that retained cache data are independent of closed step
   scopes. Put the per-layer reorder call behind a package-private scheduler seam so an
   injected failure at an interior layer can assert cleanup and cohort failure without
   relying on a real Metal fault. Benchmark peak active native bytes during compaction,
   expecting about one layer of overhead, and record it.

**Native tests:** mixed prompt lengths, uneven output lengths, EOS/stop/max-token paths,
FULL and sliding policies, capacity rejection, Dynamic-NTK grouping, cancellation
before/after prefill, queued cancellation polled during an active cohort, callback failure,
distinct admission-rejected/closed rejection races, close with queued/active requests and
drained completion stages, a failed-worker scenario (every accepted stage completes
exceptionally, later `submit` throws `SchedulerFailedException`, `close()` returns),
one-step multi-row compaction, a non-finite-row isolation test for **every** family, and
repeated batches with a stable active-memory plateau. The isolation test injects NaN into
one row's embedded activations through a package-private hook on the step API, applied right
after the embedding lookup so it flows through attention, Mixtral's expert routing, and the
output projection; the other rows' tokens must match a same-batch-shape reference run
exactly, with logits compared within tolerance. Do not inject by editing an embedding weight
row: tied-output families (the Gemma test checkpoint is one: it has no `lm_head.weight`)
project through the same table, so a NaN row would contaminate every row's logits at that
vocabulary column and prove nothing about isolation. Injecting NaN only into logits is not
enough, since it exercises only code after the forward pass. The plateau assertion must
sample active bytes **within one long cohort** across many decode steps, as well as after
repeated cohorts, so a per-step activation leak cannot hide behind end-of-cohort cleanup.
Gate 0 probe (a) stays a required native test; it runs alone in `threadProbeTest`, so extend the
required-suite check in `.github/workflows/ci.yml`, which only reads `build/test-results/test/`, to
read that task's results directory too. Probe (b) runs in its own non-gating CI step and is not
asserted. Assert required native test XML in `.github/workflows/ci.yml` so an unbootstrapped run
cannot appear green through skipped tests.
After cancelling a row, assert the other row's RNG key sequence and decode-step count are unchanged;
compare its events exactly with a reference run using the **same cancellation schedule and batch
shapes**. Compare its logits with an uncancelled run only within a documented numerical tolerance,
since compaction changes batch shape. Retain the roadmap's no-cross-batch-shape token identity rule.
Include a measurable proof that a multi-row cohort uses fewer model forward calls than one forward
per request.

## 3. Consumer example, benchmark, and support report

- Add an opt-in `jmlx-examples` program using a local checkpoint and paired tokenizer:
  construct the scheduler with `start(config, scope -> TextGenerationModels.load(scope,
  checkpoint))`, render a chat request, stream sampled text, submit two or more requests,
  cancel one, await all results, and close the scheduler. The worker closes its model and
  scope. Take paths and limits as arguments or environment configuration; no checkpoint
  download during required PR checks.
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
`se.alipsa:jmlx-models` coordinate and runtime-only `se.alipsa:jmlx-native-macos-arm64`;
transitive resolution must bring in `jmlx-core`, `jmlx-ffi`, `jmlx-tokenizer`, and
`jmlx-jinja`. Compile and run on a fresh Gradle home on macOS ARM64 with Java 25 and
`--enable-native-access=ALL-UNNAMED`, with no `jmlx.library.path` or `JMLX_LIBRARY_PATH`
override. A fresh `GRADLE_USER_HOME` does **not** reset the native extraction cache:
`ClasspathNativeExtractor` defaults to `~/Library/Application Support/se.alipsa.jmlx/native`
and reuses an existing extraction after only a file-size check, so on a machine that already
ran jmlx at the same pin the smoke would never extract from the published jar. Run the
consumer with `-Djmlx.native.cache.path=<disposable empty dir>` and assert that the pinned
files were freshly extracted there (directory empty before, expected files present after).
Use the committed synthetic Mistral checkpoint at `tools/hf-reference/goldens/checkpoints/mistral/`,
but **not** its chat tokenizer at `jmlx-tokenizer/src/test/resources/families/mistral/`: the two are
not a pair. The checkpoint has `vocab_size` 128 and `bos_token_id`/`eos_token_id` 1/2, while that
tokenizer defines only ids 0-24 with BOS/EOS 21/22. The generation size check accepts the smaller
tokenizer, and `TokenizerRuntime.decodableToken` returns null (decoded as `""`) for ids above the
largest known id, so sampling over 128 logits would silently drop ids 25-127 and stream empty
deltas; an exact golden would then pass while text is missing. Nothing can restrict selection by id
(`GenerationConfig` has no logit mask and its shape is frozen; `topK`/low temperature limit by rank
only). Commit a dedicated smoke tokenizer under `tools/release-smoke/fixtures/` that defines every
id 0-127, with special-token ids matching the checkpoint (BOS 1, EOS 2) and a chat template the
renderer accepts, so every id the model can emit decodes to something. Give it a provenance note
stating how it was built (hand-written, not Hugging Face-generated). The smoke asserts that the
tokenizer's id range equals the checkpoint's `vocab_size`, that the golden's `eosTokenIds` match the
checkpoint's, and, over the golden's generated ids, that every non-special id yields non-empty text.
Ids that are special or byte-fallback pieces (partial UTF-8 buffers) may stream empty deltas, so
choose the prompt and seed **by experiment** so the committed golden contains at least one non-empty
decoded delta, and assert that. A smoke setup task copies `config.json`, `generation_config.json`,
and `model.safetensors` from the checkpoint, plus `tokenizer.json` and `tokenizer_config.json` from
the dedicated smoke tokenizer fixture, into one disposable model directory.
The consumer reads that directory from a filesystem path; neither fixture is supplied
through the consumer classpath. Commit a seeded smoke golden under
`tools/release-smoke/goldens/` containing the chat prompt IDs, generated IDs, decoded
deltas, seed, sampling policy, and native pin/batch shape used for exact comparison. The
smoke test loads the assembled directory through the scheduler's worker-owned factory,
renders its chat template, streams the golden completion, executes a two-request bounded
batch, and closes the scheduler. Assert the packaged native pin and extracted-runtime path,
resolved module versions, and that no checkout artifact appears in the dependency report.
The smoke repository publishes Gradle module metadata, and Gradle prefers the `.module` file
over the POM when it exists, so a normal resolution never exercises the POMs that Maven
consumers depend on. Add a second resolution of the same graph with `metadataSources {
mavenPom(); ignoreGradleMetadataRedirection() }` (or a tiny Maven consumer) and assert it
yields the same six coordinates and versions on the **runtime classpath**. `jmlx-core`
depends on `jmlx-ffi` as `implementation`, so the POM puts `jmlx-ffi` in runtime scope and
it is absent from the compile classpath; comparing compile classpaths would report a false
mismatch. Verify the license/NOTICE/Javadoc artifacts and the native jar's required
dylibs/metallib.

Add an opt-in `smoke` Maven repository to each published module's Gradle publication,
located under a disposable build directory, with `publishMavenPublicationToSmokeRepository`
tasks. These tasks publish jars, POMs and Gradle metadata without invoking `release`,
Central credentials, or `publishToMavenLocal`. In PR/macOS CI, publish all six current
SNAPSHOT modules there, then run the consumer in **CI mode** against that repository; CI
mode permits SNAPSHOT versions but asserts all six module coordinates came from the
disposable repository, not a checkout project or `mavenLocal()`. Use **exclusive**
repository content so `se.alipsa` modules resolve only from the selected source, while
external libraries (Jackson is the only external runtime dependency) may resolve from
Central. Make the jmlx source a parameter covering all three modes (smoke repo in CI mode
and release-candidate mode; Central in the final release-mode run):

```groovy
repositories {
  exclusiveContent {
    forRepository { maven { url = uri(jmlxRepoUrl) } }  // smoke dir, or the Central URL
    filter { includeGroup 'se.alipsa' }
  }
  mavenCentral { content { excludeGroup 'se.alipsa' } }  // external libraries only
}
```

Use the `url = uri(...)` assignment form, not the deprecated space-assignment syntax. A
plain `content { includeGroup ... }` on one repository only limits that repository and would
still let another serve `se.alipsa` artifacts. This catches package-shape problems with the
current build. For a release candidate, set the intended non-SNAPSHOT versions and run the
same isolated smoke in **release mode**, which rejects every SNAPSHOT in the resolved graph.
Finally run release mode with Maven Central as the sole jmlx source after all six modules
are visible. Give each consumer run a fresh `GRADLE_USER_HOME` and no substitution rules or
`mavenLocal()` repository. The Central-resolved smoke is a pre-release-completion gate, not
a PR check that requires publication.

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
