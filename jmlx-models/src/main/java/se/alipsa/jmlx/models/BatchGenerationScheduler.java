package se.alipsa.jmlx.models;

import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.IntPredicate;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.IncrementalTokenDecoder;
import se.alipsa.jmlx.tokenizer.TokenizerException;

/**
 * Runs many text-generation requests through one MLX-owning worker thread, decoding compatible
 * requests together as a cohort so one model forward serves several rows. Callers submit and cancel
 * from any thread; every native object lives on the worker.
 *
 * <p><b>Threading rule.</b> MLX allows at most one MLX-using thread at a time per process. This
 * scheduler's worker is that thread: the model is created by a worker-owned factory, so its weights
 * live in a scope the worker owns, and {@link #start} rejects a second scheduler while another's
 * worker has not exited. The guard is static, so it enforces the rule only within one jmlx
 * classloader; the rule itself is per process because MLX's native state is process-wide. The
 * public direct {@code generate} API cannot be blocked, so mixing it with a running scheduler is a
 * documentation-only rule. The known exceptions are the JVM {@code Cleaner} backstops in {@code
 * MLXScope} and {@code MLXGrad.Fn}, which free native handles from the Cleaner thread for a scope
 * or function that was never closed; always close scopes and functions.
 *
 * <p><b>Lifecycle.</b> Use try-with-resources: the worker and the completion dispatcher are named
 * non-daemon threads (a daemon worker could be killed mid-native call at JVM exit), so a forgotten
 * {@link #close()} keeps the JVM alive. {@code close()} stops admission, treats every queued and
 * active request as cancelled at its next safe boundary (each gets a {@link FinishReason#CANCELLED}
 * result with its partial IDs and one terminal event), closes the model and root scope on the
 * worker, and waits for every accepted stage to be delivered. Its waits are uninterruptible: if
 * interrupted it keeps waiting, restores the interrupt flag, and never throws. Called from the
 * worker or the completion dispatcher it cannot join itself, so it only stops admission and returns
 * without waiting; the worker's exit sequence completes the shutdown. Close-then-restart works.
 *
 * <p><b>Callbacks</b> run synchronously on the worker, receive immutable events, and must return
 * promptly; a slow callback delays every row. A callback must not call a blocking scheduler
 * operation. {@link #submit} stays nonblocking when called from one.
 *
 * <p><b>Failure.</b> A row-level failure (token listener, output decoder, cancellation token, a
 * non-finite row) completes only that row's stage with {@link GenerationAbortedException} and no
 * terminal event. A failure of cohort setup (an output decoder, the cohort scope, the caches, the
 * rank indices, or a sampler), of the shared forward, selection graph, joint evaluation, or of
 * cache compaction poisons the cohort and fails each of its rows the same way -- stage {@code
 * "cohort setup"}, {@code "batch step"}, or {@code "cache compaction"} -- and a failure that
 * escapes the decode loop after setup finished is attributed to stage {@code "cohort"}; all of them
 * leave the worker serving later cohorts. A failure that leaves the worker unusable moves the
 * scheduler to {@link State#FAILED}: every accepted stage completes with a {@link
 * GenerationAbortedException} whose cause is a {@link SchedulerFailedException}, and later {@link
 * #submit} calls throw a fresh {@link SchedulerFailedException} with the same cause.
 */
public final class BatchGenerationScheduler implements AutoCloseable {
  private static final System.Logger LOGGER =
      System.getLogger(BatchGenerationScheduler.class.getName());
  private static final AtomicBoolean ACTIVE = new AtomicBoolean();

  /** Lifecycle state, readable without blocking. */
  public enum State {
    /** Accepting and serving requests. */
    RUNNING,
    /** {@link #close()} stopped admission; the worker is draining. */
    CLOSING,
    /** The worker failed; terminal and sticky. See {@link #failure()}. */
    FAILED,
    /** The worker exited and every stage was delivered. */
    CLOSED
  }

  /** Creates the model on the worker thread, inside the worker's root scope. */
  @FunctionalInterface
  public interface ModelFactory {
    /**
     * Loads the model into {@code root} or a descendant of it. Must return a {@code DecoderModel}.
     */
    TextGenerationModel create(MLXScope root) throws Exception;
  }

  /**
   * Test seams; {@link #NONE} in production. {@code cleanupFault} runs inside the exit sequence's
   * best-effort close, so a test can make cleanup throw (the model and root classes are final).
   * {@code setupFault} runs at the start of a cohort's setup (cache, rank-index, and sampler
   * construction), so a test can fail cohort setup without touching the model. {@code decodeFault}
   * runs between steps of the decode loop, so a test can make a failure escape the loop (stage
   * {@code "cohort"}) without touching the model. {@code cohortGate} receives the number of waiting
   * requests and the worker forms no cohort until it returns true, the gate wait ({@code
   * cohortGateMaxWait}) elapses, or the scheduler stops, so a test can pin exactly which requests
   * batch; it runs on the worker under the admission lock, so it must be fast and side-effect free
   * (a throw is treated as "release now", see {@link #gateAdmits}).
   */
  record Hooks(
      DecoderModel.EmbeddingHook embeddingHook,
      CacheReorderer reorderer,
      Runnable cleanupFault,
      Runnable setupFault,
      Runnable decodeFault,
      IntPredicate cohortGate,
      Duration cohortGateMaxWait) {
    static final Hooks NONE = new Hooks(null, CacheReorderer.NATIVE, null, null, null, null, null);

    Hooks withEmbeddingHook(DecoderModel.EmbeddingHook hook) {
      return new Hooks(
          hook, reorderer, cleanupFault, setupFault, decodeFault, cohortGate, cohortGateMaxWait);
    }

    Hooks withReorderer(CacheReorderer replacement) {
      return new Hooks(
          embeddingHook,
          replacement,
          cleanupFault,
          setupFault,
          decodeFault,
          cohortGate,
          cohortGateMaxWait);
    }

    Hooks withCleanupFault(Runnable fault) {
      return new Hooks(
          embeddingHook, reorderer, fault, setupFault, decodeFault, cohortGate, cohortGateMaxWait);
    }

    Hooks withSetupFault(Runnable fault) {
      return new Hooks(
          embeddingHook,
          reorderer,
          cleanupFault,
          fault,
          decodeFault,
          cohortGate,
          cohortGateMaxWait);
    }

    Hooks withDecodeFault(Runnable fault) {
      return new Hooks(
          embeddingHook, reorderer, cleanupFault, setupFault, fault, cohortGate, cohortGateMaxWait);
    }

    Hooks withCohortGate(IntPredicate gate, Duration maxWait) {
      Objects.requireNonNull(gate, "gate");
      Objects.requireNonNull(maxWait, "maxWait");
      return new Hooks(
          embeddingHook, reorderer, cleanupFault, setupFault, decodeFault, gate, maxWait);
    }
  }

  private enum Poll {
    LIVE,
    CANCELLED,
    TOKEN_FAILED
  }

  private enum Outcome {
    CONTINUE,
    DONE
  }

  private record CohortKey(KVCachePolicy cachePolicy, int equalLength) {}

  /**
   * The cohort sizes the worker has started, bounded to the last {@link #CAPACITY} of them plus
   * lifetime totals: under light load every request is its own cohort, so an unbounded list would
   * grow for the process's life and a per-cohort copy would cost more with every cohort. Written on
   * the worker, read from any thread under the monitor.
   */
  static final class CohortSizeWindow {
    /** Most recent cohort sizes kept; older ones survive only in the lifetime counts. */
    static final int CAPACITY = 1024;

    private final int[] sizes = new int[CAPACITY];
    private int nextSlot;
    private long cohorts;
    private long rows;

    /** Records one started cohort; worker only. */
    void record(int size) {
      synchronized (this) {
        sizes[nextSlot] = size;
        nextSlot = (nextSlot + 1) % CAPACITY;
        cohorts++;
        rows += size;
      }
    }

    /** The most recent cohort sizes, in start order, at most the last {@link #CAPACITY}. */
    List<Integer> recent() {
      synchronized (this) {
        if (cohorts == 0) {
          return List.of();
        }
        int kept = (int) Math.min(cohorts, CAPACITY);
        int start = (nextSlot - kept + CAPACITY) % CAPACITY;
        List<Integer> out = new ArrayList<>(kept);
        for (int i = 0; i < kept; i++) {
          out.add(sizes[(start + i) % CAPACITY]);
        }
        return out;
      }
    }

    /** Total cohorts started, including any beyond the window. */
    long cohorts() {
      synchronized (this) {
        return cohorts;
      }
    }

    /** Total rows started across all cohorts (the sum of their sizes). */
    long rows() {
      synchronized (this) {
        return rows;
      }
    }
  }

  private static final class SchedulerThread extends Thread {
    private final BatchGenerationScheduler owner;

    SchedulerThread(BatchGenerationScheduler owner, Runnable body, String name) {
      super(body, name);
      this.owner = owner;
      setDaemon(false);
    }
  }

  private final class Dispatcher extends ThreadPoolExecutor {
    Dispatcher(int capacity) {
      super(
          1,
          1,
          0,
          TimeUnit.MILLISECONDS,
          // Linked, not Array: ArrayBlockingQueue eagerly allocates Object[capacity], so a
          // validated-but-huge admissionPermits() would throw OutOfMemoryError from start().
          // LinkedBlockingQueue is bounded the same way without the pre-allocation.
          new LinkedBlockingQueue<>(capacity),
          r -> new SchedulerThread(BatchGenerationScheduler.this, r, "jmlx-batch-completion"));
    }

    @Override
    protected void terminated() {
      state.updateAndGet(s -> s == State.FAILED ? s : State.CLOSED);
    }
  }

  /** Everything the worker tracks for one admitted request. */
  private static final class Req {
    final GenerationRequest request;
    final Consumer<GenerationEvent> listener;
    final BatchRequestHandle handle;
    final int[] prompt;
    final GenerationConfig policy;
    final KVCachePolicy cachePolicy;
    final HfTokenizer tokenizer;
    final List<Integer> generated = new ArrayList<>();
    final List<Double> logProbabilities = new ArrayList<>();
    final StringBuilder text;
    final LinkedHashMap<Integer, Integer> frequencies;
    IncrementalTokenDecoder decoder;
    SamplingPipeline sampler;
    RuntimeException tokenFailure;
    boolean done;

    Req(
        GenerationRequest request,
        Consumer<GenerationEvent> listener,
        BatchRequestHandle handle,
        KVCachePolicy cachePolicy) {
      this.request = request;
      this.listener = listener;
      this.handle = handle;
      this.prompt = request.promptTokenIds();
      this.policy = request.config();
      this.cachePolicy = cachePolicy;
      this.tokenizer = request.tokenizer();
      this.text = tokenizer == null ? null : new StringBuilder();
      this.frequencies = PenaltyInputs.frequencies(prompt);
    }

    int lastToken() {
      return generated.getLast();
    }
  }

  private final BatchSchedulerConfig config;
  private final ModelFactory factory;
  private final Hooks hooks;
  private final Dispatcher dispatcher;
  private final Thread worker;
  private final Semaphore permits;
  private final ReentrantLock lock = new ReentrantLock();
  private final Condition notEmpty = lock.newCondition();
  private final ArrayDeque<Req> waiting = new ArrayDeque<>();
  private final AtomicReference<State> state = new AtomicReference<>(State.RUNNING);
  private final CountDownLatch ready = new CountDownLatch(1);
  private final AtomicLong forwardCalls = new AtomicLong();
  private final CohortSizeWindow cohortSizes = new CohortSizeWindow();
  private volatile Throwable startupFailure;
  private volatile SchedulerFailedException schedulerFailure;
  private volatile DecoderModel.BatchFacts facts;
  private boolean started;
  private DecoderModel decoder;
  private List<Req> activeRows = List.of();

  private BatchGenerationScheduler(BatchSchedulerConfig config, ModelFactory factory, Hooks hooks) {
    this.config = config;
    this.factory = factory;
    this.hooks = hooks;
    this.permits = new Semaphore(config.admissionPermits());
    this.dispatcher = new Dispatcher(config.admissionPermits());
    this.worker = new SchedulerThread(this, this::run, "jmlx-batch-worker");
  }

  /**
   * Starts a scheduler whose worker thread loads the model with {@code factory}. Blocks until the
   * model is ready or startup fails; the wait is uninterruptible (an interrupt is restored, never
   * thrown) because returning early would leave a live worker, holding the guard and the model,
   * that no caller references. There is no startup timeout: a factory that never returns blocks
   * this call.
   *
   * @throws SchedulerAlreadyRunningException if another scheduler's worker has not exited
   * @throws SchedulerStartException if the factory throws a checked exception or returns something
   *     that is not a usable {@code DecoderModel}; unchecked factory failures propagate unchanged
   */
  public static BatchGenerationScheduler start(BatchSchedulerConfig config, ModelFactory factory) {
    return start(config, factory, Hooks.NONE);
  }

  /**
   * Starts a scheduler like {@link #start(BatchSchedulerConfig, ModelFactory)}, whose worker forms
   * no cohort until {@code cohortGate} accepts the number of waiting requests. This pins the batch
   * shape: for example, {@code waiting -> waiting >= 2} holds the first request until the second
   * arrives, so the two are guaranteed to share one cohort.
   *
   * <p>The gate is consulted every time the worker wakes, against the total number of queued
   * requests, not the size of one compatibility group (see {@code CohortKey}); requests already
   * waiting that cannot cohort together are released when {@code maxBatchWait} elapses, so a
   * request the gate is holding never waits longer than that (and a {@link BatchRequestHandle
   * #cancel()} wakes the worker at once, so a cancelled waiting request completes without waiting
   * for the next request or {@link #close()}).
   *
   * <p>The predicate runs on the worker while it holds the admission lock, the same lock {@link
   * #submit} takes: it must be fast and side-effect free, and a slow one blocks {@code submit}
   * while it runs. A predicate that throws is treated as {@code true} (release now) and logged; it
   * never fails the scheduler or strands the queue.
   *
   * @param cohortGate consulted with the number of waiting requests; {@code true} releases the
   *     queue as a cohort
   * @param maxBatchWait how long the worker may hold a queue the gate rejects, measured from when
   *     it first holds one (the clock is not restarted by wakes); after it elapses the waiting
   *     requests form a cohort anyway. For {@code waiting -> waiting >= n} pick a fraction of the
   *     latency the batch is meant to save
   * @throws IllegalArgumentException if {@code maxBatchWait} is zero, negative, or does not fit a
   *     nanoseconds count
   * @throws SchedulerAlreadyRunningException if another scheduler's worker has not exited
   * @throws SchedulerStartException if the factory throws a checked exception or returns something
   *     that is not a usable {@code DecoderModel}; unchecked factory failures propagate unchanged
   */
  public static BatchGenerationScheduler start(
      BatchSchedulerConfig config,
      ModelFactory factory,
      IntPredicate cohortGate,
      Duration maxBatchWait) {
    Objects.requireNonNull(cohortGate, "cohortGate");
    Objects.requireNonNull(maxBatchWait, "maxBatchWait");
    if (maxBatchWait.isNegative() || maxBatchWait.isZero()) {
      throw new IllegalArgumentException("maxBatchWait must be positive: " + maxBatchWait);
    }
    try {
      maxBatchWait.toNanos();
    } catch (ArithmeticException e) {
      throw new IllegalArgumentException(
          "maxBatchWait does not fit a nanoseconds count: " + maxBatchWait, e);
    }
    return start(config, factory, Hooks.NONE.withCohortGate(cohortGate, maxBatchWait));
  }

  static BatchGenerationScheduler start(
      BatchSchedulerConfig config, ModelFactory factory, Hooks hooks) {
    Objects.requireNonNull(config, "config");
    Objects.requireNonNull(factory, "factory");
    Objects.requireNonNull(hooks, "hooks");
    if (!ACTIVE.compareAndSet(false, true)) {
      throw new SchedulerAlreadyRunningException();
    }
    BatchGenerationScheduler scheduler;
    try {
      scheduler = new BatchGenerationScheduler(config, factory, hooks);
    } catch (Throwable t) {
      ACTIVE.set(false);
      throw t;
    }
    try {
      scheduler.worker.start();
    } catch (Throwable t) {
      // No worker finally will ever run, so release what the exit sequence would have released.
      ACTIVE.set(false);
      scheduler.dispatcher.shutdown();
      throw t;
    }
    boolean interrupted = awaitUninterruptibly(scheduler.ready);
    Throwable failure = scheduler.startupFailure;
    if (failure != null) {
      interrupted |= joinUninterruptibly(scheduler.worker);
      restoreInterrupt(interrupted);
      if (failure instanceof RuntimeException e) {
        throw e;
      }
      if (failure instanceof Error e) {
        throw e;
      }
      throw new SchedulerStartException("model factory failed: " + failure, failure);
    }
    restoreInterrupt(interrupted);
    return scheduler;
  }

  /**
   * Submits a request. Never blocks and never touches MLX. Validation happens here, on the caller
   * thread, against model facts the worker snapshotted at startup.
   *
   * @throws IllegalArgumentException if the request can never be served (bad token IDs, a prompt
   *     over the prompt-token budget even alone, {@code maxNewTokens} above the scheduler's cap, a
   *     cache policy the checkpoint cannot honor, or bounded-FULL capacity); it takes no admission
   *     permit
   * @throws BatchAdmissionRejectedException if the scheduler is at capacity right now
   * @throws SchedulerClosedException if the scheduler is closing or closed
   * @throws SchedulerFailedException if the worker failed
   */
  public BatchRequestHandle submit(GenerationRequest request, Consumer<GenerationEvent> listener) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(listener, "listener");
    requireAcceptingState();
    KVCachePolicy cachePolicy = validate(request);
    if (!permits.tryAcquire()) {
      throw new BatchAdmissionRejectedException(
          "admission permits exhausted (" + config.admissionPermits() + " in flight)");
    }
    BatchRequestHandle handle = new BatchRequestHandle(this::wakeWorker);
    lock.lock();
    try {
      requireAcceptingState();
      if (waiting.size() >= config.maxQueuedRequests()) {
        permits.release();
        throw new BatchAdmissionRejectedException(
            "waiting queue is full (" + config.maxQueuedRequests() + " waiting)");
      }
      waiting.addLast(new Req(request, listener, handle, cachePolicy));
      notEmpty.signalAll();
    } catch (RuntimeException e) {
      if (!(e instanceof BatchAdmissionRejectedException)) {
        permits.release();
      }
      throw e;
    } finally {
      lock.unlock();
    }
    return handle;
  }

  /** The lifecycle state; never blocks. */
  public State state() {
    return state.get();
  }

  /** The failure that moved the scheduler to {@link State#FAILED}, if any. */
  public Optional<Throwable> failure() {
    SchedulerFailedException failed = schedulerFailure;
    return failed == null ? Optional.empty() : Optional.of(failed.getCause());
  }

  /**
   * Total model forwards run so far; a cohort of {@code B} rows costs one per step, not {@code B}.
   */
  long forwardCalls() {
    return forwardCalls.get();
  }

  /**
   * The size in rows of the most recent cohorts the worker has started, in start order, at most the
   * last {@value CohortSizeWindow#CAPACITY}. A cohort reports the size it started with even if rows
   * finish earlier, and a request cancelled or rejected before its cohort starts is not counted.
   * The window is bounded so a long-lived scheduler does not grow per cohort; for lifetime totals
   * use {@link #cohortCount()} and {@link #rowsBatched()}.
   */
  public List<Integer> cohortSizes() {
    return cohortSizes.recent();
  }

  /**
   * The total number of cohorts the worker has started, including any beyond {@link
   * #cohortSizes()}.
   */
  public long cohortCount() {
    return cohortSizes.cohorts();
  }

  /** The total rows started across all cohorts (the sum of their sizes). */
  public long rowsBatched() {
    return cohortSizes.rows();
  }

  private void requireAcceptingState() {
    State current = state.get();
    if (current == State.FAILED) {
      // A fresh instance per call: the shared one carries the worker's stack trace, and callers
      // on different threads could modify it concurrently (addSuppressed from
      // try-with-resources, initCause).
      throw new SchedulerFailedException(schedulerFailure.getCause());
    }
    if (current != State.RUNNING) {
      throw new SchedulerClosedException("scheduler is " + current.name().toLowerCase());
    }
  }

  private KVCachePolicy validate(GenerationRequest request) {
    DecoderModel.BatchFacts model = facts;
    GenerationConfig policy = request.config();
    int[] prompt = request.promptTokenIds();
    DecoderModel.validateTokenIds(prompt, policy, model.vocabSize());
    KVCachePolicy cachePolicy = request.cachePolicy().resolve(model.slidingWindow());
    if (policy.maxNewTokens() > config.maxNewTokensPerRequest()) {
      throw new IllegalArgumentException(
          "maxNewTokens "
              + policy.maxNewTokens()
              + " exceeds the scheduler's maxNewTokensPerRequest "
              + config.maxNewTokensPerRequest());
    }
    if (prompt.length > config.maxPromptTokenBudget()) {
      throw new IllegalArgumentException(
          "prompt length "
              + prompt.length
              + " exceeds the scheduler's maxPromptTokenBudget "
              + config.maxPromptTokenBudget());
    }
    long required =
        policy.maxNewTokens() == 0 ? 0 : (long) prompt.length + policy.maxNewTokens() - 1;
    cachePolicy.requireCapacity(Math.min(required, Integer.MAX_VALUE), "generation");
    HfTokenizer tokenizer = request.tokenizer();
    if (tokenizer != null && tokenizer.vocabSize() > model.vocabSize()) {
      throw new IllegalArgumentException(
          "tokenizer vocabulary size "
              + tokenizer.vocabSize()
              + " exceeds checkpoint vocabulary size "
              + model.vocabSize());
    }
    return cachePolicy;
  }

  /**
   * Stops admission, drains, and waits for the worker and every accepted stage. Safe to call
   * repeatedly. From the worker or the completion dispatcher it only stops admission and returns
   * without waiting. Returns normally even after a failure; use {@link #state()} and {@link
   * #failure()}.
   */
  @Override
  public void close() {
    stopAdmission();
    if (Thread.currentThread() instanceof SchedulerThread own && own.owner == this) {
      return;
    }
    boolean interrupted = joinUninterruptibly(worker);
    while (true) {
      try {
        dispatcher.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        break;
      } catch (InterruptedException e) {
        interrupted = true;
      }
    }
    restoreInterrupt(interrupted);
  }

  private void stopAdmission() {
    lock.lock();
    try {
      state.compareAndSet(State.RUNNING, State.CLOSING);
      notEmpty.signalAll();
    } finally {
      lock.unlock();
    }
  }

  /**
   * Wakes the worker so it re-polls the waiting queue. Cancellation installs this as the handle's
   * waker: without it, a cancelled request held by the cohort gate would sit pending (and holding
   * its admission permit) until the next submit, the gate wait, or {@link #close()}.
   */
  private void wakeWorker() {
    lock.lock();
    try {
      notEmpty.signalAll();
    } finally {
      lock.unlock();
    }
  }

  // ---------------------------------------------------------------------------------- worker

  private void run() {
    MLXScope root = null;
    TextGenerationModel model = null;
    Throwable runtimeFailure = null;
    try {
      root = new MLXScope();
      model = factory.create(root);
      decoder = acceptModel(root, model);
      facts = decoder.batchFacts();
      started = true;
      ready.countDown();
      serve();
    } catch (Throwable t) {
      if (!started) {
        startupFailure = t;
        ready.countDown();
      } else {
        runtimeFailure = t;
        failWorker(t);
      }
    } finally {
      exit(model, root);
    }
    if (runtimeFailure instanceof Error e) {
      throw e;
    }
  }

  private static DecoderModel acceptModel(MLXScope root, TextGenerationModel model) {
    if (!(model instanceof DecoderModel decoderModel)) {
      throw new SchedulerStartException(
          "the model factory must return a DecoderModel, got "
              + (model == null ? "null" : model.getClass().getName()));
    }
    if (!root.isAncestorOf(decoderModel.modelScope())) {
      throw new SchedulerStartException(
          "the model's scope is not the worker's root scope or a descendant of it");
    }
    return decoderModel;
  }

  /**
   * The exit sequence, run on every worker exit path in this order: close the model and root (best
   * effort), release the one-scheduler guard, then shut the dispatcher down last. The order makes
   * {@link State#CLOSED} unobservable while the guard is held, wherever {@code terminated()} runs.
   */
  private void exit(TextGenerationModel model, MLXScope root) {
    try {
      Throwable cleanup = null;
      try {
        if (model != null) {
          model.close();
        }
        if (hooks.cleanupFault() != null) {
          hooks.cleanupFault().run();
        }
      } catch (Throwable t) {
        cleanup = t;
      }
      try {
        if (root != null) {
          root.close();
        }
      } catch (Throwable t) {
        if (cleanup == null) {
          cleanup = t;
        } else {
          cleanup.addSuppressed(t);
        }
      }
      if (cleanup != null && started) {
        recordCleanupFailure(cleanup);
      }
    } finally {
      ACTIVE.set(false);
      dispatcher.shutdown();
    }
  }

  private void recordCleanupFailure(Throwable cleanup) {
    if (schedulerFailure == null) {
      schedulerFailure = new SchedulerFailedException(cleanup);
    }
    state.set(State.FAILED);
  }

  private void serve() {
    while (true) {
      List<Req> cohort = nextCohort();
      if (cohort == null) {
        return;
      }
      runCohort(cohort);
    }
  }

  /** Blocks for the next cohort, or returns null once admission stopped and the queue is empty. */
  private List<Req> nextCohort() {
    // The wall-clock deadline of the current gate wait, or -1 while the worker is not gate-blocked.
    // It is set once per hold and not restarted by wakes, so a queue trickling requests below the
    // gate's threshold cannot hold a cohort past maxBatchWait.
    long gateDeadline = -1;
    while (true) {
      drainWaiting();
      lock.lock();
      try {
        if (waiting.isEmpty()) {
          if (state.get() != State.RUNNING) {
            return null;
          }
          gateDeadline = -1;
          notEmpty.awaitUninterruptibly();
          continue;
        }
        IntPredicate gate = hooks.cohortGate();
        if (gate != null && state.get() == State.RUNNING && !gateAdmits(gate, waiting.size())) {
          if (anyWaitingCancelled()) {
            // A cancel landed after drainWaiting() ran (it runs before the lock), so its signal
            // would be lost if we parked. Loop back and drain it now instead of holding the queue
            // until the gate wait expires. close()/failWorker() need no equivalent: their state
            // change is read in this same lock section.
            continue;
          }
          if (gateDeadline < 0) {
            gateDeadline = System.nanoTime() + hooks.cohortGateMaxWait().toNanos();
          }
          long remaining = gateDeadline - System.nanoTime();
          if (remaining <= 0) {
            // The wait expired: release what is waiting rather than strand it until close().
            return formCohort();
          }
          boolean interrupted = awaitGate(gateDeadline);
          restoreInterrupt(interrupted);
          continue;
        }
        gateDeadline = -1;
        return formCohort();
      } finally {
        lock.unlock();
      }
    }
  }

  /**
   * Called with the lock held: whether any waiting request would be drained by the next {@link
   * #drainWaiting()}. Checked under the lock right before the gate wait parks, so no cancellation
   * loses its wake-up: a flag set before the check is seen here, and one set after it cannot miss
   * the park, because this check and the park run in one lock section while the waker takes that
   * same lock to signal.
   */
  private boolean anyWaitingCancelled() {
    for (Req r : waiting) {
      if (poll(r) != Poll.LIVE) {
        return true;
      }
    }
    return false;
  }

  /**
   * Asks the cohort gate whether to release the queue now. A gate that throws is treated as {@code
   * true} (release now) and logged: user code must not be able to fail the worker or strand the
   * queue it runs on.
   */
  private boolean gateAdmits(IntPredicate gate, int waitingRequests) {
    try {
      return gate.test(waitingRequests);
    } catch (RuntimeException e) {
      LOGGER.log(
          System.Logger.Level.WARNING, "the cohort gate threw; releasing the waiting requests", e);
      return true;
    }
  }

  /**
   * Waits on the gate condition until its deadline or a wake, uninterruptibly: the worker keeps
   * waiting through an interrupt and reports it for restoration.
   */
  private boolean awaitGate(long deadline) {
    boolean interrupted = false;
    while (true) {
      long remaining = deadline - System.nanoTime();
      if (remaining <= 0) {
        return interrupted;
      }
      try {
        notEmpty.awaitNanos(remaining);
        return interrupted;
      } catch (InterruptedException e) {
        interrupted = true;
        // Clear the flag so the next awaitNanos parks instead of rethrowing immediately; the
        // caller restores it once the gate wait ends.
        Thread.interrupted();
      }
    }
  }

  /** Completes cancelled waiting requests; when closing, every waiting request is cancelled. */
  private void drainWaiting() {
    List<Req> snapshot;
    lock.lock();
    try {
      snapshot = new ArrayList<>(waiting);
    } finally {
      lock.unlock();
    }
    for (Req r : snapshot) {
      Poll poll = poll(r);
      if (poll == Poll.LIVE) {
        continue;
      }
      lock.lock();
      try {
        waiting.remove(r);
      } finally {
        lock.unlock();
      }
      if (poll == Poll.CANCELLED) {
        finishRow(r, FinishReason.CANCELLED);
      } else {
        abort(r, "cancellation token", null, r.tokenFailure);
      }
    }
  }

  private Poll poll(Req r) {
    if (state.get() != State.RUNNING || r.handle.isCancellationRequested()) {
      return Poll.CANCELLED;
    }
    try {
      return r.request.cancellationToken().isCancelled() ? Poll.CANCELLED : Poll.LIVE;
    } catch (RuntimeException e) {
      r.tokenFailure = e;
      return Poll.TOKEN_FAILED;
    }
  }

  /**
   * Called with the lock held. The oldest waiting request fixes the compatibility group; later
   * requests are scanned only to fill the cohort without exceeding the batch size or the
   * prompt-token budget (rows times the maximum padded width), and, for a bounded FULL cache,
   * without making the shared padded width exceed any row's capacity.
   */
  private List<Req> formCohort() {
    Req first = waiting.peekFirst();
    CohortKey key = keyOf(first);
    List<Req> cohort = new ArrayList<>();
    cohort.add(first);
    int width = first.prompt.length;
    for (Req candidate : waiting) {
      if (cohort.size() >= config.maxBatchSize()) {
        break;
      }
      if (candidate == first || !keyOf(candidate).equals(key)) {
        continue;
      }
      int widened = Math.max(width, candidate.prompt.length);
      if ((long) (cohort.size() + 1) * widened > config.maxPromptTokenBudget()
          || !fitsCapacity(cohort, candidate, widened)) {
        continue;
      }
      cohort.add(candidate);
      width = widened;
    }
    waiting.removeAll(cohort);
    return cohort;
  }

  private static boolean fitsCapacity(List<Req> cohort, Req candidate, int width) {
    KVCachePolicy policy = candidate.cachePolicy;
    if (policy.mode() != KVCachePolicy.Mode.FULL || policy.limit() == 0) {
      return true;
    }
    for (Req r : cohort) {
      if (maxNewWidth(r, width) > policy.limit()) {
        return false;
      }
    }
    return maxNewWidth(candidate, width) <= policy.limit();
  }

  private static long maxNewWidth(Req r, int width) {
    return r.policy.maxNewTokens() == 0 ? 0 : (long) width + r.policy.maxNewTokens() - 1;
  }

  private CohortKey keyOf(Req r) {
    return new CohortKey(r.cachePolicy, facts.dynamicNtk() ? r.prompt.length : -1);
  }

  // ---------------------------------------------------------------------------------- cohort

  private void runCohort(List<Req> cohort) {
    activeRows = cohort;
    cohortSizes.record(cohort.size());
    List<Req> live = new ArrayList<>();
    RuntimeException setupFailure = null;
    // Set once the samplers exist and the decode loop starts: a failure after that point (for
    // example one escaping decodeLoop) is attributed to stage "cohort", not "cohort setup", even
    // though both fail the cohort the same way.
    boolean setupDone = false;
    MLXScope cohortScope = null;
    try {
      for (Req r : cohort) {
        Poll poll = poll(r);
        if (poll == Poll.CANCELLED) {
          finishRow(r, FinishReason.CANCELLED);
        } else if (poll == Poll.TOKEN_FAILED) {
          abort(r, "cancellation token", null, r.tokenFailure);
        } else {
          r.decoder = r.tokenizer == null ? null : r.tokenizer.newIncrementalDecoder(true);
          if (r.policy.maxNewTokens() == 0) {
            finishRow(r, FinishReason.MAX_TOKENS);
          } else {
            live.add(r);
          }
        }
      }
      if (!live.isEmpty()) {
        if (hooks.setupFault() != null) {
          hooks.setupFault().run();
        }
        int vocab = facts.vocabSize();
        cohortScope = decoder.modelScope().newChild();
        // Samplers hold keys in the cohort scope, so they must close before it does -- exactly the
        // order the direct path's try-with-resources gives (sampler first, then its scope).
        try {
          List<KVCache> caches = new ArrayList<>();
          for (int layer = 0; layer < facts.layerCount(); layer++) {
            caches.add(new KVCache(cohortScope, live.getFirst().cachePolicy));
          }
          boolean needsRank =
              live.stream().anyMatch(r -> SamplingPipeline.needsRankIndices(r.policy));
          MLXArray rank = needsRank ? SamplingPipeline.newRankIndices(cohortScope, vocab) : null;
          for (Req r : live) {
            r.sampler =
                new SamplingPipeline(
                    cohortScope,
                    r.policy,
                    vocab,
                    decoder.stepBoundaryEvaluator(),
                    SamplingPipeline.needsRankIndices(r.policy) ? rank : null);
          }
          setupDone = true;
          decodeLoop(cohortScope, live, caches);
        } finally {
          for (Req r : cohort) {
            closeSampler(r);
          }
        }
      }
    } catch (RuntimeException e) {
      // A cohort-level failure, not a worker failure -- the cohort's rows get it and the worker
      // keeps serving later cohorts. Only an Error escapes to failWorker. (decodeLoop handles its
      // own step and compaction failures with their own stages; a RuntimeException that still
      // escapes it happens mid-decode, after setup finished.)
      setupFailure = e;
    } finally {
      if (cohortScope != null) {
        cohortScope.close();
      }
    }
    if (setupFailure != null) {
      // abort() and finishRow() skip completed rows, so the whole cohort can be passed: a row
      // that never got set up is failed like one that was, and rows the first loop already
      // finished with CANCELLED/MAX_TOKENS keep their outcomes. The stage names when the failure
      // happened: before the decode loop started, or somewhere inside it.
      failCohort(cohort, setupDone ? "cohort" : "cohort setup", setupFailure);
    }
    // activeRows is cleared only on a normal return: if an Error escaped, failWorker still has to
    // see this cohort so it can complete every one of its stages.
    activeRows = List.of();
  }

  private void decodeLoop(MLXScope cohortScope, List<Req> live, List<KVCache> caches) {
    int step = 0;
    while (!live.isEmpty()) {
      SamplingPipeline.RowReadBack[] rows;
      try {
        rows = runStep(cohortScope, live, caches, step);
      } catch (RuntimeException failure) {
        caches.forEach(KVCache::poison);
        failCohort(live, "batch step", failure);
        return;
      }
      List<Req> survivors = new ArrayList<>();
      List<Integer> keep = new ArrayList<>();
      for (int row = 0; row < live.size(); row++) {
        Req r = live.get(row);
        if (processRow(r, rows[row], step) == Outcome.CONTINUE && stillWanted(r)) {
          survivors.add(r);
          keep.add(row);
        }
      }
      if (survivors.isEmpty()) {
        return;
      }
      if (survivors.size() != live.size()) {
        try {
          compact(cohortScope, caches, keep.stream().mapToInt(Integer::intValue).toArray());
        } catch (RuntimeException failure) {
          failCohort(survivors, "cache compaction", failure);
          return;
        }
      }
      live.clear();
      live.addAll(survivors);
      drainWaiting();
      // Outside runStep's own catch, so a fault injected here escapes the loop and is attributed
      // to stage "cohort" (setup had finished), not "cohort setup".
      if (hooks.decodeFault() != null) {
        hooks.decodeFault().run();
      }
      step++;
    }
  }

  /** One joint step: a single forward, per-row lazy selection, and a single joint evaluation. */
  private SamplingPipeline.RowReadBack[] runStep(
      MLXScope cohortScope, List<Req> live, List<KVCache> caches, int step) {
    int batch = live.size();
    int width = 1;
    if (step == 0) {
      width = live.stream().mapToInt(r -> r.prompt.length).max().orElseThrow();
    }
    int[] flat = new int[batch * width];
    int[] valid = new int[batch];
    for (int row = 0; row < batch; row++) {
      Req r = live.get(row);
      if (step == 0) {
        valid[row] = r.prompt.length;
        System.arraycopy(r.prompt, 0, flat, row * width + (width - valid[row]), valid[row]);
      } else {
        valid[row] = 1;
        flat[row] = r.lastToken();
      }
    }
    int vocab = facts.vocabSize();
    try (MLXScope stepScope = cohortScope.newChild()) {
      MLXArray ids = MLX.array(stepScope, flat, new int[] {batch, width});
      forwardCalls.incrementAndGet();
      MLXArray logits = decoder.stepLogits(ids, caches, valid, hooks.embeddingHook());
      List<SamplingPipeline.Built> built = new ArrayList<>(batch);
      boolean[] reports = new boolean[batch];
      for (int row = 0; row < batch; row++) {
        Req r = live.get(row);
        MLXArray rowLogits =
            MLXShape.slice(logits, new int[] {row, 0, 0}, new int[] {row + 1, 1, vocab});
        built.add(r.sampler.build(rowLogits, PenaltyInputs.from(r.frequencies, vocab)));
        reports[row] = r.policy.logProbabilities();
      }
      SamplingPipeline.Batched batched = SamplingPipeline.stack(stepScope, built);
      SamplingPipeline.evaluate(
          decoder.stepBoundaryEvaluator(), batched, DecoderModel.stepCacheArrays(caches));
      return SamplingPipeline.readBack(batched, reports);
    }
  }

  /** Applies one row's step result, mirroring the direct path's event and token semantics. */
  private Outcome processRow(Req r, SamplingPipeline.RowReadBack rb, int step) {
    if (!rb.finite()) {
      abort(r, "non-finite logits", null, SamplingPipeline.nonFiniteLogits(step));
      return Outcome.DONE;
    }
    if (!rb.temperedFinite()) {
      abort(r, "non-finite logits", null, SamplingPipeline.nonFiniteTempered(step));
      return Outcome.DONE;
    }
    int next = rb.tokenId();
    boolean eos = r.policy.eosTokenIds().contains(next);
    if (!eos && r.policy.stopTokenIds().contains(next)) {
      finishRow(r, FinishReason.STOP_TOKEN);
      return Outcome.DONE;
    }
    r.generated.add(next);
    r.frequencies.merge(next, 1, Integer::sum);
    if (rb.logProbability() != null) {
      r.logProbabilities.add(rb.logProbability());
    }
    String textDelta = null;
    if (r.decoder != null) {
      try {
        textDelta = r.decoder.append(next);
        r.text.append(textDelta);
      } catch (TokenizerException e) {
        abort(r, "output decoder", next, e);
        return Outcome.DONE;
      }
    }
    try {
      r.listener.accept(DecoderModel.tokenEvent(next, textDelta, rb.logProbability()));
    } catch (RuntimeException e) {
      abort(r, "listener", null, e);
      return Outcome.DONE;
    }
    if (eos) {
      finishRow(r, FinishReason.EOS);
      return Outcome.DONE;
    }
    if (r.generated.size() >= r.policy.maxNewTokens()) {
      finishRow(r, FinishReason.MAX_TOKENS);
      return Outcome.DONE;
    }
    return Outcome.CONTINUE;
  }

  /** Polls a row that would decode another step; a cancelled or failing token ends it now. */
  private boolean stillWanted(Req r) {
    Poll poll = poll(r);
    if (poll == Poll.LIVE) {
      return true;
    }
    if (poll == Poll.CANCELLED) {
      finishRow(r, FinishReason.CANCELLED);
    } else {
      abort(r, "cancellation token", null, r.tokenFailure);
    }
    return false;
  }

  /**
   * Compacts every layer's cache to the surviving rows, one layer at a time: {@code new =
   * reorder(old); install(new); old.reset()}. Peak extra memory is one layer's K/V, not a second
   * full cache set. If layer {@code k} fails, the old layers {@code k..L-1} and the installed new
   * layers {@code 0..k-1} are reset and the failure propagates; a partly reordered cache is never
   * decoded from.
   */
  private void compact(MLXScope cohortScope, List<KVCache> caches, int[] keep) {
    for (int layer = 0; layer < caches.size(); layer++) {
      KVCache old = caches.get(layer);
      KVCache replacement;
      try {
        replacement = hooks.reorderer().reorder(old, layer, keep, cohortScope);
      } catch (RuntimeException failure) {
        caches.forEach(KVCache::reset);
        throw failure;
      }
      caches.set(layer, replacement);
      old.reset();
    }
  }

  private void failCohort(List<Req> rows, String stage, RuntimeException cause) {
    for (Req r : new ArrayList<>(rows)) {
      abort(r, stage, null, cause);
    }
  }

  // ------------------------------------------------------------------------------- completion

  /**
   * Mirrors the direct path's tail: flush the decoder, build the result, send one terminal event.
   */
  private void finishRow(Req r, FinishReason reason) {
    if (r.done) {
      return;
    }
    String terminalDelta = null;
    if (r.tokenizer != null) {
      terminalDelta = "";
      if (r.decoder != null) {
        try {
          terminalDelta = r.decoder.finish();
        } catch (TokenizerException e) {
          abort(r, "output decoder finish", null, e);
          return;
        }
      }
      r.text.append(terminalDelta);
    }
    GenerationResult result =
        new GenerationResult(
            toList(r.prompt),
            r.generated,
            reason,
            r.logProbabilities,
            r.text == null ? null : r.text.toString());
    try {
      r.listener.accept(
          terminalDelta == null
              ? GenerationEvent.finished(reason)
              : GenerationEvent.finished(reason, terminalDelta));
    } catch (RuntimeException e) {
      LOGGER.log(System.Logger.Level.WARNING, "batch terminal listener failed", e);
    }
    closeSampler(r);
    dispatch(r, result, null);
  }

  private void abort(Req r, String stage, Integer failingToken, RuntimeException cause) {
    if (r.done) {
      return;
    }
    closeSampler(r);
    dispatch(
        r,
        null,
        new GenerationAbortedException(toList(r.prompt), r.generated, stage, failingToken, cause));
  }

  /** Hands a finished request to the dispatcher; its admission permit is released afterwards. */
  private void dispatch(Req r, GenerationResult result, Throwable error) {
    if (r.done) {
      return;
    }
    r.done = true;
    Runnable completion =
        () -> {
          try {
            if (error == null) {
              r.handle.future().complete(result);
            } else {
              r.handle.future().completeExceptionally(error);
            }
          } finally {
            permits.release();
          }
        };
    try {
      dispatcher.execute(completion);
    } catch (RejectedExecutionException e) {
      // Unreachable: the queue is sized to the admission permits and shutdown comes last.
      completion.run();
    }
  }

  private static void closeSampler(Req r) {
    if (r.sampler != null) {
      try {
        r.sampler.close();
      } catch (RuntimeException e) {
        LOGGER.log(System.Logger.Level.WARNING, "closing a row's sampler failed", e);
      }
      r.sampler = null;
    }
  }

  /**
   * The worker can no longer serve: stop admission atomically, complete every queued and active
   * stage with a {@link GenerationAbortedException} caused by a {@link SchedulerFailedException},
   * and let the exit sequence close the model and root. No listener gets a terminal event.
   */
  private void failWorker(Throwable cause) {
    SchedulerFailedException failed = new SchedulerFailedException(cause);
    List<Req> queued;
    lock.lock();
    try {
      schedulerFailure = failed;
      state.set(State.FAILED);
      queued = new ArrayList<>(waiting);
      waiting.clear();
      notEmpty.signalAll();
    } finally {
      lock.unlock();
    }
    for (Req r : activeRows) {
      abort(r, "scheduler failed", null, failed);
    }
    for (Req r : queued) {
      abort(r, "scheduler failed", null, failed);
    }
    activeRows = List.of();
  }

  // ---------------------------------------------------------------------------------- helpers

  private static List<Integer> toList(int[] ids) {
    return Arrays.stream(ids).boxed().toList();
  }

  private static boolean awaitUninterruptibly(CountDownLatch latch) {
    boolean interrupted = false;
    while (true) {
      try {
        latch.await();
        return interrupted;
      } catch (InterruptedException e) {
        interrupted = true;
      }
    }
  }

  private static boolean joinUninterruptibly(Thread thread) {
    boolean interrupted = false;
    while (true) {
      try {
        thread.join();
        return interrupted;
      } catch (InterruptedException e) {
        interrupted = true;
      }
    }
  }

  private static void restoreInterrupt(boolean interrupted) {
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }
}
