package se.alipsa.jmlx.models;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A submitted request: a read-only completion stage and a thread-safe cancellation signal. It owns
 * no native object.
 *
 * <p>A failed request completes {@link #stage()} exceptionally, but a dependent attached to that
 * stage receives a {@link java.util.concurrent.CompletionException} whose <em>cause</em> is the
 * real exception (typically a {@link GenerationAbortedException}); unwrap it. {@code join()} throws
 * the wrapper and {@code get()} throws an {@link java.util.concurrent.ExecutionException}.
 *
 * <p>Dependents run on the scheduler's completion dispatcher, never on the MLX worker, with one
 * exception: a non-async dependent attached to an <em>already completed</em> stage runs immediately
 * on the attaching thread. If that thread is the worker (for example a token listener calling
 * {@code otherHandle.stage().thenAccept(...)}), the observer runs on the worker, so it must not
 * block or use MLX. Use the {@code *Async} forms from callbacks. Observers must return promptly and
 * must not block on another request's stage, or a bounded dispatcher can deadlock.
 */
public final class BatchRequestHandle {
  private final CompletableFuture<GenerationResult> future = new CompletableFuture<>();
  private final CompletionStage<GenerationResult> stage = future.minimalCompletionStage();
  private final AtomicBoolean cancelRequested = new AtomicBoolean();
  // The owning scheduler's wake; the flag alone would not reach a worker blocked on the cohort
  // gate, which nothing else signals.
  private final Runnable cancelWaker;

  BatchRequestHandle(Runnable cancelWaker) {
    this.cancelWaker = cancelWaker;
  }

  /**
   * The request's completion. Cancelling this stage is not a request cancellation; use {@link
   * #cancel()}.
   */
  public CompletionStage<GenerationResult> stage() {
    return stage;
  }

  /**
   * Requests cancellation. Safe from any thread and idempotent. The worker observes it before
   * prefill and between decode steps, never in the middle of a native call, so the request still
   * completes normally with {@link FinishReason#CANCELLED} and its partial result. Cancellation
   * also wakes a worker that is waiting, so a request that is still queued (for example held by the
   * cohort gate) completes at once, not when the next request arrives or the scheduler closes.
   */
  public void cancel() {
    if (cancelRequested.compareAndSet(false, true)) {
      cancelWaker.run();
    }
  }

  /** Whether {@link #cancel()} has been called. */
  public boolean isCancellationRequested() {
    return cancelRequested.get();
  }

  CompletableFuture<GenerationResult> future() {
    return future;
  }
}
