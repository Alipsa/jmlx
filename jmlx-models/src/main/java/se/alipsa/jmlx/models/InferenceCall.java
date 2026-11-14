package se.alipsa.jmlx.models;

import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/**
 * A generation stage with explicit cancellation and an independent failure signal. Failure-signal
 * observers must return promptly: they may run on the MLX worker and must not call user code.
 */
public record InferenceCall(
    CompletionStage<GenerationResult> stage,
    Runnable cancellation,
    CompletionStage<Throwable> failureSignal) {
  /** Requires the result, cancellation action and independent failure signal. */
  public InferenceCall {
    Objects.requireNonNull(stage, "stage");
    Objects.requireNonNull(cancellation, "cancellation");
    Objects.requireNonNull(failureSignal, "failureSignal");
  }

  /**
   * Creates a call whose failure signal follows its result; streaming clients can signal earlier.
   */
  public InferenceCall(CompletionStage<GenerationResult> stage, Runnable cancellation) {
    this(stage, cancellation, resultFailure(stage));
  }

  private static CompletionStage<Throwable> resultFailure(CompletionStage<GenerationResult> stage) {
    CompletableFuture<Throwable> signal = new CompletableFuture<>();
    stage.whenComplete(
        (value, failure) -> {
          if (failure != null) {
            signal.complete(SchedulerInferenceClient.unwrap(failure));
          } else if (value.finishReason() == FinishReason.CANCELLED) {
            signal.complete(new CancellationException("generation cancelled"));
          }
        });
    return signal.minimalCompletionStage();
  }

  /** Waits interruptibly, cancels on interruption and propagates the underlying runtime failure. */
  public GenerationResult await() {
    return InferenceAwait.completed(InferenceAwait.await(stage, this::cancel));
  }

  /** Signals cancellation without interrupting a native call. */
  public void cancel() {
    cancellation.run();
  }
}
