package se.alipsa.jmlx.models;

import java.util.Objects;
import java.util.concurrent.CompletionStage;

/** An ordered embedding batch with cancellation at worker-operation boundaries. */
public record EmbeddingCall(CompletionStage<EmbeddingBatch> stage, Runnable cancellation) {
  /** Requires both the result and cancellation action. */
  public EmbeddingCall {
    Objects.requireNonNull(stage, "stage");
    Objects.requireNonNull(cancellation, "cancellation");
  }

  /**
   * Cancels queued or remaining texts; a native forward already in progress is allowed to finish.
   */
  public void cancel() {
    cancellation.run();
  }

  /** Waits interruptibly, cancels on interruption and propagates the underlying runtime failure. */
  public EmbeddingBatch await() {
    return InferenceAwait.await(stage, this::cancel);
  }
}
