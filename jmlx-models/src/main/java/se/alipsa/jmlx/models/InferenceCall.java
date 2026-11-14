package se.alipsa.jmlx.models;

import java.util.Objects;
import java.util.concurrent.CompletionStage;

/** A generation stage with explicit, idempotent request cancellation. */
public record InferenceCall(CompletionStage<GenerationResult> stage, Runnable cancellation) {
  /** Requires both the result and cancellation action. */
  public InferenceCall {
    Objects.requireNonNull(stage, "stage");
    Objects.requireNonNull(cancellation, "cancellation");
  }

  /** Signals cancellation without interrupting a native call. */
  public void cancel() {
    cancellation.run();
  }
}
