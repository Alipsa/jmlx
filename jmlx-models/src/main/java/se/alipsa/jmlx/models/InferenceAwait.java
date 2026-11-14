package se.alipsa.jmlx.models;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;

/** Common interruption, exception and cancellation policy for blocking framework calls. */
final class InferenceAwait {
  private InferenceAwait() {}

  static <T> T await(CompletionStage<T> stage, Runnable cancel) {
    try {
      return stage.toCompletableFuture().get();
    } catch (InterruptedException interrupted) {
      cancel.run();
      Thread.currentThread().interrupt();
      throw new CancellationException("inference interrupted");
    } catch (ExecutionException failure) {
      Throwable cause = SchedulerInferenceClient.unwrap(failure.getCause());
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IllegalStateException("inference failed", cause);
    }
  }

  static GenerationResult completed(GenerationResult result) {
    if (result.finishReason() == FinishReason.CANCELLED) {
      throw new CancellationException("generation cancelled");
    }
    return result;
  }
}
