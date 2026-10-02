package se.alipsa.jmlx.models;

/**
 * The scheduler's worker failed and can no longer serve requests. Every accepted request's stage
 * completes with a {@link GenerationAbortedException} whose cause is this exception, and a later
 * {@code submit} throws it directly. {@link #getCause()} is the original failure.
 */
public final class SchedulerFailedException extends IllegalStateException {
  private static final long serialVersionUID = 1L;

  SchedulerFailedException(Throwable cause) {
    super("batch scheduler worker failed: " + cause, cause);
  }
}
