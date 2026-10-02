package se.alipsa.jmlx.models;

/**
 * The scheduler could not start: the model factory threw a checked exception, or returned something
 * that is not a usable {@code DecoderModel} in the worker's root scope. Unchecked factory failures
 * propagate unchanged.
 */
public final class SchedulerStartException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  SchedulerStartException(String message, Throwable cause) {
    super(message, cause);
  }

  SchedulerStartException(String message) {
    super(message);
  }
}
