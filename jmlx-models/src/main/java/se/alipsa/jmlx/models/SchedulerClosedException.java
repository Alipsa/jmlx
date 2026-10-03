package se.alipsa.jmlx.models;

/** A request was submitted to a scheduler that is closing or closed. */
public final class SchedulerClosedException extends IllegalStateException {
  private static final long serialVersionUID = 1L;

  SchedulerClosedException(String message) {
    super(message);
  }
}
