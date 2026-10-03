package se.alipsa.jmlx.models;

/**
 * {@link BatchGenerationScheduler#start} was called while another scheduler's worker has not
 * exited. MLX allows at most one MLX-using thread at a time per process; this guard enforces that
 * within one jmlx classloader.
 */
public final class SchedulerAlreadyRunningException extends IllegalStateException {
  private static final long serialVersionUID = 1L;

  SchedulerAlreadyRunningException() {
    super(
        "another batch scheduler's worker has not exited; close it before starting a new one"
            + " (at most one MLX-using thread at a time per process)");
  }
}
