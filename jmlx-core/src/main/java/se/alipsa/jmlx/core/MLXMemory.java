package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_h;

/**
 * Process-wide mlx-c allocator counters. Values include allocations from every thread and scope
 * using the same native runtime; concurrent work can change them between calls. Counters are
 * sampled synchronously but do not themselves evaluate pending lazy MLX graphs. Callers measuring a
 * phase must evaluate its outputs before sampling. On the Metal backend the counters can also be
 * momentarily high right after an {@code eval} returns: the command buffer keeps the evaluated
 * graph's buffers alive until its completion handler is torn down on Metal's dispatch thread, and
 * that teardown is not ordered after the CPU-side wait, so the corresponding frees can land
 * microseconds to milliseconds later. Sample {@link #settledActiveBytes()} instead when a post-eval
 * steady-state value is needed. Native errors throw {@link MLXException}.
 */
public final class MLXMemory {
  static {
    NativeLoader.ensureLoaded();
  }

  private static final long SETTLE_INTERVAL_MS = 2;
  private static final long SETTLE_TAIL_MS = 16;
  private static final long SETTLE_TIMEOUT_MS = 1000;

  private MLXMemory() {}

  /**
   * Currently active native bytes, excluding allocator-cached free storage. A read taken
   * immediately after an {@code eval} may still include buffers whose owners have been released but
   * whose Metal completion-handler teardown has not run yet; see the class javadoc and {@link
   * #settledActiveBytes()}.
   */
  public static long activeBytes() {
    return read("mlx_get_active_memory", mlx_h::mlx_get_active_memory);
  }

  /**
   * Active native bytes at a quiescent point: polls {@link #activeBytes()} until it shows an
   * unchanging run of {@value #SETTLE_TAIL_MS} milliseconds, which means the in-flight
   * completion-handler frees described in the class javadoc have drained. Bounded: if the counter
   * is still moving {@value #SETTLE_TIMEOUT_MS} milliseconds after the first read, the last value
   * is returned anyway.
   */
  public static long settledActiveBytes() throws InterruptedException {
    long previous = activeBytes();
    long lastChange = System.nanoTime();
    long deadline = lastChange + TimeUnit.MILLISECONDS.toNanos(SETTLE_TIMEOUT_MS);
    while (System.nanoTime() < deadline) {
      Thread.sleep(SETTLE_INTERVAL_MS);
      long current = activeBytes();
      if (current != previous) {
        previous = current;
        lastChange = System.nanoTime();
      } else if (System.nanoTime() - lastChange >= TimeUnit.MILLISECONDS.toNanos(SETTLE_TAIL_MS)) {
        return current;
      }
    }
    return activeBytes();
  }

  /** Peak active native bytes since the last successful {@link #resetPeak()}. */
  public static long peakBytes() {
    return read("mlx_get_peak_memory", mlx_h::mlx_get_peak_memory);
  }

  /** Native bytes cached by the allocator for reuse, distinct from active bytes. */
  public static long cachedBytes() {
    return read("mlx_get_cache_memory", mlx_h::mlx_get_cache_memory);
  }

  /** Resets the process-wide peak counter; concurrent allocations may immediately raise it. */
  public static void resetPeak() {
    NativeOps.checked("mlx_reset_peak_memory", mlx_h::mlx_reset_peak_memory);
  }

  private static long read(String operation, Function<MemorySegment, Integer> call) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment result = arena.allocate(ValueLayout.JAVA_LONG);
      NativeOps.checked(operation, () -> call.apply(result));
      return result.get(ValueLayout.JAVA_LONG, 0);
    }
  }
}
