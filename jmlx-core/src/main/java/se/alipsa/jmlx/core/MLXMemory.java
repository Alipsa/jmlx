package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.Function;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_h;

/**
 * Process-wide mlx-c allocator counters. Values include allocations from every thread and scope
 * using the same native runtime; concurrent work can change them between calls. Counters are
 * sampled synchronously but do not themselves evaluate pending lazy MLX graphs. Callers measuring a
 * phase must evaluate its outputs before sampling. On the Metal backend the counters can also be
 * momentarily high right after an {@code eval} returns. Verified against the pinned mlx-metal
 * 0.31.2 source: {@code gpu::eval} (mlx/backend/metal/eval.cpp) captures each kernel's input and
 * sibling buffers into the command buffer's completion handler, and the CPU-side wait ({@code
 * Event::wait}, mlx/backend/metal/event.cpp) is not ordered after that handler's teardown on
 * Metal's dispatch thread, so the corresponding frees can land microseconds to milliseconds later.
 * When a post-eval steady-state value is needed, sample {@code activeBytes()} repeatedly until the
 * value stops changing before asserting on it. Native errors throw {@link MLXException}.
 */
public final class MLXMemory {
  static {
    NativeLoader.ensureLoaded();
  }

  private MLXMemory() {}

  /**
   * Currently active native bytes, excluding allocator-cached free storage. A read taken
   * immediately after an {@code eval} may still include buffers whose owners have been released but
   * whose Metal completion-handler teardown has not run yet; see the class javadoc.
   */
  public static long activeBytes() {
    return read("mlx_get_active_memory", mlx_h::mlx_get_active_memory);
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
