package se.alipsa.jmlx.ffi;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Samples mlx-c's own active-memory counter ({@code mlx_get_active_memory}) -- the one way tests in this project
 * observe whether native memory was actually freed, rather than relying on GC/Cleaner timing as a proxy. Lives here, in
 * jmlx-ffi's testFixtures, rather than in jmlx-core: jmlx-core tests and jmlx-ffi tests both need it, and jmlx-ffi is
 * the lower module in that dependency (jmlx-core depends on jmlx-ffi, not the reverse). Throws
 * {@link IllegalStateException} rather than jmlx-core's {@code MLXException} on failure for the same reason -- that
 * type isn't visible from here.
 */
public final class NativeMemoryProbe {

  // Same reasoning as MLX's and MLXScope's own static initializers: this
  // class calls mlx_h.mlx_get_active_memory directly and is a public
  // fixture meant to be reached for from new tests, so it needs the guard
  // independently rather than relying on a caller having already touched
  // one of those classes first.
  static {
    NativeLoader.ensureLoaded();
  }

  private NativeMemoryProbe() {}

  /** Current active (not cached, not process RSS) native memory, in bytes, as reported by mlx-c. */
  public static long activeMemoryBytes() {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment out = tmp.allocate(ValueLayout.JAVA_LONG);
      NativeLoader.clearLastNativeError();
      int status = mlx_h.mlx_get_active_memory(out);
      if (status != 0) {
        String nativeMessage = NativeLoader.lastNativeError();
        NativeLoader.clearLastNativeError();
        throw new IllegalStateException(
            "mlx_get_active_memory failed with status " + status + (nativeMessage != null ? ": " + nativeMessage : ""));
      }
      return out.get(ValueLayout.JAVA_LONG, 0);
    }
  }
}
