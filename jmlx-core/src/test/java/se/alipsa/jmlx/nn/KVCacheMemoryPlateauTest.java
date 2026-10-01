package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Native allocator regression for a repeatedly trimmed single-head cache. */
@EnabledIfNativeAvailable
class KVCacheMemoryPlateauTest {
  @Test
  void slidingCacheActiveBytesPlateauAfterWarmup() {
    int width = 8192;
    int window = 4;
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope, KVCachePolicy.slidingWindow(window));
      long baseline = MLXMemory.activeBytes();
      MLXMemory.resetPeak();
      long first = 0;
      long second = 0;
      for (int i = 0; i < 48; i++) {
        try (MLXScope step = scope.newChild()) {
          MLXArray token = MLX.array(step, new float[width], new int[] {1, 1, 1, width});
          cache.append(token, token);
          cache.trimToLast(Math.min(cache.length(), window - 1));
          MLX.eval(cache.keys(), cache.values());
        }
        if (i == 24) {
          first = MLXMemory.activeBytes();
        }
        if (i == 47) {
          second = MLXMemory.activeBytes();
        }
      }
      long logical = 2L * (window - 1) * width * Float.BYTES;
      long allowance = 3 * logical + 8L * 1024 * 1024;
      assertTrue(second - baseline <= allowance, "retained active bytes exceeded plateau bound");
      assertTrue(second - first <= allowance, "retained active bytes trended upward");
      assertTrue(MLXMemory.peakBytes() >= second, "peak counter must include active bytes");
      assertTrue(MLXMemory.cachedBytes() >= 0, "allocator cached bytes are nonnegative");
    }
  }
}
