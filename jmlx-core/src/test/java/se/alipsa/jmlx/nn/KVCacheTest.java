package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class KVCacheTest {

  private static final float EPS = 1e-5f;

  @Test
  void firstAppendHoistsDirectlyAndAdvancesOffset() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      MLXArray k = MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 1, 2});
      MLXArray v = MLX.array(scope, new float[] {3, 4}, new int[] {1, 1, 1, 2});
      cache.append(k, v);
      assertEquals(1, cache.offset());
      assertArrayEquals(new float[] {1, 2}, cache.keys().toFloatArray(), EPS);
      assertArrayEquals(new float[] {3, 4}, cache.values().toFloatArray(), EPS);
    }
  }

  @Test
  void secondAppendConcatenatesAlongTheSequenceAxisAndAdvancesOffset() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      cache.append(
          MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 1, 2}),
          MLX.array(scope, new float[] {3, 4}, new int[] {1, 1, 1, 2}));
      cache.append(
          MLX.array(scope, new float[] {5, 6}, new int[] {1, 1, 1, 2}),
          MLX.array(scope, new float[] {7, 8}, new int[] {1, 1, 1, 2}));
      assertEquals(2, cache.offset());
      assertArrayEquals(new int[] {1, 1, 2, 2}, cache.keys().shape());
      assertArrayEquals(new float[] {1, 2, 5, 6}, cache.keys().toFloatArray(), EPS);
      assertArrayEquals(new float[] {3, 4, 7, 8}, cache.values().toFloatArray(), EPS);
    }
  }

  /**
   * A step scope closing after {@code append} must not invalidate the cache's own copy -- proves
   * the hoist-then-close discipline in {@link KVCache#append}'s javadoc actually decouples the two.
   */
  @Test
  void cacheSurvivesTheStepScopeClosingAfterAppend() {
    try (MLXScope modelScope = new MLXScope()) {
      KVCache cache = new KVCache(modelScope);
      try (MLXScope step = modelScope.newChild()) {
        cache.append(
            MLX.array(step, new float[] {1, 2}, new int[] {1, 1, 1, 2}),
            MLX.array(step, new float[] {3, 4}, new int[] {1, 1, 1, 2}));
      }
      try (MLXScope step2 = modelScope.newChild()) {
        cache.append(
            MLX.array(step2, new float[] {5, 6}, new int[] {1, 1, 1, 2}),
            MLX.array(step2, new float[] {7, 8}, new int[] {1, 1, 1, 2}));
      }
      assertArrayEquals(new float[] {1, 2, 5, 6}, cache.keys().toFloatArray(), EPS);
    }
  }

  @Test
  void appendFromAnUnrelatedScopeThrows() {
    try (MLXScope cacheScope = new MLXScope();
        MLXScope unrelated = new MLXScope()) {
      KVCache cache = new KVCache(cacheScope);
      MLXArray k = MLX.array(unrelated, new float[] {1, 2}, new int[] {1, 1, 1, 2});
      MLXArray v = MLX.array(unrelated, new float[] {3, 4}, new int[] {1, 1, 1, 2});
      assertThrows(IllegalArgumentException.class, () -> cache.append(k, v));
    }
  }

  /**
   * The memory-growth hazard {@link KVCache}'s own javadoc names: without the {@code close()} calls
   * in {@code append}, active memory after N appends would sum every superseded generation --
   * roughly {@code N/2} times larger than the correctly-freed figure. A generous-but-
   * discriminating multiple over the correct (linear) figure catches that shape without tripping on
   * ordinary allocator overhead.
   */
  @Test
  void activeMemoryGrowsLinearlyNotQuadraticallyAcrossManyAppends() {
    int elementsPerToken = 50_000; // ~200 KB per tensor per token (float32)
    int appends = 30;
    try (MLXScope modelScope = new MLXScope()) {
      KVCache cache = new KVCache(modelScope);
      long baseline = NativeMemoryProbe.activeMemoryBytes();
      for (int i = 0; i < appends; i++) {
        try (MLXScope step = modelScope.newChild()) {
          MLXArray k =
              MLX.array(step, new float[elementsPerToken], new int[] {1, 1, 1, elementsPerToken});
          MLXArray v =
              MLX.array(step, new float[elementsPerToken], new int[] {1, 1, 1, elementsPerToken});
          cache.append(k, v);
          MLX.eval(cache.keys(), cache.values());
        }
      }
      long after = NativeMemoryProbe.activeMemoryBytes();
      long grew = after - baseline;
      long expectedLinear = 2L * appends * elementsPerToken * 4;
      assertTrue(
          grew <= expectedLinear * 4,
          "active memory grew by "
              + grew
              + " bytes over "
              + appends
              + " appends (expected roughly "
              + expectedLinear
              + " bytes for correctly-freed superseded generations -- this suggests"
              + " KVCache.append is not closing the previous keys/values handle after hoisting)");
    }
  }
}
