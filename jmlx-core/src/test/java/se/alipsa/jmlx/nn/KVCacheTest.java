package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.core.DType;
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

  /**
   * {@code k}/{@code v} may independently be aliased-in-place or hoisted-as-a-copy on the first
   * {@code append} -- {@code ownsKeys}/{@code ownsValues} must be tracked separately, not derived
   * from {@code keys} alone. Here {@code k0} lives in a descendant scope (hoisted, owned) while
   * {@code v0} lives in the cache's own scope (aliased, not owned). Deriving both flags from {@code
   * keys} would wrongly close {@code v0} -- the caller's own array -- on the next append.
   */
  @Test
  void appendTracksKeyAndValueOwnershipIndependently() {
    try (MLXScope cacheScope = new MLXScope()) {
      KVCache cache = new KVCache(cacheScope);
      MLXArray v0 = MLX.array(cacheScope, new float[] {3, 4}, new int[] {1, 1, 1, 2});
      try (MLXScope step = cacheScope.newChild()) {
        MLXArray k0 = MLX.array(step, new float[] {1, 2}, new int[] {1, 1, 1, 2});
        cache.append(k0, v0);
      }
      try (MLXScope step2 = cacheScope.newChild()) {
        MLXArray k1 = MLX.array(step2, new float[] {5, 6}, new int[] {1, 1, 1, 2});
        MLXArray v1 = MLX.array(step2, new float[] {7, 8}, new int[] {1, 1, 1, 2});
        cache.append(k1, v1);
      }
      // v0 must still be open -- the caller's own array, never owned by the cache.
      assertArrayEquals(new float[] {3, 4}, v0.toFloatArray(), EPS);
      assertArrayEquals(new float[] {3, 4, 7, 8}, cache.values().toFloatArray(), EPS);
    }
  }

  @Test
  void firstAppendRejectsKAndVWithMismatchedLeadingBatchAxisShapes() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      // Axis 1 (numHeads) differs: 2 vs 3 -- a leading (batch) axis, not the sequence or last axis.
      MLXArray k = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {1, 2, 1, 2});
      MLXArray v = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {1, 3, 1, 2});
      assertThrows(IllegalArgumentException.class, () -> cache.append(k, v));
    }
  }

  /**
   * The container only needs k/v to agree on rank and sequence length -- the two tensors are
   * concatenated independently, so their last axis (head dim) is free to differ, e.g. for MLA-style
   * attention where V's head dim differs from Q/K's.
   */
  @Test
  void firstAppendAllowsKAndVWithDifferentHeadDims() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      MLXArray k = MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 1, 2});
      MLXArray v = MLX.array(scope, new float[] {3, 4, 5}, new int[] {1, 1, 1, 3});
      cache.append(k, v);
      assertEquals(1, cache.offset());
      assertArrayEquals(new float[] {1, 2}, cache.keys().toFloatArray(), EPS);
      assertArrayEquals(new float[] {3, 4, 5}, cache.values().toFloatArray(), EPS);
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

  @Test
  void boundedFullRejectsBeforeChangingCacheAndResetAllowsReuse() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope, KVCachePolicy.full(1));
      MLXArray k = MLX.array(scope, new float[] {1}, new int[] {1, 1, 1, 1});
      cache.append(k, k);
      assertThrows(IllegalArgumentException.class, () -> cache.append(k, k));
      assertEquals(1, cache.offset());
      assertArrayEquals(new float[] {1}, cache.keys().toFloatArray(), EPS);
      cache.poison();
      assertThrows(IllegalStateException.class, () -> cache.append(k, k));
      cache.reset();
      assertEquals(0, cache.offset());
      cache.append(k, k);
      assertEquals(1, cache.offset());
    }
  }

  @Test
  void slidingTrimPreservesAbsolutePosition() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope, KVCachePolicy.slidingWindow(2));
      MLXArray k = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 1, 3, 1});
      cache.append(k, k);
      cache.trimToLast(1);
      assertEquals(3, cache.nextPosition());
      assertEquals(2, cache.startPosition());
      assertEquals(1, cache.length());
      assertArrayEquals(new float[] {3}, cache.keys().toFloatArray(), EPS);
      cache.append(
          MLX.array(scope, new float[] {4}, new int[] {1, 1, 1, 1}),
          MLX.array(scope, new float[] {4}, new int[] {1, 1, 1, 1}));
      assertArrayEquals(new float[] {3, 4}, cache.keys().toFloatArray(), EPS);
    }
  }

  @Test
  void reorderMaterializesIndependentRowsBeforeReturning() {
    try (MLXScope destination = new MLXScope()) {
      KVCache reordered;
      try (MLXScope source = destination.newChild()) {
        KVCache cache = new KVCache(source);
        MLXArray data = MLX.array(source, new float[] {1, 2}, new int[] {2, 1, 1, 1});
        cache.append(data, data);
        reordered = cache.reorder(new int[] {1, 1, 0}, destination);
        assertThrows(
            IllegalArgumentException.class, () -> cache.reorder(new int[] {2}, destination));
      }
      assertArrayEquals(new float[] {2, 2, 1}, reordered.keys().toFloatArray(), EPS);
    }
  }

  @Test
  void reorderShortRaggedRowRestoresItsPaddedWidthAndPosition() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      MLXArray data = MLX.array(scope, new float[] {0, 7, 2, 3}, new int[] {2, 1, 2, 1});
      cache.append(data, data, new int[] {1, 2});
      KVCache selected = cache.reorder(new int[] {0}, scope);
      assertEquals(1, selected.offset());
      assertEquals(1, selected.length());
      assertArrayEquals(new float[] {7}, selected.keys().toFloatArray(), EPS);
    }
  }

  @Test
  void forkIntoSiblingScopeSurvivesSourceClose() {
    try (MLXScope parent = new MLXScope();
        MLXScope destination = parent.newChild()) {
      KVCache copy;
      try (MLXScope source = parent.newChild()) {
        KVCache cache = new KVCache(source);
        MLXArray data = MLX.array(source, new float[] {5}, new int[] {1, 1, 1, 1});
        cache.append(data, data);
        copy = cache.fork(destination);
      }
      assertArrayEquals(new float[] {5}, copy.keys().toFloatArray(), EPS);
    }
  }

  /**
   * Polls rather than sampling once: MLX can release the freed source buffer slightly after {@code
   * eval} returns, so under full-suite load a single sample occasionally still counted the two-row
   * source (observed: 6,553,604 bytes, settling to 3,276,800 within milliseconds). A cache that
   * really retains the source never settles, so the bounded poll still fails it.
   */
  @Test
  @Timeout(value = 30, unit = TimeUnit.SECONDS)
  void reorderedSubsetDoesNotRetainFullSourceBuffer() throws InterruptedException {
    int width = 8192;
    int positions = 50;
    try (MLXScope parent = new MLXScope();
        MLXScope destination = parent.newChild()) {
      long baseline = NativeMemoryProbe.activeMemoryBytes();
      KVCache selected;
      try (MLXScope source = parent.newChild()) {
        KVCache cache = new KVCache(source);
        float[] contents = new float[2 * positions * width];
        MLXArray data = MLX.array(source, contents, new int[] {2, 1, positions, width});
        cache.append(data, data);
        selected = cache.reorder(new int[] {1}, destination);
      }
      MLX.eval(selected.keys(), selected.values());
      long oneRowBytes = 2L * positions * width * Float.BYTES;
      long limit = oneRowBytes + 1024 * 1024;
      // Bounded below the @Timeout so a real retention fails with the measured numbers.
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
      long growth = NativeMemoryProbe.activeMemoryBytes() - baseline;
      while (growth > limit && System.nanoTime() < deadline) {
        Thread.sleep(10);
        growth = NativeMemoryProbe.activeMemoryBytes() - baseline;
      }
      assertTrue(
          growth <= limit,
          "reordered cache retained the full two-row source backing allocation (growth="
              + growth
              + ", limit="
              + limit
              + ")");
    }
  }

  @Test
  void forkOfRankTwoCacheCopiesAlongSequenceAxis() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      cache.append(
          MLX.array(scope, new float[] {1, 2}, new int[] {2, 1}),
          MLX.array(scope, new float[] {3, 4}, new int[] {2, 1}));
      cache.append(
          MLX.array(scope, new float[] {5}, new int[] {1, 1}),
          MLX.array(scope, new float[] {6}, new int[] {1, 1}));
      KVCache copy = cache.fork(scope);
      assertArrayEquals(new float[] {1, 2, 5}, copy.keys().toFloatArray(), EPS);
      assertArrayEquals(new float[] {3, 4, 6}, copy.values().toFloatArray(), EPS);
      assertEquals(3, copy.offset());
    }
  }

  @Test
  void appendRejectsZeroRowBatchBeforeInstallingMetadata() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      MLXArray empty = MLX.zeros(scope, new int[] {0, 1, 1, 1}, DType.FLOAT32);
      assertThrows(IllegalArgumentException.class, () -> cache.append(empty, empty));
      assertEquals(0, cache.batchSize());
      assertEquals(0, cache.offset());
    }
  }

  @Test
  void raggedAppendRejectsPoisonedCache() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      MLXArray data = MLX.array(scope, new float[] {0, 7, 2, 3}, new int[] {2, 1, 2, 1});
      cache.append(data, data, new int[] {1, 2});
      cache.poison();
      assertThrows(IllegalStateException.class, () -> cache.append(data, data, new int[] {1, 2}));
    }
  }
}
