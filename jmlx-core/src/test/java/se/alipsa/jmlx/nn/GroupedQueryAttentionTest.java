package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class GroupedQueryAttentionTest {

  @Test
  void retainsCompactKeyValueHeadsWhileDecoding() {
    try (MLXScope modelScope = new MLXScope()) {
      GroupedQueryAttention attention = attention(modelScope);
      KVCache cache = new KVCache(modelScope);
      try (MLXScope step = modelScope.newChild()) {
        MLXArray first = MLX.array(step, new float[] {1, 2, 3, 4, 5, 6, 7, 8}, new int[] {1, 2, 4});
        MLXArray result = attention.forward(first, cache);
        assertArrayEquals(new int[] {1, 2, 4}, result.shape());
        assertArrayEquals(new float[8], result.toFloatArray());
        assertEquals(2, cache.offset());
        assertArrayEquals(new int[] {1, 1, 2, 2}, cache.keys().shape());
        assertArrayEquals(new int[] {1, 1, 2, 2}, cache.values().shape());
      }
      try (MLXScope step = modelScope.newChild()) {
        MLXArray next = MLX.array(step, new float[] {1, 2, 3, 4}, new int[] {1, 1, 4});
        MLXArray result = attention.forward(next, cache);
        assertArrayEquals(new int[] {1, 1, 4}, result.shape());
        assertEquals(3, cache.offset());
        assertArrayEquals(new int[] {1, 1, 3, 2}, cache.keys().shape());
      }
    }
  }

  private static GroupedQueryAttention attention(MLXScope scope) {
    MLXArray qWeight = MLX.zeros(scope, new int[] {4, 4}, se.alipsa.jmlx.core.DType.FLOAT32);
    MLXArray kWeight = MLX.zeros(scope, new int[] {2, 4}, se.alipsa.jmlx.core.DType.FLOAT32);
    MLXArray vWeight = MLX.zeros(scope, new int[] {2, 4}, se.alipsa.jmlx.core.DType.FLOAT32);
    MLXArray oWeight = MLX.zeros(scope, new int[] {4, 4}, se.alipsa.jmlx.core.DType.FLOAT32);
    return new GroupedQueryAttention(
        scope, 2, 1, 10000f, qWeight, null, kWeight, null, vWeight, null, oWeight, null);
  }
}
