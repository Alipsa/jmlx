package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
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

  /**
   * Regression test for the classic GQA head-repetition bug: a query head must map to key/value
   * head {@code queryHead / queriesPerKeyValueHead} (contiguous groups, matching Hugging Face's
   * {@code repeat_kv}), not {@code queryHead % numKeyValueHeads} (interleaved). With a single-token
   * sequence, softmax over one key is always 1, so attention output equals the per-head value bias
   * exactly regardless of query/key content -- letting this test hand-compute the expected output
   * without modeling softmax at all.
   */
  @Test
  void repeatsKeyValueHeadsContiguouslyNotInterleaved() {
    try (MLXScope scope = new MLXScope()) {
      int numHeads = 4;
      int numKeyValueHeads = 2;
      int headDim = 2;
      int embedDim = numHeads * headDim;
      int keyValueDim = numKeyValueHeads * headDim;
      MLXArray queryWeight = MLX.zeros(scope, new int[] {embedDim, embedDim}, DType.FLOAT32);
      MLXArray keyWeight = MLX.zeros(scope, new int[] {keyValueDim, embedDim}, DType.FLOAT32);
      MLXArray valueWeight = MLX.zeros(scope, new int[] {keyValueDim, embedDim}, DType.FLOAT32);
      // kv head 0 -> [1, 2], kv head 1 -> [3, 4]; zero value weight makes this the value output
      // for every position, regardless of x.
      MLXArray valueBias = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {keyValueDim});
      MLXArray outWeight = MLX.array(scope, identity(embedDim), new int[] {embedDim, embedDim});
      GroupedQueryAttention attention =
          new GroupedQueryAttention(
              scope,
              numHeads,
              numKeyValueHeads,
              10000f,
              queryWeight,
              null,
              keyWeight,
              null,
              valueWeight,
              valueBias,
              outWeight,
              null);
      MLXArray x = MLX.zeros(scope, new int[] {1, 1, embedDim}, DType.FLOAT32);
      MLXArray result = attention.forward(x, null);
      assertArrayEquals(
          new float[] {1, 2, 1, 2, 3, 4, 3, 4},
          result.toFloatArray(),
          "query heads 0,1 -> kv head 0; query heads 2,3 -> kv head 1");
    }
  }

  private static float[] identity(int n) {
    float[] matrix = new float[n * n];
    for (int i = 0; i < n; i++) {
      matrix[i * n + i] = 1f;
    }
    return matrix;
  }

  private static GroupedQueryAttention attention(MLXScope scope) {
    MLXArray queryWeight = MLX.zeros(scope, new int[] {4, 4}, DType.FLOAT32);
    MLXArray keyWeight = MLX.zeros(scope, new int[] {2, 4}, DType.FLOAT32);
    MLXArray valueWeight = MLX.zeros(scope, new int[] {2, 4}, DType.FLOAT32);
    MLXArray outWeight = MLX.zeros(scope, new int[] {4, 4}, DType.FLOAT32);
    return new GroupedQueryAttention(
        scope,
        2,
        1,
        10000f,
        queryWeight,
        null,
        keyWeight,
        null,
        valueWeight,
        null,
        outWeight,
        null);
  }
}
