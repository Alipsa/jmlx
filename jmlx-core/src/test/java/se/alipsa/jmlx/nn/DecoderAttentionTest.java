package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class DecoderAttentionTest {

  @Test
  void fullAttentionMatchesGroupedQueryAndParameterNames() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray q = MLX.array(scope, identity(4), new int[] {4, 4});
      MLXArray k = MLX.array(scope, new float[] {1, 0, 0, 0, 0, 1, 0, 0}, new int[] {2, 4});
      MLXArray v = MLX.array(scope, new float[] {0, 0, 1, 0, 0, 0, 0, 1}, new int[] {2, 4});
      MLXArray out = MLX.array(scope, identity(4), new int[] {4, 4});
      GroupedQueryAttention original =
          new GroupedQueryAttention(scope, 2, 1, 10000f, q, null, k, null, v, null, out, null);
      DecoderAttention configurable =
          new DecoderAttention(
              scope,
              2,
              1,
              2,
              new RopeSpec.Base(10000f),
              2,
              null,
              null,
              new Linear(scope, q, null),
              new Linear(scope, k, null),
              new Linear(scope, v, null),
              new Linear(scope, out, null));
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4, 2, 1, 4, 3}, new int[] {1, 2, 4});
      assertArrayEquals(
          original.forward(x, null).toFloatArray(),
          configurable.forward(x, null).toFloatArray(),
          1e-5f);
      assertEquals(original.parameters().keySet(), configurable.parameters().keySet());
    }
  }

  @Test
  void explicitHeadDimAndWindowPreflight() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention =
          new DecoderAttention(
              scope,
              2,
              1,
              4,
              new RopeSpec.Base(10000f),
              4,
              null,
              2,
              new Linear(scope, MLX.zeros(scope, new int[] {8, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 8}, DType.FLOAT32), null));
      KVCache cache = new KVCache(scope);
      MLXArray prompt = MLX.ones(scope, new int[] {1, 3, 4}, DType.FLOAT32);
      IllegalArgumentException error =
          assertThrows(IllegalArgumentException.class, () -> attention.forward(prompt, cache));
      assertTrue(error.getMessage().contains("slidingWindow"));
      assertEquals(0, cache.offset());
      MLXArray mask = AttentionMask.slidingWindow(scope, 3, 3, 2);
      assertArrayEquals(new int[] {1, 3, 4}, attention.forward(prompt, cache, mask).shape());
      assertEquals(3, cache.offset());
    }
  }

  @Test
  void explicitHeadDimensionCanDifferFromHiddenPerHeadWidth() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention =
          new DecoderAttention(
              scope,
              2,
              1,
              3,
              new RopeSpec.Base(10000f),
              2,
              null,
              null,
              new Linear(scope, MLX.zeros(scope, new int[] {6, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {3, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {3, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 6}, DType.FLOAT32), null));
      MLXArray x = MLX.ones(scope, new int[] {1, 2, 4}, DType.FLOAT32);
      assertArrayEquals(new int[] {1, 2, 4}, attention.forward(x, null).shape());
    }
  }

  @Test
  void windowedLastRowMatchesDirectAttentionAtPrefillAndDecodeOffsets() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention =
          new DecoderAttention(
              scope,
              1,
              1,
              2,
              new RopeSpec.Base(10000f),
              2,
              null,
              2,
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null));
      MLXArray queries =
          MLX.array(
              scope, new float[] {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, new int[] {1, 1, 6, 2});
      MLXArray keys =
          MLX.array(
              scope, new float[] {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}, new int[] {1, 1, 6, 2});
      MLXArray values =
          MLX.array(
              scope, new float[] {0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8}, new int[] {1, 1, 6, 2});
      RopeSpec rope = new RopeSpec.Base(10000f);
      queries = rope.apply(queries, 2, 0, null);
      keys = rope.apply(keys, 2, 0, null);
      MLXArray lastQuery = MLXShape.slice(queries, new int[] {0, 0, 5, 0}, new int[] {1, 1, 6, 2});
      MLXArray lastKeys = MLXShape.slice(keys, new int[] {0, 0, 4, 0}, new int[] {1, 1, 6, 2});
      MLXArray lastValues = MLXShape.slice(values, new int[] {0, 0, 4, 0}, new int[] {1, 1, 6, 2});
      MLXArray direct =
          MLXFast.scaledDotProductAttention(
              lastQuery, lastKeys, lastValues, (float) (1 / Math.sqrt(2)), false, null, null);
      MLXArray prefill = attention.attend(queries, keys, values, 0);
      MLXArray prefillLast =
          MLXShape.slice(prefill, new int[] {0, 0, 5, 0}, new int[] {1, 1, 6, 2});
      assertArrayEquals(direct.toFloatArray(), prefillLast.toFloatArray(), 1e-5f);
      assertArrayEquals(
          direct.toFloatArray(),
          attention.attend(lastQuery, keys, values, 5).toFloatArray(),
          1e-5f);
    }
  }

  private static float[] identity(int size) {
    float[] data = new float[size * size];
    for (int i = 0; i < size; i++) {
      data[i * size + i] = 1;
    }
    return data;
  }
}
