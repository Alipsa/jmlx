package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Pins the single-row batched arithmetic that the WP4 rotation path relies on (plan §6, Unit 2):
 * for a uniform batch row, {@code cache.append(k, v, {T})} is the same arithmetic as {@code
 * cache.append(k, v)} -- bit-identical retained tensors -- and the B = 1 {@link
 * AttentionMask#batched} mask is elementwise equal to the 2-D {@link AttentionMask#slidingWindow}
 * mask. Both facts let the single-row rotation path route through the batch-aware overloads without
 * changing the old graph's results.
 */
@EnabledIfNativeAvailable
class SingleRowDecodeEquivalenceTest {

  @Test
  void uniformSingleRowAppendIsBitIdenticalToPlainAppend() {
    try (MLXScope scope = new MLXScope()) {
      KVCache plain = new KVCache(scope);
      KVCache batched = new KVCache(scope);
      MLXArray k1 = MLX.array(scope, values(1 * 2 * 4 * 8, 0), new int[] {1, 2, 4, 8});
      MLXArray v1 = MLX.array(scope, values(1 * 2 * 4 * 8, 5), new int[] {1, 2, 4, 8});
      plain.append(k1, v1);
      batched.append(k1, v1, new int[] {4});
      assertArrayEquals(plain.keys().toFloatArray(), batched.keys().toFloatArray());
      assertArrayEquals(plain.values().toFloatArray(), batched.values().toFloatArray());
      assertEquals(plain.offset(), batched.offset());
      assertEquals(plain.nextPosition(0), batched.nextPosition(0));

      MLXArray k2 = MLX.array(scope, values(1 * 2 * 3 * 8, 1), new int[] {1, 2, 3, 8});
      MLXArray v2 = MLX.array(scope, values(1 * 2 * 3 * 8, 6), new int[] {1, 2, 3, 8});
      plain.append(k2, v2);
      batched.append(k2, v2, new int[] {3});
      assertArrayEquals(plain.keys().toFloatArray(), batched.keys().toFloatArray());
      assertArrayEquals(plain.values().toFloatArray(), batched.values().toFloatArray());
      assertEquals(plain.offset(), batched.offset());
      assertEquals(plain.nextPosition(0), batched.nextPosition(0));

      MLXArray k3 = MLX.array(scope, values(1 * 2 * 1 * 8, 2), new int[] {1, 2, 1, 8});
      MLXArray v3 = MLX.array(scope, values(1 * 2 * 1 * 8, 7), new int[] {1, 2, 1, 8});
      plain.append(k3, v3);
      batched.append(k3, v3, new int[] {1});
      assertArrayEquals(plain.keys().toFloatArray(), batched.keys().toFloatArray());
      assertArrayEquals(plain.values().toFloatArray(), batched.values().toFloatArray());
      assertEquals(plain.offset(), batched.offset());
      assertEquals(plain.nextPosition(0), batched.nextPosition(0));
    }
  }

  @Test
  void singleRowBatchedMaskMatchesTwoDimensionalMask() {
    try (MLXScope scope = new MLXScope()) {
      // Prefill rows have T == S (no key padding), so the batched row is unpaddded and
      // elementwise equal to the 2-D mask; decode rows are the queryStart = S - 1 case below.
      for (int[] shape : new int[][] {{4, 4}, {90, 90}, {7, 7}}) {
        int queryLength = shape[0];
        int keyLength = shape[1];
        MLXArray batched =
            AttentionMask.batched(
                scope,
                new int[] {0},
                new int[] {0},
                new int[] {queryLength},
                queryLength,
                keyLength,
                0);
        MLXArray twoD = AttentionMask.slidingWindow(scope, 0, 0, queryLength, keyLength, keyLength);
        assertArrayEquals(new int[] {1, 1, queryLength, keyLength}, batched.shape());
        assertArrayEquals(asFloat(batched), asFloat(twoD));
      }
      for (int keyLength : new int[] {4, 90, 7}) {
        MLXArray batched =
            AttentionMask.batched(
                scope, new int[] {keyLength - 1}, new int[] {0}, new int[] {1}, 1, keyLength, 0);
        MLXArray twoD =
            AttentionMask.slidingWindow(scope, keyLength - 1, 0, 1, keyLength, keyLength);
        assertArrayEquals(asFloat(batched), asFloat(twoD));
      }
    }
  }

  private static float[] asFloat(MLXArray boolMask) {
    return MLX.astype(boolMask, DType.FLOAT32).toFloatArray();
  }

  private static float[] values(int n, int seed) {
    float[] data = new float[n];
    for (int j = 0; j < n; j++) {
      data[j] = (((j + seed) % 13) - 6) / 6.0f;
    }
    return data;
  }
}
