package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class BatchAttentionMaskTest {
  @Test
  void masksLeftPaddingAndUsesEachRowsAbsolutePositions() {
    try (MLXScope scope = new MLXScope()) {
      assertArrayEquals(
          new float[] {
            0, 0, 1, 0, 1, 1,
            1, 1, 0, 0, 1, 1
          },
          MLX.astype(
                  AttentionMask.batched(
                      scope, new int[] {2, 1}, new int[] {0, 0}, new int[] {1, 2}, 2, 3, 2),
                  DType.FLOAT32)
              .toFloatArray());
    }
  }

  @Test
  void batchedBoolMaskIsAcceptedByNativeSdpa() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray q = MLX.ones(scope, new int[] {2, 1, 2, 2}, DType.FLOAT32);
      MLXArray k = MLX.ones(scope, new int[] {2, 1, 3, 2}, DType.FLOAT32);
      MLXArray v = MLX.ones(scope, new int[] {2, 1, 3, 2}, DType.FLOAT32);
      MLXArray mask =
          AttentionMask.batched(
              scope, new int[] {2, 1}, new int[] {0, 0}, new int[] {1, 2}, 2, 3, 2);
      MLXArray result = MLXFast.scaledDotProductAttention(q, k, v, 1f, false, mask, null);
      assertArrayEquals(new float[] {1, 1, 1, 1, 1, 1, 1, 1}, result.toFloatArray(), 1e-5f);
    }
  }
}
