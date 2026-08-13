package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Reuses {@code MLXFastTest}'s {@code rmsNormWithWeightScalesTheNormalizedResult} golden through
 * the layer.
 */
@EnabledIfNativeAvailable
class RMSNormTest {

  // Irrational sqrt, same reasoning as MLXFastTest's EPS.
  private static final float EPS = 1e-3f;

  @Test
  void forwardMatchesTheMLXFastRmsNormGolden() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {2, 2, 2, 2}, new int[] {4});
      RMSNorm rmsNorm = new RMSNorm(scope, weight, 1e-5f);
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});

      MLXArray result = rmsNorm.forward(x);

      assertArrayEquals(
          new float[] {0.730296f, 1.460592f, 2.190888f, 2.921184f}, result.toFloatArray(), EPS);
    }
  }
}
