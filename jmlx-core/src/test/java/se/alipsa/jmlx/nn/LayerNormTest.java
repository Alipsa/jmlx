package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Reuses {@code MLXFastTest}'s {@code layerNormWithWeightAndBiasScalesAndShiftsTheNormalizedResult}
 * golden through the layer.
 */
@EnabledIfNativeAvailable
class LayerNormTest {

  // Irrational sqrt, same reasoning as MLXFastTest's EPS.
  private static final float EPS = 1e-3f;

  @Test
  void forwardMatchesTheMLXFastLayerNormGolden() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {2, 2, 2, 2}, new int[] {4});
      MLXArray bias = MLX.array(scope, new float[] {1, 1, 1, 1}, new int[] {4});
      LayerNorm layerNorm = new LayerNorm(scope, weight, bias, 1e-5f);
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});

      MLXArray result = layerNorm.forward(x);

      assertArrayEquals(
          new float[] {-1.683282f, 0.105573f, 1.894427f, 3.683282f}, result.toFloatArray(), EPS);
    }
  }
}
