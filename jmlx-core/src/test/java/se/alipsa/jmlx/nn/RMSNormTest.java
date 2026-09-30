package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
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

  @Test
  void offsetWeightKeepsFloat32InActivationScope() {
    try (MLXScope model = new MLXScope();
        MLXScope activation = model.newChild()) {
      MLXArray weight =
          MLX.astype(
              MLX.array(model, new float[] {0.003f, -0.004f}, new int[] {2}), DType.BFLOAT16);
      RMSNorm norm = new RMSNorm(model, weight, 1e-6f, true);
      MLXArray x = MLX.array(activation, new float[] {1, 2}, new int[] {1, 2});
      MLXArray offset = norm.offsetWeight(x);
      assertEquals(DType.FLOAT32, offset.dtype());
      assertEquals(activation, offset.scope());
      float[] actual = offset.toFloatArray();
      assertEquals(1f + weight.toFloatArray()[0], actual[0], 1e-6f);
      float rounded = MLX.astype(offset, DType.BFLOAT16).toFloatArray()[0];
      assertTrue(Math.abs(actual[0] - rounded) > 1e-4f);
    }
  }

  @Test
  void offsetWeightRemainsDifferentiableAfterRebind() {
    try (MLXScope model = new MLXScope()) {
      RMSNorm norm =
          new RMSNorm(
              model, MLX.array(model, new float[] {0.1f, -0.2f, 0.3f}, new int[] {3}), 1e-6f, true);
      java.util.LinkedHashMap<String, MLXArray> rebound = new java.util.LinkedHashMap<>();
      rebound.put("weight", MLX.array(model, new float[] {0.2f, 0.1f, -0.1f}, new int[] {3}));
      norm.rebind(rebound);
      try (ModuleGrad grad =
              ModuleGrad.of(
                  norm, (params, inputs) -> new MLXArray[] {MLXOps.sum(norm.forward(inputs[0]))});
          MLXScope step = model.newChild()) {
        float[] x = {1, 2, 3};
        MLXArray input = MLX.array(step, x, new int[] {1, 3});
        float[] actual =
            grad.apply(step, new MLXArray[] {input}).grads().get("weight").toFloatArray();
        double invRms = 1 / Math.sqrt((1 + 4 + 9) / 3.0 + 1e-6);
        float[] expected = {
          (float) (x[0] * invRms), (float) (x[1] * invRms), (float) (x[2] * invRms)
        };
        assertArrayEquals(expected, actual, 1e-5f);
        assertTrue(Math.abs(actual[0]) > 0);
      }
    }
  }
}
