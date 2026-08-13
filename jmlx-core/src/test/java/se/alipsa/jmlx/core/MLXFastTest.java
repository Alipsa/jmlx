package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/initial-plan.md, Testing approach, "Numeric correctness": every facade op against hand-computed values.
 */
@EnabledIfNativeAvailable
class MLXFastTest {

  // 1e-3f, not MLXNumericTest's 1e-5f EPS: both goldens below involve an
  // irrational sqrt, and the rmsNorm golden additionally ignores eps's
  // sub-1e-5 perturbation of mean(x^2) -- ignoring it is fine well within
  // this tolerance.
  private static final float EPS = 1e-3f;

  @Test
  void rmsNormWithNullWeightNormalizesByTheRootMeanSquare() {
    try (MLXScope scope = new MLXScope()) {
      // mean(x^2) = (1+4+9+16)/4 = 7.5, so x * (1/sqrt(7.5)):
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      MLXArray result = MLXFast.rmsNorm(x, null, 1e-5f);
      assertArrayEquals(new float[] {0.365148f, 0.730296f, 1.095444f, 1.460592f}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void rmsNormWithWeightScalesTheNormalizedResult() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      MLXArray weight = MLX.array(scope, new float[] {2, 2, 2, 2}, new int[] {4});
      MLXArray result = MLXFast.rmsNorm(x, weight, 1e-5f);
      assertArrayEquals(new float[] {0.730296f, 1.460592f, 2.190888f, 2.921184f}, result.toFloatArray(), EPS);
    }
  }

  /**
   * Mirrors the spec's Testing-approach row "layerNorm(x, weight, bias) with a null weight still picks the innermost of
   * the non-null operands": weight is null and x/bias live in different scopes, so this exercises scopeOf's
   * nullable-operand skip together with its cross-scope resolution, not just the null-weight null-mapping alone.
   */
  @Test
  void rmsNormWithXInAChildScopeAndNullWeightAllocatesIntoTheChild() {
    try (MLXScope parent = new MLXScope()) {
      try (MLXScope child = parent.newChild()) {
        MLXArray x = MLX.array(child, new float[] {1, 2, 3, 4}, new int[] {4});
        MLXArray result = MLXFast.rmsNorm(x, null, 1e-5f);
        assertSame(child, result.scope());
        assertArrayEquals(new float[] {0.365148f, 0.730296f, 1.095444f, 1.460592f}, result.toFloatArray(), EPS);
      }
    }
  }

  @Test
  void layerNormWithNullWeightAndNullBiasNormalizesByMeanAndStd() {
    try (MLXScope scope = new MLXScope()) {
      // mean=2.5, var=1.25 (exact), std=sqrt(1.25)~=1.118034, (x-mean)/std:
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      MLXArray result = MLXFast.layerNorm(x, null, null, 1e-5f);
      assertArrayEquals(new float[] {-1.341641f, -0.447214f, 0.447214f, 1.341641f}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void layerNormWithWeightAndBiasScalesAndShiftsTheNormalizedResult() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      MLXArray weight = MLX.array(scope, new float[] {2, 2, 2, 2}, new int[] {4});
      MLXArray bias = MLX.array(scope, new float[] {1, 1, 1, 1}, new int[] {4});
      MLXArray result = MLXFast.layerNorm(x, weight, bias, 1e-5f);
      assertArrayEquals(new float[] {-1.683282f, 0.105573f, 1.894427f, 3.683282f}, result.toFloatArray(), EPS);
    }
  }

  /**
   * Both weight and bias non-null but in a different scope from x, proving scopeOf resolves the innermost across all
   * three real operands, not just skips the nullable ones.
   */
  @Test
  void layerNormWithXInAChildScopeAndWeightAndBiasInTheParentAllocatesIntoTheChild() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray weight = MLX.array(parent, new float[] {2, 2, 2, 2}, new int[] {4});
      MLXArray bias = MLX.array(parent, new float[] {1, 1, 1, 1}, new int[] {4});
      try (MLXScope child = parent.newChild()) {
        MLXArray x = MLX.array(child, new float[] {1, 2, 3, 4}, new int[] {4});
        MLXArray result = MLXFast.layerNorm(x, weight, bias, 1e-5f);
        assertSame(child, result.scope());
        assertArrayEquals(new float[] {-1.683282f, 0.105573f, 1.894427f, 3.683282f}, result.toFloatArray(), EPS);
      }
    }
  }
}
