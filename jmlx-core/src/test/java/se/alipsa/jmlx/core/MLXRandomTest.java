package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/initial-plan.md, Testing approach, "Numeric correctness". Unlike the other op classes, {@code normal}/
 * {@code uniform}'s actual values are the RNG's own, not a formula's -- the only meaningful assertions here are shape/
 * dtype, seeded determinism, and (for uniform) a range property, not hand-computed goldens.
 *
 * <p>
 * {@link #seedMakesNormalDeterministic} and {@link #uniformEveryElementFallsWithinTheRequestedHalfOpenRange} both call
 * {@link MLXRandom#seed}, which reseeds mlx's process-wide RNG state -- not per-thread or per-scope. This is only safe
 * because JUnit runs this suite's methods sequentially (no {@code junit-platform.properties} enabling parallel
 * execution anywhere in this project); enabling parallel test execution later would make these two tests flaky against
 * each other and any other test that seeds the RNG.
 */
@EnabledIfNativeAvailable
class MLXRandomTest {

  @Test
  void normalProducesTheRequestedShapeAndDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray result = MLXRandom.normal(scope, new int[] {100}, DType.FLOAT32, 0f, 1f);
      assertArrayEquals(new int[] {100}, result.shape());
      assertEquals(DType.FLOAT32, result.dtype());
    }
  }

  @Test
  void uniformProducesTheRequestedShapeAndDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray result = MLXRandom.uniform(scope, new int[] {100}, DType.FLOAT32, -1f, 1f);
      assertArrayEquals(new int[] {100}, result.shape());
      assertEquals(DType.FLOAT32, result.dtype());
    }
  }

  /**
   * The standard way to test a seeded RNG without pinning this repo to mlx's exact algorithm: same seed produces
   * bit-identical results across two independent scopes, and re-seeding back to the original value reproduces the first
   * call's values exactly.
   */
  @Test
  void seedMakesNormalDeterministic() {
    try (MLXScope scopeA = new MLXScope()) {
      MLXRandom.seed(42);
      float[] first = MLXRandom.normal(scopeA, new int[] {50}, DType.FLOAT32, 0f, 1f).toFloatArray();
      try (MLXScope scopeB = new MLXScope()) {
        MLXRandom.seed(42);
        float[] second = MLXRandom.normal(scopeB, new int[] {50}, DType.FLOAT32, 0f, 1f).toFloatArray();
        assertArrayEquals(first, second, 0f);
      }
      MLXRandom.seed(7);
      try (MLXScope scopeC = new MLXScope()) {
        MLXRandom.seed(42);
        float[] third = MLXRandom.normal(scopeC, new int[] {50}, DType.FLOAT32, 0f, 1f).toFloatArray();
        assertArrayEquals(first, third, 0f);
      }
    }
  }

  @Test
  void uniformEveryElementFallsWithinTheRequestedHalfOpenRange() {
    // No golden is possible here: the values are the RNG's own, not a formula's. A fixed seed plus a
    // range-property assertion is the achievable substitute -- this comment documents that it's a
    // deliberate choice, not an oversight.
    try (MLXScope scope = new MLXScope()) {
      MLXRandom.seed(1234);
      float[] values = MLXRandom.uniform(scope, new int[] {1000}, DType.FLOAT32, -1f, 1f).toFloatArray();
      for (float v : values) {
        assertTrue(v >= -1f && v < 1f, "value " + v + " outside [-1, 1)");
      }
    }
  }
}
