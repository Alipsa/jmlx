package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SequencedMap;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Fixture: {@code [2, 64]} float weight ({@code out=2}, {@code in=64} -- the smallest shape
 * divisible by the {@code groupSize=32} used throughout, doubled to {@code out=2} so shape-mismatch
 * tests on {@code bias}/{@code scales} have a non-trivial dimension to get wrong), built with
 * {@code (i % 7 - 3) * 0.3f}; {@code x} fixture {@code (i % 5 - 2) * 0.2f}, shape {@code [1,64]};
 * {@code bias = [0.5, -0.5]}. {@code EPS = 1e-3f} -- verified against this exact fixture
 * (req/plans/phase4-m4-plan.md's Task 2 test-table note): a composition identity on identical
 * quantized components (fused vs. unfused arithmetic), not a lossy comparison against the true
 * unquantized weight.
 */
@EnabledIfNativeAvailable
class QuantizedLinearTest {

  private static final float EPS = 1e-3f;
  private static final int OUT = 2;
  private static final int IN = 64;
  private static final int GROUP_SIZE = 32;
  private static final int BITS = 4;

  private static final int WARMUP_ITERATIONS = 50;
  private static final int MEASURED_ITERATIONS = 200;
  private static final long LEAK_THRESHOLD_BYTES = 2_000_000;

  // Dedicated to the leak test below, at LinearTest's own fixture scale -- not this file's
  // shape-assertion OUT/IN (2/64), whose per-iteration x+y footprint (~264 bytes) is too small for
  // LEAK_THRESHOLD_BYTES to discriminate a real per-iteration leak from noise (a leak that size
  // over MEASURED_ITERATIONS iterations would land ~two orders of magnitude under the threshold).
  private static final int LEAK_IN_FEATURES = 100_000;
  private static final int LEAK_OUT_FEATURES = 4;

  private static float[] weightFixture() {
    float[] w = new float[OUT * IN];
    for (int i = 0; i < w.length; i++) {
      w[i] = (i % 7 - 3) * 0.3f;
    }
    return w;
  }

  private static float[] inputFixture() {
    float[] x = new float[IN];
    for (int i = 0; i < x.length; i++) {
      x[i] = (i % 5 - 2) * 0.2f;
    }
    return x;
  }

  private static MLXArray[] quantizedWeight(MLXScope scope) {
    MLXArray w = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
    return MLXQuant.quantize(w, GROUP_SIZE, BITS, "affine", null);
  }

  @Test
  void forwardMatchesLinearOnTheDequantizedWeight() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray bias = MLX.array(scope, new float[] {0.5f, -0.5f}, new int[] {OUT});
      MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});

      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], bias, GROUP_SIZE, BITS);
      MLXArray quantizedResult = quantizedLinear.forward(x);

      MLXArray dequantizedWeight =
          MLXQuant.dequantize(q[0], q[1], q[2], GROUP_SIZE, BITS, "affine", null, null);
      Linear linear = new Linear(scope, dequantizedWeight, bias);
      MLXArray linearResult = linear.forward(x);

      assertArrayEquals(linearResult.toFloatArray(), quantizedResult.toFloatArray(), EPS);
    }
  }

  @Test
  void forwardWithoutBiasComputesTheQuantizedTransformOnly() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});

      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray quantizedResult = quantizedLinear.forward(x);

      MLXArray dequantizedWeight =
          MLXQuant.dequantize(q[0], q[1], q[2], GROUP_SIZE, BITS, "affine", null, null);
      Linear linear = new Linear(scope, dequantizedWeight, null);
      MLXArray linearResult = linear.forward(x);

      assertArrayEquals(linearResult.toFloatArray(), quantizedResult.toFloatArray(), EPS);
    }
  }

  @Test
  void parametersExposesWeightScalesBiasesAndOptionalBiasInTheCheckpointLayout() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray bias = MLX.array(scope, new float[] {0.5f, -0.5f}, new int[] {OUT});

      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], bias, GROUP_SIZE, BITS);
      SequencedMap<String, MLXArray> params = quantizedLinear.parameters();

      assertArrayEquals(q[0].shape(), params.get("weight").shape());
      assertArrayEquals(q[1].shape(), params.get("scales").shape());
      assertArrayEquals(q[2].shape(), params.get("biases").shape());
      assertArrayEquals(new float[] {0.5f, -0.5f}, params.get("bias").toFloatArray(), EPS);
    }
  }

  @Test
  void constructorRejectsScalesAndBiasesWithMismatchedShapes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray wrongShapedBiases = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3});
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], q[1], wrongShapedBiases, null, GROUP_SIZE, BITS));
    }
  }

  @Test
  void constructorRejectsScalesWithWrongOutDimension() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray wrongOutScales = MLX.array(scope, new float[] {1, 2}, new int[] {1, 2});
      MLXArray wrongOutBiases = MLX.array(scope, new float[] {1, 2}, new int[] {1, 2});
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new QuantizedLinear(
                  scope, q[0], wrongOutScales, wrongOutBiases, null, GROUP_SIZE, BITS));
    }
  }

  @Test
  void constructorRejectsABiasOfTheWrongLength() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray wrongBias = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], q[1], q[2], wrongBias, GROUP_SIZE, BITS));
    }
  }

  @Test
  void constructorRejectsAnUnsupportedGroupSizeOrBits() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], q[1], q[2], null, 100, BITS));
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, 7));
    }
  }

  @Test
  void constructorRejectsAFloat32Weight() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray floatWeight = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, floatWeight, q[1], q[2], null, GROUP_SIZE, BITS));
    }
  }

  @Test
  void constructorRejectsAWeightWithAPackedColumnCountInconsistentWithGroupSizeAndBits() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], q[1], q[2], null, 64, BITS));
    }
  }

  @Test
  void activeMemoryDoesNotGrowWithPerIterationChildScopeUnderALongLivedParent() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray weight =
          MLX.array(
              parent,
              new float[LEAK_OUT_FEATURES * LEAK_IN_FEATURES],
              new int[] {LEAK_OUT_FEATURES, LEAK_IN_FEATURES});
      MLXArray[] q = MLXQuant.quantize(weight, GROUP_SIZE, BITS, "affine", null);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(parent, q[0], q[1], q[2], null, GROUP_SIZE, BITS);

      for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        runLeakTestIteration(parent, quantizedLinear);
      }

      long baseline = NativeMemoryProbe.activeMemoryBytes();

      for (int i = 0; i < MEASURED_ITERATIONS; i++) {
        runLeakTestIteration(parent, quantizedLinear);
      }

      long after = NativeMemoryProbe.activeMemoryBytes();

      assertTrue(
          after - baseline <= LEAK_THRESHOLD_BYTES,
          "active memory grew from "
              + baseline
              + " to "
              + after
              + " bytes over "
              + MEASURED_ITERATIONS
              + " per-iteration child scopes (threshold "
              + LEAK_THRESHOLD_BYTES
              + " bytes)");
    }
  }

  private static void runLeakTestIteration(MLXScope parent, QuantizedLinear quantizedLinear) {
    try (MLXScope child = parent.newChild()) {
      MLXArray x = MLX.array(child, new float[LEAK_IN_FEATURES], new int[] {1, LEAK_IN_FEATURES});
      MLXArray y = quantizedLinear.forward(x);
      MLX.eval(y);
    }
  }
}
