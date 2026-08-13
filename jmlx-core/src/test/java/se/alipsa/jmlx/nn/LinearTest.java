package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.SequencedMap;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/phase4-plan.md §2 for the withdrawn-cache mitigation {@link Linear#forward} implements:
 * {@code weight} is registered in the checkpoint's {@code [out, in]} layout and transposed fresh,
 * into the caller's scope, on every call.
 */
@EnabledIfNativeAvailable
class LinearTest {

  private static final float EPS = 1e-5f;

  private static final int IN_FEATURES = 100_000;
  private static final int OUT_FEATURES = 4;
  private static final int WARMUP_ITERATIONS = 50;
  private static final int MEASURED_ITERATIONS = 200;
  private static final long LEAK_THRESHOLD_BYTES = 2_000_000;

  /**
   * Wraps a single child module named "proj" -- test-local, like {@code ModuleTest}'s {@code
   * Branch}.
   */
  private static final class Wrapper extends Module {
    Wrapper(MLXScope scope, Module child) {
      super(scope);
      child("proj", child);
    }
  }

  @Test
  void forwardComputesTheAffineTransform() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1, 0, 1, 0, 1, 1}, new int[] {2, 3});
      MLXArray bias = MLX.array(scope, new float[] {10, 20}, new int[] {2});
      Linear linear = new Linear(scope, weight, bias);
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3});

      MLXArray y = linear.forward(x);

      assertArrayEquals(new int[] {1, 2}, y.shape());
      assertArrayEquals(new float[] {14, 25}, y.toFloatArray(), EPS);
    }
  }

  /**
   * {@code null} bias's only other coverage is {@code constructorRejectsARankOneWeight}, which
   * never reaches {@code forward()} -- this is the only test that actually exercises the {@code
   * hasBias ? add : y} branch's {@code false} side with a value assertion.
   */
  @Test
  void forwardWithoutBiasComputesTheLinearTransformOnly() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1, 0, 1, 0, 1, 1}, new int[] {2, 3});
      Linear linear = new Linear(scope, weight, null);
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3});

      MLXArray y = linear.forward(x);

      assertArrayEquals(new int[] {1, 2}, y.shape());
      assertArrayEquals(new float[] {4, 5}, y.toFloatArray(), EPS);
    }
  }

  /**
   * The spec's named regression test: a {@code W.T} registration would still pass {@link
   * #forwardComputesTheAffineTransform} above and only fail here.
   */
  @Test
  void parametersReturnsWeightInTheCheckpointsOutInShapeNotItsTranspose() {
    try (MLXScope scope = new MLXScope()) {
      float[] weightData = {1, 0, 1, 0, 1, 1};
      MLXArray weight = MLX.array(scope, weightData, new int[] {2, 3});
      MLXArray bias = MLX.array(scope, new float[] {10, 20}, new int[] {2});
      Linear linear = new Linear(scope, weight, bias);

      SequencedMap<String, MLXArray> params = linear.parameters();

      assertArrayEquals(new int[] {2, 3}, params.get("weight").shape());
      assertArrayEquals(weightData, params.get("weight").toFloatArray(), EPS);
    }
  }

  @Test
  void constructorRejectsARankOneWeight() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> new Linear(scope, weight, null));
    }
  }

  @Test
  void constructorRejectsABiasOfTheWrongLength() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1, 0, 1, 0, 1, 1}, new int[] {2, 3});
      MLXArray bias = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> new Linear(scope, weight, bias));
    }
  }

  /**
   * The {@code Linear} is the nested module, not the direct receiver of {@code update} -- a {@code
   * Linear} that IS the receiver would pass even if the depth-first tree walk in {@code
   * Module.update} were broken.
   */
  @Test
  void updateOnAParentHoldingANestedLinearThenForwardUsesTheNewWeights() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1, 0, 1, 0, 1, 1}, new int[] {2, 3});
      MLXArray bias = MLX.array(scope, new float[] {10, 20}, new int[] {2});
      Linear linear = new Linear(scope, weight, bias);
      Wrapper wrapper = new Wrapper(scope, linear);
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3});

      MLXArray original = linear.forward(x);
      assertArrayEquals(new float[] {14, 25}, original.toFloatArray(), EPS);

      MLXArray newWeight = MLX.array(scope, new float[] {2, 0, 0, 0, 2, 0}, new int[] {2, 3});
      wrapper.update(Map.of("proj.weight", newWeight));

      // new weight row0=[2,0,0], row1=[0,2,0]; x=[1,2,3] -> [2,4] + bias[10,20] = [12,24]
      MLXArray updated = linear.forward(x);
      assertArrayEquals(new float[] {12, 24}, updated.toFloatArray(), EPS);
    }
  }

  /**
   * The one test in this task with no compile-time signal for the bug it catches: a {@code
   * transpose(weight)} accidentally allocating into {@code weight.scope()} instead of {@code
   * x.scope()} would leak the transposed weight copy into the long-lived parent once per iteration.
   * Same warmup/measured-iteration/threshold shape as {@code MLXMemoryLeakTest}'s {@code
   * activeMemoryDoesNotGrowWithPerIterationChildScopeUnderALongLivedParent}.
   */
  @Test
  void activeMemoryDoesNotGrowWithPerIterationChildScopeUnderALongLivedParent() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray weight =
          MLX.array(
              parent, new float[OUT_FEATURES * IN_FEATURES], new int[] {OUT_FEATURES, IN_FEATURES});
      Linear linear = new Linear(parent, weight, null);

      for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        runChildScopeIteration(parent, linear);
      }

      long baseline = NativeMemoryProbe.activeMemoryBytes();

      for (int i = 0; i < MEASURED_ITERATIONS; i++) {
        runChildScopeIteration(parent, linear);
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

  private static void runChildScopeIteration(MLXScope parent, Linear linear) {
    try (MLXScope child = parent.newChild()) {
      MLXArray x = MLX.array(child, new float[IN_FEATURES], new int[] {1, IN_FEATURES});
      MLXArray y = linear.forward(x);
      MLX.eval(y);
    }
  }
}
