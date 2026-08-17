package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SequencedMap;
import java.util.function.BiFunction;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXException;
import se.alipsa.jmlx.core.MLXOps;
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

  /**
   * This PR's round-10 review finding 3: {@link ModuleGrad} used to differentiate with respect to
   * every entry of {@code tree.parameters()} unconditionally, so this scenario used to throw {@code
   * MLXException} on the first {@code apply} call ({@code QuantizedMatmul}'s native backward pass
   * has no gradient with respect to a quantized weight at all). {@link ModuleGrad} now excludes a
   * non-floating-dtype parameter (in practice, only this layer's {@code UINT32}-packed {@code
   * weight}) from differentiation entirely, by dtype -- confirmed empirically that this avoids the
   * native failure outright rather than merely working around it, since the failure only fires when
   * the quantized weight's own gradient is actually requested. This pins the new behavior: {@code
   * apply} now succeeds even though {@code loss} reaches this layer's {@code forward}, {@code
   * grads()} contains real gradients for {@code scales}/{@code biases}, and does not contain {@code
   * "weight"} at all.
   */
  @Test
  void moduleGradOnAQuantizedLinearExcludesTheQuantizedWeightButTrainsScalesAndBiases() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss =
          (params, inputs) -> new MLXArray[] {MLXOps.sum(quantizedLinear.forward(inputs[0]))};
      try (ModuleGrad mg = ModuleGrad.of(quantizedLinear, loss)) {
        MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});

        ModuleGrad.Result result = mg.apply(scope, new MLXArray[] {x});

        SequencedMap<String, MLXArray> grads = result.grads();
        assertFalse(grads.containsKey("weight"), "weight must be excluded from differentiation");
        assertEquals(DType.FLOAT32, grads.get("scales").dtype());
        assertEquals(DType.FLOAT32, grads.get("biases").dtype());
      }
    }
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

  /**
   * This PR's round-7 review finding 1: on the "inverted" scope layout -- the model built in a
   * scope that is itself a descendant of {@code x}'s own (here, {@code x} from a root scope and the
   * model from a child of it) -- {@link MLXQuant#quantizedMatmul}'s default (innermost-of-all-
   * operands) overload would allocate the result into the model scope, leaking one array per {@code
   * forward} call. Confirmed empirically before this fix existed. {@code forward} now passes {@code
   * x.scope()} explicitly, so the result must land there instead, matching {@link Linear#forward}'s
   * own (bias-less) behavior on the same layout.
   */
  @Test
  void forwardTargetsXScopeUnderTheInvertedScopeLayout() {
    try (MLXScope root = new MLXScope()) {
      MLXArray x = MLX.array(root, inputFixture(), new int[] {1, IN});

      try (MLXScope model = root.newChild()) {
        MLXArray[] q = quantizedWeight(model);
        QuantizedLinear quantizedLinear =
            new QuantizedLinear(model, q[0], q[1], q[2], null, GROUP_SIZE, BITS);

        MLXArray result = quantizedLinear.forward(x);

        assertSame(root, result.scope());
      }
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

  /**
   * This PR's round-7 review finding 3: neither {@code scales} nor {@code biases} had any dtype
   * check, so an {@code INT32} pair passed construction and only failed at the next {@link
   * #forward} call, with native's own error ({@code "[quantized_matmul] Only real floating types
   * are supported ..."}) -- the same deferred-native-failure mode the existing {@code
   * weight.dtype()}/{@code groupSize}/{@code bits} checks exist to prevent, confirmed empirically
   * before this test existed.
   */
  @Test
  void constructorRejectsNonFloatingScalesOrBiases() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray int32Scales = MLX.astype(q[1], DType.INT32);
      MLXArray int32Biases = MLX.astype(q[2], DType.INT32);

      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], int32Scales, q[2], null, GROUP_SIZE, BITS));
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], q[1], int32Biases, null, GROUP_SIZE, BITS));
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

  /**
   * The same guard as {@link
   * #constructorRejectsAWeightWithAPackedColumnCountInconsistentWithGroupSizeAndBits}, but for a
   * non-power-of-2 {@code bits} value: {@code packedCols = in * bits / 32} holds for {@code bits=3}
   * too (confirmed empirically -- an earlier version of this constructor's check ran only when
   * {@code 32 % bits == 0}, on the wrong belief that {@code bits} in {@code {3, 5, 6}} pack
   * unevenly enough to break the formula, and so silently skipped this exact case).
   */
  @Test
  void constructorRejectsInconsistentPackedColumnCountForNonPowerOfTwoBits() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
      MLXArray[] q = MLXQuant.quantize(w, GROUP_SIZE, 3, "affine", null);
      assertThrows(
          IllegalArgumentException.class,
          () -> new QuantizedLinear(scope, q[0], q[1], q[2], null, 64, 3));
    }
  }

  /**
   * The positive direction {@link
   * #constructorRejectsInconsistentPackedColumnCountForNonPowerOfTwoBits} does not cover: a {@code
   * bits=3} weight whose {@code groupSize}/{@code bits} <em>are</em> consistent with the actual
   * {@code weight}/{@code scales} shapes must be accepted and must {@code forward} correctly.
   * Without this row, a regression that made the packed-column check wrongly reject every
   * non-power-of-2 {@code bits} value (or a future mlx-c version that changes 3/5/6-bit packing)
   * would still pass the whole suite, since the only other non-power-of-2 test asserts a rejection.
   */
  @Test
  void forwardWorksForNonPowerOfTwoBits() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
      MLXArray[] q = MLXQuant.quantize(w, GROUP_SIZE, 3, "affine", null);
      MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});

      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, 3);
      MLXArray quantizedResult = quantizedLinear.forward(x);

      MLXArray dequantizedWeight =
          MLXQuant.dequantize(q[0], q[1], q[2], GROUP_SIZE, 3, "affine", null, null);
      Linear linear = new Linear(scope, dequantizedWeight, null);
      MLXArray linearResult = linear.forward(x);

      assertArrayEquals(linearResult.toFloatArray(), quantizedResult.toFloatArray(), EPS);
    }
  }

  /**
   * This PR's round-9 review finding 1, narrowed by round-12 review finding 1: {@link
   * QuantizedLinear#onParametersUpdated(java.util.Set)} rejects any write touching {@code weight}
   * unconditionally (the shape check the constructor's own javadoc documents can never verify a
   * replacement was quantized under this layer's own {@code groupSize}/{@code bits} rather than a
   * different pair sharing the same product) -- this pins that a {@code weight} write is still
   * rejected and rolled back even when accompanied by a shape-consistent {@code scales}/ {@code
   * biases} replacement, unlike a {@code scales}/{@code biases}-only write (see {@link
   * #updateOfScalesAndBiasesAloneViaModuleUpdateSucceeds}), which round 12 fixed to succeed instead
   * of also being rejected unconditionally.
   */
  @Test
  void updateOfWeightViaModuleUpdateAlwaysThrowsAndRollsBack() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);

      float[] otherWeightData = new float[OUT * IN];
      for (int i = 0; i < otherWeightData.length; i++) {
        otherWeightData[i] = (i % 5 - 2) * 0.4f;
      }
      MLXArray w2 = MLX.array(scope, otherWeightData, new int[] {OUT, IN});
      MLXArray[] q2 = MLXQuant.quantize(w2, GROUP_SIZE, BITS, "affine", null);

      assertThrows(
          IllegalStateException.class,
          () -> quantizedLinear.update(Map.of("weight", q2[0], "scales", q2[1], "biases", q2[2])));

      // weight is UINT32 (the packed dtype), so toFloatArray()/toIntArray() don't apply here --
      // reference identity is both sufficient and exact: rollback restores the same MLXArray.
      assertSame(q[0], quantizedLinear.parameters().get("weight"));
      assertSame(q[1], quantizedLinear.parameters().get("scales"));
      assertSame(q[2], quantizedLinear.parameters().get("biases"));
    }
  }

  /**
   * This PR's round-12 review finding 1: {@code onParametersUpdated} previously rejected a {@code
   * scales}/{@code biases} write unconditionally, on the mistaken premise that the constructor's
   * groupSize/bits-versus-arithmetic ambiguity (which is specifically about {@code weight}'s own
   * identity) applied to a {@code scales}/{@code biases}-only write too -- it does not, since
   * {@code weight} never changes in that case. This pins that a shape/dtype-consistent {@code
   * scales}/ {@code biases}-only {@link Module#update} now succeeds, with {@code forward}
   * reflecting the new values -- the exact write a generic training loop applying a {@code
   * ModuleGrad} gradient step needs to make, without requiring {@link
   * QuantizedLinear#updateScalesAndBiases} or any typed reference to this layer at all.
   */
  @Test
  void updateOfScalesAndBiasesAloneViaModuleUpdateSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});

      MLXArray newScales = MLXOps.multiply(q[1], MLX.array(scope, new float[] {2f}, new int[] {}));
      MLXArray newBiases = MLXOps.add(q[2], MLX.array(scope, new float[] {0.1f}, new int[] {}));

      quantizedLinear.update(Map.of("scales", newScales, "biases", newBiases));
      MLXArray result = quantizedLinear.forward(x);

      QuantizedLinear rebuilt =
          new QuantizedLinear(scope, q[0], newScales, newBiases, null, GROUP_SIZE, BITS);
      MLXArray expected = rebuilt.forward(x);

      assertArrayEquals(expected.toFloatArray(), result.toFloatArray(), EPS);
      assertSame(newScales, quantizedLinear.parameters().get("scales"));
      assertSame(newBiases, quantizedLinear.parameters().get("biases"));
    }
  }

  /**
   * The validation {@link #updateOfScalesAndBiasesAloneViaModuleUpdateSucceeds} relies on actually
   * runs, rather than every {@code scales}/{@code biases} write now being silently accepted: a
   * replacement whose shape is inconsistent with this layer's unchanged {@code weight} (wrong
   * {@code out} dimension) is rejected and rolled back, exactly like the constructor's own check.
   */
  @Test
  void updateOfScalesAndBiasesWithAMismatchedShapeViaModuleUpdateThrowsAndRollsBack() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray wrongOutScales = MLX.array(scope, new float[] {1, 2}, new int[] {1, 2});
      MLXArray wrongOutBiases = MLX.array(scope, new float[] {1, 2}, new int[] {1, 2});

      assertThrows(
          IllegalArgumentException.class,
          () -> quantizedLinear.update(Map.of("scales", wrongOutScales, "biases", wrongOutBiases)));

      assertSame(q[1], quantizedLinear.parameters().get("scales"));
      assertSame(q[2], quantizedLinear.parameters().get("biases"));
    }
  }

  /**
   * This PR's round-10 review finding 1: {@code onParametersUpdated()} used to reject every write
   * to any of this layer's own parameters unconditionally, including a {@code bias}-only write --
   * even though {@code bias} has no quantization relationship to {@code groupSize}/{@code bits} at
   * all, unlike {@code weight}/{@code scales}/{@code biases}. This pins that a {@code bias}-only
   * {@link Module#update} now succeeds.
   */
  @Test
  void updateOfBiasAloneViaModuleUpdateSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray bias = MLX.array(scope, new float[] {0.5f, -0.5f}, new int[] {OUT});
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], bias, GROUP_SIZE, BITS);
      MLXArray newBias = MLX.array(scope, new float[] {1f, -1f}, new int[] {OUT});

      quantizedLinear.update(Map.of("bias", newBias));

      assertSame(newBias, quantizedLinear.parameters().get("bias"));
    }
  }

  /**
   * This PR's round-10 review finding 1's second half: before this fix, {@code onParametersUpdated}
   * over-rejecting a {@code bias}-only write meant a single {@link Module#update} call spanning
   * both a plain {@code Linear} sibling and this layer's {@code bias} would throw and roll back the
   * unrelated sibling's write too (per {@code Module}'s own documented write-rollback-on-any-
   * throwing-notify guarantee) -- collateral damage on a write this layer never had a real reason
   * to reject. This pins that such a combined call now succeeds for both siblings.
   */
  @Test
  void updateSpanningALinearSiblingAndThisLayersBiasSucceedsForBoth() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray linWeight = MLX.array(scope, new float[] {1f, 2f, 3f, 4f}, new int[] {2, 2});
      Linear lin = new Linear(scope, linWeight, null);
      MLXArray[] q = quantizedWeight(scope);
      MLXArray bias = MLX.array(scope, new float[] {0.5f, -0.5f}, new int[] {OUT});
      QuantizedLinear ql = new QuantizedLinear(scope, q[0], q[1], q[2], bias, GROUP_SIZE, BITS);
      TwoModuleTree tree = new TwoModuleTree(scope, lin, ql);

      MLXArray newLinWeight = MLX.array(scope, new float[] {5f, 6f, 7f, 8f}, new int[] {2, 2});
      MLXArray newBias = MLX.array(scope, new float[] {1f, -1f}, new int[] {OUT});

      tree.update(Map.of("lin.weight", newLinWeight, "ql.bias", newBias));

      assertSame(newLinWeight, lin.parameters().get("weight"));
      assertSame(newBias, ql.parameters().get("bias"));
    }
  }

  /**
   * The exact scenario this PR's round-12 review finding 1 reported as broken: for {@code Tree{lin:
   * Linear, ql: QuantizedLinear}}, a single {@link Module#update} call carrying shape-matched
   * gradient-style replacements for {@code lin.weight}/{@code ql.scales}/{@code ql.biases} used to
   * throw and roll back {@code lin.weight} too, collaterally -- so no layer got its step.
   * Deliberately never touches {@code ql} directly after construction (only {@code tree}),
   * demonstrating this PR's round-12 review finding 2's resolution as a side effect: a generic
   * training loop holding only the root module and a dotted-path grads map needs no typed reference
   * to the nested {@code QuantizedLinear} to apply a {@code scales}/{@code biases} step.
   */
  @Test
  void updateSpanningALinearSiblingAndThisLayersScalesAndBiasesSucceedsForBoth() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray linWeight = MLX.array(scope, new float[] {1f, 2f, 3f, 4f}, new int[] {2, 2});
      Linear lin = new Linear(scope, linWeight, null);
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear ql = new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      TwoModuleTree tree = new TwoModuleTree(scope, lin, ql);

      MLXArray newLinWeight = MLX.array(scope, new float[] {5f, 6f, 7f, 8f}, new int[] {2, 2});
      MLXArray newScales = MLXOps.multiply(q[1], MLX.array(scope, new float[] {2f}, new int[] {}));
      MLXArray newBiases = MLXOps.add(q[2], MLX.array(scope, new float[] {0.1f}, new int[] {}));

      tree.update(
          Map.of("lin.weight", newLinWeight, "ql.scales", newScales, "ql.biases", newBiases));

      assertSame(newLinWeight, tree.parameters().get("lin.weight"));
      assertSame(newScales, tree.parameters().get("ql.scales"));
      assertSame(newBiases, tree.parameters().get("ql.biases"));
    }
  }

  private static final class TwoModuleTree extends Module {
    TwoModuleTree(MLXScope scope, Linear lin, QuantizedLinear ql) {
      super(scope);
      child("lin", lin);
      child("ql", ql);
    }
  }

  /**
   * The safe replacement path {@link QuantizedLinear#updateQuantization} exists for: {@code
   * weight}/{@code scales}/{@code biases} and {@code groupSize}/{@code bits} change together, as
   * one atomic unit, so there is no window where the cached {@code groupSize}/{@code bits} describe
   * data other than the one they are paired with. Constructed at {@code groupSize=32,bits=4};
   * replaced with a payload actually quantized at {@code groupSize=64,bits=2} -- the exact pair
   * this PR's round-7/8 tests used to probe the (then-real) {@code update} blind spot -- and {@code
   * forward} must now compute the mathematically correct result for the new configuration, not a
   * mismatched one.
   */
  @Test
  void updateQuantizationReplacesWeightScalesBiasesAndGroupSizeBitsTogetherAndForwardIsCorrect() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});

      MLXArray w2 = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
      MLXArray[] q2 = MLXQuant.quantize(w2, 64, 2, "affine", null);

      quantizedLinear.updateQuantization(q2[0], q2[1], q2[2], null, 64, 2);
      MLXArray quantizedResult = quantizedLinear.forward(x);

      MLXArray dequantizedWeight =
          MLXQuant.dequantize(q2[0], q2[1], q2[2], 64, 2, "affine", null, null);
      Linear linear = new Linear(scope, dequantizedWeight, null);
      MLXArray linearResult = linear.forward(x);

      assertArrayEquals(linearResult.toFloatArray(), quantizedResult.toFloatArray(), EPS);
    }
  }

  /**
   * This PR's round-11 review finding 1: {@code updateQuantization} is the only sanctioned way to
   * replace {@code scales}/{@code biases}, but it forces re-supplying {@code weight}/{@code bias}/
   * {@code groupSize}/{@code bits} unchanged too -- and {@code groupSize}/{@code bits} are not
   * exposed by this class at all, so a generic training loop applying a gradient step has no way to
   * obtain them. {@link QuantizedLinear#updateScalesAndBiases} is the narrower escape hatch: this
   * pins that it replaces {@code scales}/{@code biases} alone and {@code forward} reflects the new
   * values, without touching {@code weight}/{@code groupSize}/{@code bits} (confirmed by comparing
   * against a fresh {@code QuantizedLinear} built directly from the new {@code scales}/ {@code
   * biases} and this layer's original {@code weight}).
   */
  @Test
  void updateScalesAndBiasesReplacesBothAndForwardReflectsTheNewValues() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});

      MLXArray newScales = MLXOps.multiply(q[1], MLX.array(scope, new float[] {2f}, new int[] {}));
      MLXArray newBiases = MLXOps.add(q[2], MLX.array(scope, new float[] {0.1f}, new int[] {}));

      quantizedLinear.updateScalesAndBiases(newScales, newBiases);
      MLXArray result = quantizedLinear.forward(x);

      QuantizedLinear rebuilt =
          new QuantizedLinear(scope, q[0], newScales, newBiases, null, GROUP_SIZE, BITS);
      MLXArray expected = rebuilt.forward(x);

      assertArrayEquals(expected.toFloatArray(), result.toFloatArray(), EPS);
      assertSame(newScales, quantizedLinear.parameters().get("scales"));
      assertSame(newBiases, quantizedLinear.parameters().get("biases"));
    }
  }

  @Test
  void updateScalesAndBiasesRejectsNullArguments() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      assertThrows(
          NullPointerException.class, () -> quantizedLinear.updateScalesAndBiases(null, q[2]));
      assertThrows(
          NullPointerException.class, () -> quantizedLinear.updateScalesAndBiases(q[1], null));
    }
  }

  @Test
  void updateScalesAndBiasesRejectsNonFloatingDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray int32Scales = MLX.astype(q[1], DType.INT32);
      MLXArray int32Biases = MLX.astype(q[2], DType.INT32);
      assertThrows(
          IllegalArgumentException.class,
          () -> quantizedLinear.updateScalesAndBiases(int32Scales, q[2]));
      assertThrows(
          IllegalArgumentException.class,
          () -> quantizedLinear.updateScalesAndBiases(q[1], int32Biases));
    }
  }

  @Test
  void updateScalesAndBiasesRejectsAShapeMismatch() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray wrongShapedScales = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3});
      assertThrows(
          IllegalArgumentException.class,
          () -> quantizedLinear.updateScalesAndBiases(wrongShapedScales, q[2]));
      assertThrows(
          IllegalArgumentException.class,
          () -> quantizedLinear.updateScalesAndBiases(q[1], wrongShapedScales));
    }
  }

  /**
   * {@code updateScalesAndBiases} uses {@link Module#rebind}, not {@link Module#update}, the same
   * choice {@link QuantizedLinear#updateQuantization} already makes -- this pins that a {@code
   * scales}/{@code biases} replacement through it does NOT throw, even though the identical
   * replacement through {@link Module#update} always does (see {@link
   * #updateOfWeightScalesOrBiasesViaModuleUpdateAlwaysThrowsAndRollsBack}).
   */
  @Test
  void updateScalesAndBiasesDoesNotFireOnParametersUpdated() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray[] q2 = quantizedWeight(scope);

      quantizedLinear.updateScalesAndBiases(q2[1], q2[2]);

      assertSame(q2[1], quantizedLinear.parameters().get("scales"));
      assertSame(q2[2], quantizedLinear.parameters().get("biases"));
    }
  }

  /**
   * This PR's round-12 review finding 3: {@link Module#rebind} is {@code public} and {@code final}
   * -- inherited, un-overridable, un-narrowable -- so it bypasses every invariant this class tries
   * to protect, including the one {@link QuantizedLinear#updateQuantization}'s own javadoc used to
   * claim was now "impossible": a caller can replace {@code weight} directly via {@code rebind},
   * leaving {@code groupSize}/{@code bits} stale, with zero validation. Reuses the exact
   * mismatched-quantization construction {@link
   * #constructorAcceptsAGroupSizeBitsMismatchAndForwardSilentlyMiscomputes} pins for the
   * constructor, but via {@code rebind} on an already-constructed layer instead: {@code rebind}
   * itself does not throw, and {@code forward} then silently miscomputes exactly like the
   * constructor case.
   */
  @Test
  void rebindOfWeightAloneBypassesValidationAndSilentlyBreaksTheGroupSizeBitsInvariant() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] originalQ = quantizedWeight(scope);

      // Genuinely quantized at groupSize=64,bits=2 (product 128), rebound onto a layer that still
      // believes groupSize=32,bits=4 (same product) -- the identical mismatch the constructor's own
      // javadoc documents, reached here through rebind instead of construction.
      MLXArray w = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
      MLXArray[] mismatchedQ = MLXQuant.quantize(w, 64, 2, "affine", null);
      SequencedMap<String, MLXArray> replacement = new LinkedHashMap<>();
      replacement.put("weight", mismatchedQ[0]);
      replacement.put("scales", mismatchedQ[1]);
      replacement.put("biases", mismatchedQ[2]);

      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, originalQ[0], originalQ[1], originalQ[2], null, 32, 4);
      quantizedLinear.rebind(replacement);

      assertSame(mismatchedQ[0], quantizedLinear.parameters().get("weight"));

      // x's width (32) matches the wrongly-derived in = scales.shape()[1] * declaredGroupSize
      // (1 * 32), not the true in=64 -- the same coincidence
      // constructorAcceptsAGroupSizeBitsMismatchAndForwardSilentlyMiscomputes exploits.
      float[] onesData = new float[32];
      Arrays.fill(onesData, 1f);
      MLXArray x = MLX.array(scope, onesData, new int[] {1, 32});

      MLXArray miscomputed = quantizedLinear.forward(x);

      MLXArray dequantizedUnderTheWrongDeclaration =
          MLXQuant.dequantize(
              mismatchedQ[0], mismatchedQ[1], mismatchedQ[2], 32, 4, "affine", null, null);
      Linear linear = new Linear(scope, dequantizedUnderTheWrongDeclaration, null);
      MLXArray expectedMiscomputedResult = linear.forward(x);

      assertArrayEquals(expectedMiscomputedResult.toFloatArray(), miscomputed.toFloatArray(), EPS);
    }
  }

  /**
   * The other direction of the same round-12 review finding 3 hole: {@code rebind} does not even
   * dtype-check {@code weight}, so a {@code FLOAT32} replacement succeeds silently at {@code
   * rebind} time and only fails later, deep inside native code, at the next {@link #forward} call
   * -- the exact deferred-native-failure mode this class's constructor validation otherwise exists
   * to prevent, reopened here because {@code rebind} cannot run that validation at all.
   */
  @Test
  void rebindOfWeightWithAFloat32ArraySucceedsButFailsLaterAtForward() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      MLXArray floatWeight = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
      SequencedMap<String, MLXArray> replacement = new LinkedHashMap<>();
      replacement.put("weight", floatWeight);

      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      quantizedLinear.rebind(replacement);
      assertSame(floatWeight, quantizedLinear.parameters().get("weight"));

      MLXArray x = MLX.array(scope, inputFixture(), new int[] {1, IN});
      assertThrows(MLXException.class, () -> quantizedLinear.forward(x));
    }
  }

  @Test
  void updateQuantizationValidatesLikeTheConstructor() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear quantizedLinear =
          new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);

      float[] threeRowWeightData = new float[3 * IN];
      for (int i = 0; i < threeRowWeightData.length; i++) {
        threeRowWeightData[i] = (i % 7 - 3) * 0.3f;
      }
      MLXArray threeRowWeight = MLX.array(scope, threeRowWeightData, new int[] {3, IN});
      MLXArray[] threeRowQuantized =
          MLXQuant.quantize(threeRowWeight, GROUP_SIZE, BITS, "affine", null);

      assertThrows(
          IllegalArgumentException.class,
          () ->
              quantizedLinear.updateQuantization(
                  threeRowQuantized[0], q[1], q[2], null, GROUP_SIZE, BITS));
    }
  }

  @Test
  void updateQuantizationRejectsABiasNullnessMismatch() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedWeight(scope);
      QuantizedLinear noBias = new QuantizedLinear(scope, q[0], q[1], q[2], null, GROUP_SIZE, BITS);
      MLXArray bias = MLX.array(scope, new float[] {0.5f, -0.5f}, new int[] {OUT});
      assertThrows(
          IllegalArgumentException.class,
          () -> noBias.updateQuantization(q[0], q[1], q[2], bias, GROUP_SIZE, BITS));

      QuantizedLinear withBias =
          new QuantizedLinear(scope, q[0], q[1], q[2], bias, GROUP_SIZE, BITS);
      assertThrows(
          IllegalArgumentException.class,
          () -> withBias.updateQuantization(q[0], q[1], q[2], null, GROUP_SIZE, BITS));
    }
  }

  /**
   * This PR's round-9 review finding 1, the "real pin" finding 2 asked for: a fundamental, proven
   * limitation, not a narrow coincidence. {@code packedCols = in * bits / 32}, with {@code in =
   * scales.shape()[1] * groupSize}, depends only on the product {@code groupSize * bits} -- so a
   * {@code weight}/{@code scales} pair genuinely quantized at {@code groupSize=64,bits=2} (product
   * 128) is indistinguishable, by any shape check, from one genuinely quantized at {@code
   * groupSize=32,bits=4} (same product) -- confirmed here by constructing directly (not via {@code
   * update}) with the mismatched declaration and an activation width ({@code [1,32]}) that
   * coincidentally matches the wrongly-derived {@code in=32}: no exception anywhere, and the
   * computed result provably disagrees with the correct ({@code groupSize=64,bits=2}) computation
   * on the same data. This is a documented, permanent limitation of shape-based validation (see the
   * constructor's own javadoc) -- this test exists to keep that limitation visible and correctly
   * characterized, not to bless it as acceptable or to imply a future fix is expected here.
   */
  @Test
  void constructorAcceptsAGroupSizeBitsMismatchAndForwardSilentlyMiscomputes() {
    try (MLXScope scope = new MLXScope()) {
      // IN=64 is a multiple of both the true groupSize (64) and the wrongly-declared one (32),
      // so this reproduces the exact scenario the constructor's javadoc describes: q is genuinely
      // quantized at groupSize=64,bits=2 (product 128), then declared groupSize=32,bits=4 (same
      // product 128) -- accepted outright by the constructor, not just via update().
      MLXArray w = MLX.array(scope, weightFixture(), new int[] {OUT, IN});
      MLXArray[] q = MLXQuant.quantize(w, 64, 2, "affine", null);

      QuantizedLinear mismatched = new QuantizedLinear(scope, q[0], q[1], q[2], null, 32, 4);

      // x's width (32) matches the wrongly-derived in = scales.shape()[1] * declaredGroupSize
      // (1 * 32), not the true in=64 -- the coincidence that lets forward proceed with no error
      // at all, silently decoding a groupSize=64,bits=2 payload as groupSize=32,bits=4.
      float[] onesData = new float[32];
      Arrays.fill(onesData, 1f);
      MLXArray x = MLX.array(scope, onesData, new int[] {1, 32});

      MLXArray miscomputed = mismatched.forward(x);

      // Pin the miscomputation as exactly the deterministic result of decoding q under the wrong
      // (declared) groupSize/bits -- not merely "some value", and not an apples-to-apples
      // comparison against the true groupSize=64,bits=2 decoding, which would need a [1,64] input
      // and so has no valid comparison against this [1,32] one at all (that mismatch is the point).
      MLXArray dequantizedUnderTheWrongDeclaration =
          MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, "affine", null, null);
      Linear linear = new Linear(scope, dequantizedUnderTheWrongDeclaration, null);
      MLXArray expectedMiscomputedResult = linear.forward(x);

      assertArrayEquals(expectedMiscomputedResult.toFloatArray(), miscomputed.toFloatArray(), EPS);
    }
  }

  /**
   * This PR's round-9 review finding 4: {@link QuantizedLinear#forward}'s own javadoc documents
   * that the {@code hasBias} branch's {@code MLXOps.add(y, bias)} is not scope-targeted (unlike the
   * {@code quantizedMatmul} call above it), so it can still leak into the model scope under the
   * inverted layout {@link #forwardTargetsXScopeUnderTheInvertedScopeLayout} covers only for the
   * bias-less path. This pins that documented residual gap for the bias-bearing path, rather than
   * leaving it as a claim with no regression guard: the result's scope is the model scope, not
   * {@code x}'s, precisely because there is currently no fix for it (see {@code forward}'s javadoc
   * for why closing it is out of scope for this class).
   *
   * <p><strong>This assertion pins a known, deliberately-unfixed limitation, not desired
   * behavior.</strong> If {@code MLXOps#add} ever gains an explicit-target overload and {@code
   * forward} is updated to use it (the fix this class's own javadoc says would require touching
   * {@code MLXOps}, shared by every op in this codebase), this exact assertion is expected to start
   * failing -- and should then be inverted to {@code assertSame(x.scope(), result.scope())} to pin
   * the fixed (non-leaking) behavior instead, not treated as a regression to chase down.
   */
  @Test
  void forwardWithBiasLeaksIntoModelScopeUnderTheInvertedScopeLayout() {
    try (MLXScope root = new MLXScope()) {
      MLXArray x = MLX.array(root, inputFixture(), new int[] {1, IN});

      try (MLXScope model = root.newChild()) {
        MLXArray[] q = quantizedWeight(model);
        MLXArray bias = MLX.array(model, new float[] {0.5f, -0.5f}, new int[] {OUT});
        QuantizedLinear quantizedLinear =
            new QuantizedLinear(model, q[0], q[1], q[2], bias, GROUP_SIZE, BITS);

        MLXArray result = quantizedLinear.forward(x);

        assertSame(model, result.scope());
      }
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
