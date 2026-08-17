package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/initial-plan.md, Testing approach, "Numeric correctness": every facade op against
 * hand-computed values. Fixture (this file's default, unless a row is specifically testing a
 * different value): {@code w[i] = i * 0.1f - 3.2f} for {@code i} in {@code 0..63}, shape {@code
 * [1,64]}, {@code group_size=32}, {@code bits=4}, {@code mode="affine"} -- pinned to the exact
 * fixture req/plans/phase4-m4-plan.md's Findings section measured, not a restated formula that may
 * drift from the measured number.
 */
@EnabledIfNativeAvailable
class MLXQuantTest {

  // Quantization error is not float rounding error: above the Findings section's measured
  // ~0.1067 max absolute round-trip error for this exact fixture.
  private static final float EPS_ROUNDTRIP = 0.15f;

  // Composition-identity rows compare two computations over the *same* quantized components
  // (float32 rounding noise only, Findings section: measured diffs ~1.9e-6/~2.4e-7), so these use
  // the facade's usual exact tolerance rather than a freshly loosened constant.
  private static final float EPS_EXACT = 1e-5f;

  private static float[] defaultFixture() {
    float[] w = new float[64];
    for (int i = 0; i < 64; i++) {
      w[i] = i * 0.1f - 3.2f;
    }
    return w;
  }

  /**
   * The absent-{@code dtype} default is not unconditionally FLOAT32 (see {@link #dequantize}'s own
   * javadoc note): it follows {@code scales}' own dtype. Confirmed empirically by quantizing a
   * weight already {@code astype}'d to FLOAT16 -- {@code quantize} then produces FLOAT16 {@code
   * scales}, and a subsequent {@code dequantize(..., dtype=null)} returns FLOAT16, not FLOAT32.
   */
  @Test
  void dequantizeDefaultDtypeFollowsScalesDtypeNotAlwaysFloat32() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray w16 = MLX.astype(w, DType.FLOAT16);
      MLXArray[] q = MLXQuant.quantize(w16, 32, 4, "affine", null);
      assertEquals(DType.FLOAT16, q[1].dtype());
      MLXArray result = MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, "affine", null, null);
      assertEquals(DType.FLOAT16, result.dtype());
    }
  }

  @Test
  void quantizeWithMxfp4ModeReturnsOnlyTwoArrays() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] result = MLXQuant.quantize(w, 32, 4, "mxfp4", null);
      assertEquals(2, result.length);
    }
  }

  @Test
  void quantizeProducesUint32PackedWeightAndFloat32ScalesAndBiases() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] result = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertEquals(3, result.length);
      MLXArray weightQ = result[0];
      assertArrayEquals(new int[] {1, 8}, weightQ.shape());
      assertEquals(DType.UINT32, weightQ.dtype());
      MLXArray scales = result[1];
      assertArrayEquals(new int[] {1, 2}, scales.shape());
      assertEquals(DType.FLOAT32, scales.dtype());
      MLXArray biases = result[2];
      assertArrayEquals(new int[] {1, 2}, biases.shape());
      assertEquals(DType.FLOAT32, biases.dtype());
    }
  }

  @Test
  void dequantizeRoundTripsWithinQuantizationError() {
    try (MLXScope scope = new MLXScope()) {
      float[] weightData = defaultFixture();
      MLXArray w = MLX.array(scope, weightData, new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      MLXArray result = MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, "affine", null, null);
      assertArrayEquals(weightData, result.toFloatArray(), EPS_ROUNDTRIP);
    }
  }

  @Test
  void quantizedMatmulMatchesDequantizedFloatMatmul() {
    try (MLXScope scope = new MLXScope()) {
      float[] fixture = new float[64];
      for (int i = 0; i < 64; i++) {
        fixture[i] = (i % 7 - 3) * 0.3f;
      }
      MLXArray w = MLX.array(scope, fixture, new int[] {1, 64});
      MLXArray x = MLX.array(scope, fixture, new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      MLXArray fused = MLXQuant.quantizedMatmul(x, q[0], q[1], q[2], true, 32, 4, "affine");
      MLXArray dequantized = MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, "affine", null, null);
      MLXArray composed = MLXOps.matmul(x, MLXShape.transpose(dequantized));
      assertArrayEquals(composed.toFloatArray(), fused.toFloatArray(), EPS_EXACT);
    }
  }

  @Test
  void quantizedMatmulWithTransposeFalseMatchesDequantizedFloatMatmulWithoutTranspose() {
    try (MLXScope scope = new MLXScope()) {
      float[] weightData = new float[64 * 64];
      for (int i = 0; i < weightData.length; i++) {
        weightData[i] = (i % 7 - 3) * 0.3f;
      }
      float[] activationData = new float[64];
      for (int i = 0; i < 64; i++) {
        activationData[i] = (i % 5 - 2) * 0.2f;
      }
      MLXArray w = MLX.array(scope, weightData, new int[] {64, 64});
      MLXArray x = MLX.array(scope, activationData, new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      MLXArray fused = MLXQuant.quantizedMatmul(x, q[0], q[1], q[2], false, 32, 4, "affine");
      MLXArray dequantized = MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, "affine", null, null);
      MLXArray composed = MLXOps.matmul(x, dequantized);
      assertArrayEquals(composed.toFloatArray(), fused.toFloatArray(), EPS_EXACT);
    }
  }

  @Test
  void dequantizeWithExplicitFloat16DtypeProducesAFloat16Result() {
    try (MLXScope scope = new MLXScope()) {
      float[] fixture = new float[64];
      for (int i = 0; i < 64; i++) {
        fixture[i] = (i % 7 - 3) * 0.3f;
      }
      MLXArray w = MLX.array(scope, fixture, new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      MLXArray result = MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, "affine", null, DType.FLOAT16);
      assertEquals(DType.FLOAT16, result.dtype());
    }
  }

  @Test
  void quantizeWithNullGlobalScaleAndAbsentGroupSizeBitsSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] result = MLXQuant.quantize(w, null, null, "affine", null);
      assertEquals(3, result.length);
    }
  }

  @Test
  void dequantizeWithBiasesInAChildScopeOfWAllocatesIntoTheChild() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray w = MLX.array(parent, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      try (MLXScope child = parent.newChild()) {
        MLXArray biases = MLX.array(child, q[2].toFloatArray(), q[2].shape());
        MLXArray result = MLXQuant.dequantize(q[0], q[1], biases, 32, 4, "affine", null, null);
        assertSame(child, result.scope());
      }
    }
  }

  /**
   * This PR's round-7 review finding 1: the no-target {@code quantizedMatmul} overload allocates
   * into {@link NativeOps#scopeOf}'s innermost-of-all-operands pick, which leaks into the model
   * scope on the "inverted" layout ({@code x}'s scope an ancestor of the model's) -- see {@code
   * QuantizedLinear}'s own javadoc. This pins the explicit-{@code target} overload itself: passing
   * a {@code target} strictly more descendant than every operand's natural scope must land the
   * result there, not at the operands' own (shallower) scope.
   */
  @Test
  void quantizedMatmulWithExplicitTargetAllocatesIntoTarget() {
    try (MLXScope root = new MLXScope()) {
      float[] fixture = new float[64];
      for (int i = 0; i < 64; i++) {
        fixture[i] = (i % 7 - 3) * 0.3f;
      }
      MLXArray w = MLX.array(root, fixture, new int[] {1, 64});
      MLXArray x = MLX.array(root, fixture, new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);

      try (MLXScope target = root.newChild()) {
        MLXArray result =
            MLXQuant.quantizedMatmul(x, q[0], q[1], q[2], true, 32, 4, "affine", target);
        assertSame(target, result.scope());
      }
    }
  }

  @Test
  void quantizedMatmulRejectsAnUnrelatedTarget() {
    try (MLXScope s1 = new MLXScope();
        MLXScope s2 = new MLXScope()) {
      float[] fixture = new float[64];
      for (int i = 0; i < 64; i++) {
        fixture[i] = (i % 7 - 3) * 0.3f;
      }
      MLXArray w = MLX.array(s1, fixture, new int[] {1, 64});
      MLXArray x = MLX.array(s1, fixture, new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);

      assertThrows(
          IllegalArgumentException.class,
          () -> MLXQuant.quantizedMatmul(x, q[0], q[1], q[2], true, 32, 4, "affine", s2));
    }
  }

  @Test
  void dequantizeRejectsBiasesFromAnUnrelatedScope() {
    try (MLXScope s1 = new MLXScope();
        MLXScope s2 = new MLXScope()) {
      MLXArray w = MLX.array(s1, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      MLXArray biases = MLX.array(s2, q[2].toFloatArray(), q[2].shape());
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXQuant.dequantize(q[0], q[1], biases, 32, 4, "affine", null, null));
    }
  }

  /**
   * A genuine finding beyond this plan's pre-work probes (which only exercised a non-null {@code
   * biases}, e.g. {@link #dequantizeWithBiasesInAChildScopeOfWAllocatesIntoTheChild}): {@code
   * mlx_dequantize} with {@code mode="affine"} unconditionally rejects a null {@code biases} --
   * confirmed against the shipped native error ("{@code [dequantize] Biases must be provided for
   * affine quantization}", `mlx-c/mlx/c/ops.cpp`, from `mlx::core::dequantize` itself, not this
   * facade). {@code biases} stays nullable API surface on {@link MLXQuant#dequantize} regardless
   * (Global Constraint 5: a future non-affine mode may not need it, and this facade cannot verify
   * that either way) -- but under the only mode this codebase exercises, null {@code biases} is not
   * a legal call, so this is the regression pin for that, not the "succeeds" row an earlier reading
   * of the plan assumed.
   */
  @Test
  void dequantizeRejectsNullBiasesUnderAffineMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          MLXException.class,
          () -> MLXQuant.dequantize(q[0], q[1], null, 32, 4, "affine", null, null));
    }
  }

  /**
   * Same finding as {@link #dequantizeRejectsNullBiasesUnderAffineMode}, confirmed directly against
   * {@code mlx_quantized_matmul} rather than assumed from {@code dequantize}'s behavior: {@code
   * mode="affine"} unconditionally rejects a null {@code biases} here too ("{@code
   * [quantized_matmul] Biases must be provided for affine quantization}", `mlx-c/mlx/c/ops.cpp`).
   */
  @Test
  void quantizedMatmulRejectsNullBiasesUnderAffineMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray x = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          MLXException.class,
          () -> MLXQuant.quantizedMatmul(x, q[0], q[1], null, true, 32, 4, "affine"));
    }
  }

  /**
   * This PR's round-11 review finding 4: {@code quantize}/{@code dequantize}/{@code
   * quantizedMatmul} document {@code w}/{@code scales}/{@code x} as non-null, but (before this fix)
   * left them unchecked -- a null value reached {@code w.handle()}/{@code scales.handle()}/{@code
   * x.handle()} as a bare {@code NullPointerException} naming neither the operation nor the
   * parameter, unlike {@code mode}'s own null handling ({@code checkMode} already gives a named
   * error for that one). This pins the named error for each documented-non-null array operand --
   * {@code biases} deliberately excluded throughout, since it stays legitimate, nullable API
   * surface (see {@link #dequantizeRejectsNullBiasesUnderAffineMode}/{@link
   * #quantizedMatmulRejectsNullBiasesUnderAffineMode}, which already pin native's own well-named
   * rejection of a null {@code biases} under {@code mode="affine"} specifically).
   */
  @Test
  void quantizeRejectsANullW() {
    assertThrows(NullPointerException.class, () -> MLXQuant.quantize(null, 32, 4, "affine", null));
  }

  @Test
  void dequantizeRejectsANullWOrScales() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          NullPointerException.class,
          () -> MLXQuant.dequantize(null, q[1], q[2], 32, 4, "affine", null, null));
      assertThrows(
          NullPointerException.class,
          () -> MLXQuant.dequantize(q[0], null, q[2], 32, 4, "affine", null, null));
    }
  }

  @Test
  void quantizedMatmulRejectsANullXWOrScales() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray x = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          NullPointerException.class,
          () -> MLXQuant.quantizedMatmul(null, q[0], q[1], q[2], true, 32, 4, "affine"));
      assertThrows(
          NullPointerException.class,
          () -> MLXQuant.quantizedMatmul(x, null, q[1], q[2], true, 32, 4, "affine"));
      assertThrows(
          NullPointerException.class,
          () -> MLXQuant.quantizedMatmul(x, q[0], null, q[2], true, 32, 4, "affine"));
    }
  }

  /**
   * This PR's round-11 review finding 4 fixed x/w/scales on the explicit-target {@code
   * quantizedMatmul} overload but missed {@code target} itself, which reaches {@link
   * se.alipsa.jmlx.memory.MLXScope#innermost} and NPEs out of its own {@code depth()} call with no
   * named message -- confirmed by round-12's own review.
   */
  @Test
  void quantizedMatmulRejectsANullTarget() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray x = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          NullPointerException.class,
          () -> MLXQuant.quantizedMatmul(x, q[0], q[1], q[2], true, 32, 4, "affine", null));
    }
  }

  @Test
  void quantizeRejectsAnUnsupportedMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      assertThrows(IllegalArgumentException.class, () -> MLXQuant.quantize(w, 32, 4, "int4", null));
    }
  }

  @Test
  void dequantizeRejectsAnUnsupportedMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, "int4", null, null));
    }
  }

  @Test
  void quantizedMatmulRejectsAnUnsupportedMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray x = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXQuant.quantizedMatmul(x, q[0], q[1], q[2], true, 32, 4, "int4"));
    }
  }

  @Test
  void quantizeRejectsANullMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      assertThrows(IllegalArgumentException.class, () -> MLXQuant.quantize(w, 32, 4, null, null));
    }
  }

  @Test
  void dequantizeRejectsANullMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXQuant.dequantize(q[0], q[1], q[2], 32, 4, null, null, null));
    }
  }

  @Test
  void quantizedMatmulRejectsANullMode() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray x = MLX.array(scope, defaultFixture(), new int[] {1, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXQuant.quantizedMatmul(x, q[0], q[1], q[2], true, 32, 4, null));
    }
  }
}
