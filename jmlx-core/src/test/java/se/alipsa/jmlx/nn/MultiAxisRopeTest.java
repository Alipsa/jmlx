package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Strict tests for the frozen WP4 multi-axis RoPE component (req/plans/phase7-3b-reference-
 * findings.md §6.9). The float32-comparison cases are tagged {@code full-float32} so they select
 * into {@code :jmlx-core:float32GoldenTest} with {@code MLX_ENABLE_TF32=0}; the rejection and
 * structural cases run in the ordinary {@code :jmlx-core:test} task.
 */
@EnabledIfNativeAvailable
class MultiAxisRopeTest {

  private static final int[] PINNED_SECTIONS = {24, 20, 20};
  private static final float PINNED_THETA = 5_000_000f;

  // ---------------------------------------------------------------------
  // structural / rejection cases (ordinary test task, TF32=1)
  // ---------------------------------------------------------------------

  @Test
  void gettersExposeThetaSectionsAndRotaryDims() {
    MultiAxisRope rope = new MultiAxisRope(PINNED_THETA, PINNED_SECTIONS);
    assertEquals(PINNED_THETA, rope.theta());
    assertArrayEquals(PINNED_SECTIONS, rope.mropeSection());
    assertEquals(128, rope.rotaryDims());
    int[] returned = rope.mropeSection();
    returned[0] = 0;
    assertArrayEquals(PINNED_SECTIONS, rope.mropeSection());
  }

  @Test
  void prepareReturnsFloat32TablesBroadcastableOverHeads() {
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(10_000f, new int[] {2, 1, 2});
      MultiAxisRotation rotation = rope.prepare(scope, 2, 3, rows3(6));
      assertArrayEquals(new int[] {2, 1, 3, 10}, rotation.cos().shape());
      assertArrayEquals(new int[] {2, 1, 3, 10}, rotation.sin().shape());
      assertEquals(DType.FLOAT32, rotation.cos().dtype());
      assertEquals(DType.FLOAT32, rotation.sin().dtype());
      assertEquals(10, rotation.rotaryDims());
      MLXArray x = MLX.array(scope, values(2 * 1 * 3 * 16, 9), new int[] {2, 1, 3, 16});
      assertArrayEquals(new int[] {2, 1, 3, 16}, rotation.rotate(x).shape());
    }
  }

  @Test
  void constructorRejectsInvalidThetaAndSections() {
    assertThrows(IllegalArgumentException.class, () -> new MultiAxisRope(0f, PINNED_SECTIONS));
    assertThrows(IllegalArgumentException.class, () -> new MultiAxisRope(-1f, PINNED_SECTIONS));
    assertThrows(
        IllegalArgumentException.class, () -> new MultiAxisRope(Float.NaN, PINNED_SECTIONS));
    assertThrows(
        IllegalArgumentException.class,
        () -> new MultiAxisRope(Float.POSITIVE_INFINITY, PINNED_SECTIONS));
    assertThrows(
        IllegalArgumentException.class,
        () -> new MultiAxisRope(Float.NEGATIVE_INFINITY, PINNED_SECTIONS));
    assertThrows(IllegalArgumentException.class, () -> new MultiAxisRope(1000f, null));
    assertThrows(
        IllegalArgumentException.class, () -> new MultiAxisRope(1000f, new int[] {24, 20}));
    assertThrows(
        IllegalArgumentException.class, () -> new MultiAxisRope(1000f, new int[] {24, 20, 20, 4}));
    assertThrows(
        IllegalArgumentException.class, () -> new MultiAxisRope(1000f, new int[] {24, -1, 20}));
    assertThrows(
        IllegalArgumentException.class, () -> new MultiAxisRope(1000f, new int[] {0, 0, 0}));
    assertThrows(
        IllegalArgumentException.class,
        () -> new MultiAxisRope(1000f, new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE, 0}));
  }

  @Test
  void prepareRejectsInvalidShapeAndPositions() {
    MultiAxisRope rope = new MultiAxisRope(1000f, new int[] {2, 1, 2});
    try (MLXScope scope = new MLXScope()) {
      assertThrows(IllegalArgumentException.class, () -> rope.prepare(scope, 0, 4, rows3(4)));
      assertThrows(IllegalArgumentException.class, () -> rope.prepare(scope, 1, 0, rows3(0)));
      assertThrows(IllegalArgumentException.class, () -> rope.prepare(scope, 1, 4, null));
      assertThrows(
          IllegalArgumentException.class,
          () -> rope.prepare(scope, 1, 4, new int[][] {row(4), row(4)}));
      assertThrows(
          IllegalArgumentException.class,
          () -> rope.prepare(scope, 1, 4, new int[][] {row(4), row(4), row(4), row(4)}));
      assertThrows(
          IllegalArgumentException.class,
          () -> rope.prepare(scope, 1, 4, new int[][] {row(3), row(4), row(4)}));
      assertThrows(
          IllegalArgumentException.class,
          () -> rope.prepare(scope, 1, 4, new int[][] {null, row(4), row(4)}));
      int[] negative = row(4);
      negative[1] = -1;
      assertThrows(
          IllegalArgumentException.class,
          () -> rope.prepare(scope, 1, 4, new int[][] {row(4), negative, row(4)}));
    }
  }

  @Test
  void rotateRejectsInvalidInputs() {
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(10_000f, new int[] {2, 1, 2}); // rotaryDims 10
      MultiAxisRotation rotation = rope.prepare(scope, 1, 2, rows3(2));
      IllegalArgumentException tooNarrow =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  rotation.rotate(
                      MLX.array(scope, values(1 * 1 * 2 * 8, 0), new int[] {1, 1, 2, 8})));
      assertTrue(tooNarrow.getMessage().contains("rotaryDims exceeds head dimension"));
      assertThrows(
          IllegalArgumentException.class,
          () -> rotation.rotate(MLX.array(scope, values(1 * 1 * 2 * 16, 1), new int[] {1, 1, 32})));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              rotation.rotate(
                  MLX.array(scope, values(2 * 1 * 2 * 16, 2), new int[] {2, 1, 2, 16})));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              rotation.rotate(
                  MLX.array(scope, values(1 * 1 * 3 * 16, 3), new int[] {1, 1, 3, 16})));
      assertThrows(IllegalArgumentException.class, () -> rotation.rotate(null));
    }
  }

  // ---------------------------------------------------------------------
  // float32-comparison cases (full-float32 tag, MLX_ENABLE_TF32=0)
  // ---------------------------------------------------------------------

  @Test
  @Tag("full-float32")
  void equalAxesReduceToOneDimensionalRoPE() {
    int[] axis = {2, 5, 0, 7};
    int[] sections = {2, 2, 2};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(10_000f, sections);
      float[] x = values(1 * 1 * 4 * 12, 0);
      MultiAxisRotation rotation = rope.prepare(scope, 1, 4, new int[][] {axis, axis, axis});
      MLXArray arrayX = MLX.array(scope, x, new int[] {1, 1, 4, 12});
      assertArrayEquals(
          expectedRotation(10_000, sections, 1, 1, 4, x, 12, axis, axis, axis),
          rotation.rotate(arrayX).toFloatArray(),
          1e-5f);
      float[][] tables = expectedTables(10_000, sections, axis, axis, axis);
      assertArrayEquals(tables[0], rotation.cos().toFloatArray(), 1e-5f);
      assertArrayEquals(tables[1], rotation.sin().toFloatArray(), 1e-5f);
    }
  }

  @Test
  @Tag("full-float32")
  void pinnedOwnerPatternWithIndexCodedAxes() {
    int[] rowT = {0, 1, 2, 3};
    int[] rowH = {10, 11, 12, 13};
    int[] rowW = {20, 21, 22, 23};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(PINNED_THETA, PINNED_SECTIONS);
      float[] x = values(1 * 1 * 4 * 128, 0);
      MultiAxisRotation rotation = rope.prepare(scope, 1, 4, new int[][] {rowT, rowH, rowW});
      MLXArray arrayX = MLX.array(scope, x, new int[] {1, 1, 4, 128});
      assertArrayEquals(
          expectedRotation(5_000_000, PINNED_SECTIONS, 1, 1, 4, x, 128, rowT, rowH, rowW),
          rotation.rotate(arrayX).toFloatArray(),
          1e-5f);

      // Per-frequency owner discrimination at token 0 (t=0, h=10, w=20). The candidates are
      // derived from the component's own float32 invFreq order so this isolates the owner
      // selection from the frequency derivation (the latter is pinned separately by the
      // double-precision comparisons above and the pinned-venv probe).
      double[] invFreq = invFrequenciesOptionB(PINNED_THETA, 64);
      float[] sinTable = rotation.sin().toFloatArray();
      double minimumSeparation = Double.MAX_VALUE;
      for (int f = 0; f < 64; f++) {
        double[] candidates = {
          Math.sin(0.0 * invFreq[f]), Math.sin(10.0 * invFreq[f]), Math.sin(20.0 * invFreq[f])
        };
        int owner = ownerAxis(f, PINNED_SECTIONS);
        for (int axis = 0; axis < 3; axis++) {
          if (axis != owner) {
            minimumSeparation =
                Math.min(minimumSeparation, Math.abs(candidates[axis] - candidates[owner]));
          }
        }
        float actual = sinTable[f];
        assertTrue(
            Math.abs(actual - candidates[owner]) <= 5e-7,
            "frequency "
                + f
                + " should use the "
                + "THW".charAt(owner)
                + " axis, actual "
                + actual
                + " vs candidate "
                + candidates[owner]);
        for (int axis = 0; axis < 3; axis++) {
          if (axis != owner) {
            assertTrue(
                Math.abs(actual - candidates[axis]) >= 1e-6,
                "frequency " + f + " must not use the " + "THW".charAt(axis) + " axis");
          }
        }
      }
      assertTrue(minimumSeparation >= 1e-6, "case is not discriminable: " + minimumSeparation);
    }
  }

  @Test
  @Tag("full-float32")
  void smallSectionsExerciseEveryOwnerAtConstantIndexCodedAxes() {
    // [2,1,2]: owners T, H, W, T, T at constant index-coded coordinates.
    int[] sections = {2, 1, 2};
    int[] rowT = {1, 1};
    int[] rowH = {2, 2};
    int[] rowW = {4, 4};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(10_000f, sections);
      float[] x = values(1 * 1 * 2 * 10, 0);
      MultiAxisRotation rotation = rope.prepare(scope, 1, 2, new int[][] {rowT, rowH, rowW});
      MLXArray arrayX = MLX.array(scope, x, new int[] {1, 1, 2, 10});
      assertArrayEquals(
          expectedRotation(10_000, sections, 1, 1, 2, x, 10, rowT, rowH, rowW),
          rotation.rotate(arrayX).toFloatArray(),
          1e-5f);
    }
    // [1,1,1]: owners T, H, W on every frequency.
    sections = new int[] {1, 1, 1};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(1000f, sections);
      float[] x = values(1 * 1 * 2 * 6, 5);
      MultiAxisRotation rotation = rope.prepare(scope, 1, 2, new int[][] {rowT, rowH, rowW});
      MLXArray arrayX = MLX.array(scope, x, new int[] {1, 1, 2, 6});
      assertArrayEquals(
          expectedRotation(1000, sections, 1, 1, 2, x, 6, rowT, rowH, rowW),
          rotation.rotate(arrayX).toFloatArray(),
          1e-5f);
    }
  }

  @Test
  @Tag("full-float32")
  void decodePositionsWithNegativeDeltaAndBatchedRows() {
    // B=1, T=1: cache position 82 plus delta -56 gives coordinate 26 on all three axes -- the
    // pinned 2B decode step.
    int[] row = {26};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(PINNED_THETA, PINNED_SECTIONS);
      float[] x = values(1 * 1 * 1 * 128, 0);
      MultiAxisRotation rotation = rope.prepare(scope, 1, 1, new int[][] {row, row, row});
      MLXArray arrayX = MLX.array(scope, x, new int[] {1, 1, 1, 128});
      assertArrayEquals(
          expectedRotation(5_000_000, PINNED_SECTIONS, 1, 1, 1, x, 128, row, row, row),
          rotation.rotate(arrayX).toFloatArray(),
          1e-5f);
    }
    // B=2, T=2 with per-row positions (row 0 low, row 1 at the prefill-envelope edge).
    int[] rowT = {0, 1, 30, 31};
    int[] rowH = {4, 5, 26, 27};
    int[] rowW = {8, 9, 22, 23};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(PINNED_THETA, PINNED_SECTIONS);
      float[] x = values(2 * 1 * 2 * 128, 0);
      MultiAxisRotation rotation = rope.prepare(scope, 2, 2, new int[][] {rowT, rowH, rowW});
      MLXArray arrayX = MLX.array(scope, x, new int[] {2, 1, 2, 128});
      assertArrayEquals(
          expectedRotation(5_000_000, PINNED_SECTIONS, 2, 1, 2, x, 128, rowT, rowH, rowW),
          rotation.rotate(arrayX).toFloatArray(),
          1e-5f);
    }
  }

  @Test
  @Tag("full-float32")
  void partialRotaryTailPassesThroughBitIdentically() {
    int[] sections = {2, 1, 2}; // rotaryDims 10
    int headDim = 16;
    int[] rowT = {1, 1};
    int[] rowH = {2, 2};
    int[] rowW = {4, 4};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(10_000f, sections);
      float[] x = values(1 * 1 * 2 * headDim, 0);
      for (int t = 0; t < 2; t++) {
        for (int j = 0; j < headDim - 10; j++) {
          x[t * headDim + 10 + j] = 90 + j; // distinctive, non-deterministic tail values
        }
      }
      MultiAxisRotation rotation = rope.prepare(scope, 1, 2, new int[][] {rowT, rowH, rowW});
      MLXArray arrayX = MLX.array(scope, x, new int[] {1, 1, 2, headDim});
      float[] actual = rotation.rotate(arrayX).toFloatArray();
      assertArrayEquals(
          expectedRotation(10_000, sections, 1, 1, 2, x, headDim, rowT, rowH, rowW), actual, 1e-5f);
      for (int t = 0; t < 2; t++) {
        for (int j = 0; j < headDim - 10; j++) {
          assertEquals(
              x[t * headDim + 10 + j],
              actual[t * headDim + 10 + j],
              "tail element (" + t + ", " + (10 + j) + ") must pass through bit-identically");
        }
      }
    }
  }

  @Test
  @Tag("full-float32")
  void rotationCastsCosSinToInputDtype() {
    int[] sections = {2, 1, 2};
    int[] rowT = {1, 1};
    int[] rowH = {2, 2};
    int[] rowW = {4, 4};
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(10_000f, sections);
      MultiAxisRotation rotation = rope.prepare(scope, 1, 2, new int[][] {rowT, rowH, rowW});
      float[] x32 = values(1 * 1 * 2 * 10, 0);
      MLXArray float32 = MLX.array(scope, x32, new int[] {1, 1, 2, 10});
      MLXArray out32 = rotation.rotate(float32);
      assertEquals(DType.FLOAT32, out32.dtype());
      assertArrayEquals(
          expectedRotation(10_000, sections, 1, 1, 2, x32, 10, rowT, rowH, rowW),
          out32.toFloatArray(),
          1e-5f);

      MLXArray bfloat = MLX.astype(float32, DType.BFLOAT16);
      MLXArray outBfloat = rotation.rotate(bfloat);
      assertEquals(DType.BFLOAT16, outBfloat.dtype());
      assertArrayEquals(
          roundedToDtype(
              expectedRotation(
                  10_000, sections, 1, 1, 2, bfloat.toFloatArray(), 10, rowT, rowH, rowW),
              DType.BFLOAT16),
          outBfloat.toFloatArray(),
          0.05f);

      MLXArray half = MLX.astype(float32, DType.FLOAT16);
      MLXArray outHalf = rotation.rotate(half);
      assertEquals(DType.FLOAT16, outHalf.dtype());
      assertArrayEquals(
          roundedToDtype(
              expectedRotation(
                  10_000, sections, 1, 1, 2, half.toFloatArray(), 10, rowT, rowH, rowW),
              DType.FLOAT16),
          outHalf.toFloatArray(),
          0.02f);
    }
  }

  // ---------------------------------------------------------------------
  // helpers
  // ---------------------------------------------------------------------

  /** Distinct deterministic values in [-1, 1] with a per-case seed offset. */
  private static float[] values(int n, int seed) {
    float[] data = new float[n];
    for (int j = 0; j < n; j++) {
      data[j] = (((j + seed) % 13) - 6) / 6.0f;
    }
    return data;
  }

  private static int[] row(int length) {
    int[] data = new int[length];
    for (int i = 0; i < length; i++) {
      data[i] = i;
    }
    return data;
  }

  private static int[][] rows3(int length) {
    return new int[][] {row(length), row(length), row(length)};
  }

  /**
   * The reference's owner of frequency index {@code f} for the interleaved layout: H owns {@code f
   * % 3 == 1} with {@code f < 1 + 3 * s1}, W owns {@code f % 3 == 2} with {@code f < 2 + 3 * s2},
   * everything else stays T.
   */
  private static int ownerAxis(int f, int[] sections) {
    if (f % 3 == 1 && f < 1 + 3 * sections[1]) {
      return 1;
    }
    if (f % 3 == 2 && f < 2 + 3 * sections[2]) {
      return 2;
    }
    return 0;
  }

  /**
   * The component's float32 invFreq derivation order (the documented option B, see {@code
   * MultiAxisRope.deriveInvFrequencies}) recomputed here as a double-precision stand-in for the
   * owner-candidate discrimination only.
   */
  private static double[] invFrequenciesOptionB(float theta, int half) {
    int rotaryDims = 2 * half;
    double[] invFreq = new double[half];
    for (int i = 0; i < half; i++) {
      float exponent = (float) (2 * i) / (float) rotaryDims;
      invFreq[i] = 1.0 / (float) Math.pow(theta, exponent);
    }
    return invFreq;
  }

  /**
   * Java double-precision-derived float32 reference: float32 cos/sin tables of shape {@code
   * [tokens, rotaryDims]} (the {@code cat((freqs, freqs), -1)} layout) for the given flattened axis
   * position rows.
   */
  private static float[][] expectedTables(
      double theta, int[] sections, int[] rowT, int[] rowH, int[] rowW) {
    int half = sections[0] + sections[1] + sections[2];
    int rotaryDims = 2 * half;
    int tokens = rowT.length;
    double[] invFreq = new double[half];
    for (int i = 0; i < half; i++) {
      invFreq[i] = Math.pow(theta, -2.0 * i / (2.0 * half));
    }
    float[] cos = new float[tokens * rotaryDims];
    float[] sin = new float[tokens * rotaryDims];
    for (int t = 0; t < tokens; t++) {
      for (int f = 0; f < half; f++) {
        int axis = ownerAxis(f, sections);
        int position = axis == 0 ? rowT[t] : axis == 1 ? rowH[t] : rowW[t];
        double angle = position * invFreq[f];
        float c = (float) Math.cos(angle);
        float s = (float) Math.sin(angle);
        cos[t * rotaryDims + f] = c;
        cos[t * rotaryDims + half + f] = c;
        sin[t * rotaryDims + f] = s;
        sin[t * rotaryDims + half + f] = s;
      }
    }
    return new float[][] {cos, sin};
  }

  /**
   * The frozen composition in double precision: for a flat {@code [batch, heads, sequence,
   * headDim]} input, {@code out = x * cos + rotate_half(x) * sin} on the {@code rotaryDims} prefix
   * ({@code rotate_half = cat((−x2, x1))}) with the tail passing through unchanged.
   */
  private static float[] expectedRotation(
      double theta,
      int[] sections,
      int batch,
      int heads,
      int sequence,
      float[] x,
      int headDim,
      int[] rowT,
      int[] rowH,
      int[] rowW) {
    float[][] tables = expectedTables(theta, sections, rowT, rowH, rowW);
    int half = sections[0] + sections[1] + sections[2];
    int rotaryDims = 2 * half;
    float[] out = new float[batch * heads * sequence * headDim];
    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < heads; h++) {
        for (int t = 0; t < sequence; t++) {
          int base = (b * heads + h) * sequence * headDim + t * headDim;
          int table = (b * sequence + t) * rotaryDims;
          for (int d = 0; d < headDim; d++) {
            if (d >= rotaryDims) {
              out[base + d] = x[base + d];
              continue;
            }
            int f = d % half;
            float c = tables[0][table + f];
            float s = tables[1][table + f];
            if (d < half) {
              out[base + d] = x[base + d] * c - x[base + d + half] * s;
            } else {
              out[base + d] = x[base + d] * c + x[base + d - half] * s;
            }
          }
        }
      }
    }
    return out;
  }

  private static float[] roundedToDtype(float[] values, DType dtype) {
    float[] out = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = dtype == DType.BFLOAT16 ? toBf16(values[i]) : toF16(values[i]);
    }
    return out;
  }

  /**
   * Round-to-nearest-even from float32 to bfloat16 at the bit level: a bfloat16 value is the top 16
   * bits of the float32, so the low 16 bits are rounded with the half-ulp plus retained-LSB carry
   * trick.
   */
  private static float toBf16(float value) {
    long bits = Float.floatToRawIntBits(value) & 0xFFFFFFFFL;
    long lsb = (bits >>> 15) & 1;
    long rounded = (bits + 0x7FFFL + lsb) & 0xFFFF0000L;
    return Float.intBitsToFloat((int) rounded);
  }

  /**
   * Round-to-nearest-even from float32 to float16 via the exact double promotion: the low 48 double
   * bits are rounded with the same half-ulp plus retained-LSB carry trick, sign handled separately.
   */
  private static float toF16(float value) {
    long bits = Double.doubleToRawLongBits((double) value);
    boolean negative = bits < 0;
    long magnitude = bits & 0x7FFFFFFFFFFFFFFFL;
    long lsb = (magnitude >>> 48) & 1;
    long rounded = (magnitude + 0x7FFFFFFFFFFFL + lsb) & 0x7FFF800000000000L;
    if (negative) {
      rounded |= 0x8000000000000000L;
    }
    return (float) Double.longBitsToDouble(rounded);
  }
}
