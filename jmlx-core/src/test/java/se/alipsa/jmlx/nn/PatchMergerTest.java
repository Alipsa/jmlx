package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Strict tests for the WP5 {@link PatchMerger} (req/plans/phase7-3b-reference-findings.md §11.2
 * "Merger ordering"). Runs in the ordinary {@code :jmlx-core:test} task. The numeric cases
 * hand-compute the expected output (LayerNorm with population variance and eps 1e-6, exact erf
 * GELU) in double in the test and compare with a 1e-5 tolerance; the two flag placements use the
 * same linear weights so their different normalization width and order are the only structural
 * source of output difference.
 */
@EnabledIfNativeAvailable
class PatchMergerTest {

  private static final float EPS = 1e-6f;

  // merge 2 (merge^2 4), per-patch hidden 2, m = hidden * merge^2 = 8, outHidden 2. The input is
  // N = 4 rows of width hidden, so exactly one merged row of width m.
  private static final int MERGE = 2;
  private static final int N = 4;
  private static final int HIDDEN = 2;
  private static final int M = HIDDEN * MERGE * MERGE;
  private static final int OUT = 2;

  // The shared MLP weights: all 64 fc1 entries distinct (0.01 * (i + 1)) so a wrong row/column
  // mapping is observable; the same arrays are used by both flag placements.
  private static final float[] FC1_WEIGHT = {
    0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f,
    0.09f, 0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f,
    0.17f, 0.18f, 0.19f, 0.20f, 0.21f, 0.22f, 0.23f, 0.24f,
    0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.30f, 0.31f, 0.32f,
    0.33f, 0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f, 0.40f,
    0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f, 0.47f, 0.48f,
    0.49f, 0.50f, 0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f,
    0.57f, 0.58f, 0.59f, 0.60f, 0.61f, 0.62f, 0.63f, 0.64f
  };
  private static final float[] FC1_BIAS = {
    0.05f, -0.1f, 0.15f, -0.05f, 0.02f, -0.08f, 0.11f, -0.03f
  };
  private static final float[] FC2_WEIGHT = {
    0.2f, -0.1f, 0.3f, 0.1f, -0.2f, 0.3f, -0.1f, 0.2f,
    -0.15f, 0.25f, -0.05f, 0.1f, 0.05f, -0.2f, 0.15f, -0.1f
  };
  private static final float[] FC2_BIAS = {0.1f, -0.2f};

  @Test
  void mainPlacementMatchesHandComputedExpectation() {
    float[] normWeight = {2.0f, 2.0f};
    float[] normBias = {0.25f, 0.25f};
    float[] x = {0.5f, -0.25f, -0.5f, 0.75f, 1.0f, 0.25f, -0.75f, -1.0f};
    try (MLXScope scope = new MLXScope()) {
      PatchMerger merger =
          new PatchMerger(
              scope,
              MERGE,
              false,
              MLX.array(scope, normWeight, new int[] {HIDDEN}),
              MLX.array(scope, normBias, new int[] {HIDDEN}),
              MLX.array(scope, FC1_WEIGHT, new int[] {M, M}),
              MLX.array(scope, FC1_BIAS, new int[] {M}),
              MLX.array(scope, FC2_WEIGHT, new int[] {OUT, M}),
              MLX.array(scope, FC2_BIAS, new int[] {OUT}));
      assertEquals(MERGE, merger.spatialMergeSize());
      assertFalse(merger.usePostShuffleNorm());
      assertEquals(HIDDEN, merger.hiddenSize());
      assertEquals(OUT, merger.outHiddenSize());
      MLXArray input = MLX.array(scope, x, new int[] {N, HIDDEN});
      MLXArray outArray = merger.forward(input);
      assertArrayEquals(new int[] {N / (MERGE * MERGE), OUT}, outArray.shape());
      float[] out = outArray.toFloatArray();
      assertArrayEqualsLoose(expectedMerger(false, x, normWeight, normBias), out, 1e-5f);
    }
  }

  @Test
  void postShufflePlacementMatchesHandComputedExpectationAndDiffersFromMain() {
    float[] normWeight = {1.5f, 2.0f, 2.5f, 3.0f, 2.0f, 1.5f, 3.0f, 2.5f};
    float[] normBias = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float[] x = {0.5f, -0.25f, -0.5f, 0.75f, 1.0f, 0.25f, -0.75f, -1.0f};
    float[] mainNormWeight = {2.0f, 2.0f};
    float[] mainNormBias = {0.25f, 0.25f};
    try (MLXScope scope = new MLXScope()) {
      PatchMerger merger =
          new PatchMerger(
              scope,
              MERGE,
              true,
              MLX.array(scope, normWeight, new int[] {M}),
              MLX.array(scope, normBias, new int[] {M}),
              MLX.array(scope, FC1_WEIGHT, new int[] {M, M}),
              MLX.array(scope, FC1_BIAS, new int[] {M}),
              MLX.array(scope, FC2_WEIGHT, new int[] {OUT, M}),
              MLX.array(scope, FC2_BIAS, new int[] {OUT}));
      assertTrue(merger.usePostShuffleNorm());
      MLXArray input = MLX.array(scope, x, new int[] {N, HIDDEN});
      MLXArray outArray = merger.forward(input);
      assertArrayEquals(new int[] {N / (MERGE * MERGE), OUT}, outArray.shape());
      float[] out = outArray.toFloatArray();
      assertArrayEqualsLoose(expectedMerger(true, x, normWeight, normBias), out, 1e-5f);
      // The same linear weights with the main placement must produce a different output: the
      // normalization runs over the per-patch rows instead of the merged row, which is the
      // observable fingerprint of the flag.
      PatchMerger mainMerger =
          new PatchMerger(
              scope,
              MERGE,
              false,
              MLX.array(scope, mainNormWeight, new int[] {HIDDEN}),
              MLX.array(scope, mainNormBias, new int[] {HIDDEN}),
              MLX.array(scope, FC1_WEIGHT, new int[] {M, M}),
              MLX.array(scope, FC1_BIAS, new int[] {M}),
              MLX.array(scope, FC2_WEIGHT, new int[] {OUT, M}),
              MLX.array(scope, FC2_BIAS, new int[] {OUT}));
      float[] mainOut = mainMerger.forward(input).toFloatArray();
      assertArrayEqualsLoose(
          expectedMerger(false, x, mainNormWeight, mainNormBias), mainOut, 1e-5f);
      double maxDiff = 0;
      for (int i = 0; i < out.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(out[i] - mainOut[i]));
      }
      assertTrue(
          maxDiff > 1e-6, "norm placement must be observable in the output, diff " + maxDiff);
    }
  }

  @Test
  void constructorRejectsInvalidGeometry() {
    float[] goodNorm2 = {1.0f, 1.0f};
    float[] goodNorm8 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float[] goodFc1 = new float[M * M];
    float[] goodFc1Bias = new float[M];
    float[] goodFc2 = new float[OUT * M];
    float[] goodFc2Bias = new float[OUT];
    try (MLXScope scope = new MLXScope()) {
      MLXArray norm2 = MLX.array(scope, goodNorm2, new int[] {2});
      MLXArray fc1 = MLX.array(scope, goodFc1, new int[] {M, M});
      MLXArray fc1Bias = MLX.array(scope, goodFc1Bias, new int[] {M});
      MLXArray fc2 = MLX.array(scope, goodFc2, new int[] {OUT, M});
      MLXArray fc2Bias = MLX.array(scope, goodFc2Bias, new int[] {OUT});
      // spatialMergeSize not positive
      assertThrows(
          IllegalArgumentException.class,
          () -> new PatchMerger(scope, 0, false, norm2, norm2, fc1, fc1Bias, fc2, fc2Bias));
      // fc1 not square
      MLXArray fc1Rect = MLX.array(scope, new float[6], new int[] {2, 3});
      assertThrows(
          IllegalArgumentException.class,
          () -> new PatchMerger(scope, MERGE, false, norm2, norm2, fc1Rect, fc1Bias, fc2, fc2Bias));
      // m not divisible by merge^2 (merge 3 -> merge^2 9; m = 8)
      assertThrows(
          IllegalArgumentException.class,
          () -> new PatchMerger(scope, 3, false, norm2, norm2, fc1, fc1Bias, fc2, fc2Bias));
      // norm dim mismatch for the main placement (expects hidden = 2, given 8)
      MLXArray norm8 = MLX.array(scope, goodNorm8, new int[] {8});
      assertThrows(
          IllegalArgumentException.class,
          () -> new PatchMerger(scope, MERGE, false, norm8, norm8, fc1, fc1Bias, fc2, fc2Bias));
      // norm dim mismatch for the post-shuffle placement (expects m = 8, given 2)
      assertThrows(
          IllegalArgumentException.class,
          () -> new PatchMerger(scope, MERGE, true, norm2, norm2, fc1, fc1Bias, fc2, fc2Bias));
      // fc2 input dim mismatch (expects m = 8, given 3)
      MLXArray fc2Bad = MLX.array(scope, new float[OUT * 3], new int[] {OUT, 3});
      assertThrows(
          IllegalArgumentException.class,
          () -> new PatchMerger(scope, MERGE, false, norm2, norm2, fc1, fc1Bias, fc2Bad, fc2Bias));
      // fc1 bias length mismatch
      MLXArray fc1BiasShort = MLX.array(scope, new float[M - 1], new int[] {M - 1});
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new PatchMerger(scope, MERGE, false, norm2, norm2, fc1, fc1BiasShort, fc2, fc2Bias));
    }
  }

  @Test
  void forwardRejectsInvalidRows() {
    float[] normWeight = {1.0f, 1.0f};
    float[] fc1Weight = new float[M * M];
    float[] fc1Bias = new float[M];
    float[] fc2Weight = new float[OUT * M];
    float[] fc2Bias = new float[OUT];
    try (MLXScope scope = new MLXScope()) {
      PatchMerger merger =
          new PatchMerger(
              scope,
              MERGE,
              false,
              MLX.array(scope, normWeight, new int[] {2}),
              MLX.array(scope, normWeight, new int[] {2}),
              MLX.array(scope, fc1Weight, new int[] {M, M}),
              MLX.array(scope, fc1Bias, new int[] {M}),
              MLX.array(scope, fc2Weight, new int[] {OUT, M}),
              MLX.array(scope, fc2Bias, new int[] {OUT}));
      // N = 6 is not divisible by merge^2 = 4
      MLXArray six = MLX.array(scope, new float[6 * HIDDEN], new int[] {6, HIDDEN});
      IllegalArgumentException notDivisible =
          assertThrows(IllegalArgumentException.class, () -> merger.forward(six));
      assertTrue(notDivisible.getMessage().contains("merge^2"));
      // wrong hidden size
      MLXArray wrongHidden = MLX.array(scope, new float[N * 3], new int[] {N, 3});
      assertThrows(IllegalArgumentException.class, () -> merger.forward(wrongHidden));
      // wrong rank
      MLXArray rank3 = MLX.array(scope, new float[2 * N * HIDDEN], new int[] {2, N, HIDDEN});
      assertThrows(IllegalArgumentException.class, () -> merger.forward(rank3));
      assertThrows(IllegalArgumentException.class, () -> merger.forward(null));
    }
  }

  /**
   * The hand-computed merger output in double: population-variance LayerNorm (eps 1e-6, same
   * placement rule as the component -- per-patch rows of width {@code HIDDEN} for the main
   * placement, the merged row of width {@code M} for the post-shuffle placement), then {@code
   * fc2(GELU(fc1(v)))} with the exact erf GELU.
   */
  private static double[] expectedMerger(
      boolean postShuffle, float[] x, float[] normWeight, float[] normBias) {
    int n = x.length / HIDDEN;
    int merged = n / (MERGE * MERGE);
    double[][] rows = new double[merged][];
    for (int r = 0; r < merged; r++) {
      if (postShuffle) {
        double[] v = new double[M];
        for (int i = 0; i < M; i++) {
          v[i] = x[r * M + i];
        }
        rows[r] = layerNorm(v, normWeight, normBias, M);
      } else {
        double[] v = new double[M];
        for (int p = 0; p < MERGE * MERGE; p++) {
          double[] row = new double[HIDDEN];
          for (int i = 0; i < HIDDEN; i++) {
            row[i] = x[(r * MERGE * MERGE + p) * HIDDEN + i];
          }
          double[] normalized = layerNorm(row, normWeight, normBias, HIDDEN);
          System.arraycopy(normalized, 0, v, p * HIDDEN, HIDDEN);
        }
        rows[r] = v;
      }
    }
    double[][] out = new double[merged][OUT];
    for (int r = 0; r < merged; r++) {
      double[] h1 = new double[M];
      for (int i = 0; i < M; i++) {
        double sum = FC1_BIAS[i];
        for (int j = 0; j < M; j++) {
          sum += FC1_WEIGHT[i * M + j] * rows[r][j];
        }
        h1[i] = gelu(sum);
      }
      for (int i = 0; i < OUT; i++) {
        double sum = FC2_BIAS[i];
        for (int j = 0; j < M; j++) {
          sum += FC2_WEIGHT[i * M + j] * h1[j];
        }
        out[r][i] = sum;
      }
    }
    double[] flat = new double[merged * OUT];
    for (int r = 0; r < merged; r++) {
      System.arraycopy(out[r], 0, flat, r * OUT, OUT);
    }
    return flat;
  }

  /** Population-variance LayerNorm over {@code length} elements, double. */
  private static double[] layerNorm(double[] v, float[] weight, float[] bias, int length) {
    double mean = 0;
    for (int i = 0; i < length; i++) {
      mean += v[i];
    }
    mean /= length;
    double var = 0;
    for (int i = 0; i < length; i++) {
      double d = v[i] - mean;
      var += d * d;
    }
    var /= length;
    double invStd = 1.0 / Math.sqrt(var + EPS);
    double[] out = new double[length];
    for (int i = 0; i < length; i++) {
      out[i] = (v[i] - mean) * invStd * weight[i] + bias[i];
    }
    return out;
  }

  /** The exact (erf) GELU, double. */
  private static double gelu(double v) {
    return 0.5 * v * (1.0 + erf(v / Math.sqrt(2.0)));
  }

  /**
   * The erf in double, Abramowitz-Stegun 7.1.26 approximation (max absolute error 1.5e-7, far
   * inside the 1e-5 test tolerance): {@code erf(x) = sign(x) * (1 - (a1 t + a2 t^2 + a3 t^3 + a4
   * t^4 + a5 t^5) e^{-x^2})} with {@code t = 1 / (1 + 0.3275911 |x|)} and the standard five
   * coefficients.
   */
  private static double erf(double x) {
    double sign = x < 0 ? -1.0 : 1.0;
    double a = Math.abs(x);
    double t = 1.0 / (1.0 + 0.3275911 * a);
    double p =
        t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
            * Math.exp(-a * a);
    return sign * (1.0 - p);
  }

  private static void assertArrayEqualsLoose(double[] expected, float[] actual, float delta) {
    assertEquals(expected.length, actual.length);
    for (int i = 0; i < expected.length; i++) {
      assertTrue(
          Math.abs(expected[i] - actual[i]) <= delta,
          "element " + i + ": expected " + expected[i] + ", got " + actual[i]);
    }
  }
}
