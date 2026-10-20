package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Strict tests for the frozen WP4 visual-injection compositions (req/plans/
 * phase7-3b-reference-findings.md §6.9): the row-gather replacement and the masked residual add.
 * Every float32 case builds the expected result by hand in Java and compares it at zero tolerance
 * after {@code toFloatArray()} forces evaluation: the destination rows must equal the feature rows
 * bit-for-bit, every other row must equal the original row, and the masked add's exact {@code +
 * 0.0} on the non-image rows stays observable (the zero-feature case uses exactly representable
 * values such as {@code -1.5f}, where any non-exact add would change bits; the negative-zero case
 * records that an untouched {@code -0.0} is normalised to {@code +0.0} while every non-zero value
 * keeps its exact bits). The float32 cases are tagged {@code full-float32} so they select into
 * {@code :jmlx-models:float32GoldenTest} with {@code MLX_ENABLE_TF32=0}; the rejection cases run in
 * the ordinary {@code :jmlx-models:test} task and expect {@link IllegalArgumentException} from the
 * helper's own validation, before any MLX array is written.
 */
@EnabledIfNativeAvailable
class VisualFeatureInjectionTest {

  private static final int HIDDEN = 4;

  private static MLXScope scope;

  @BeforeAll
  static void openScope() {
    scope = new MLXScope();
  }

  @AfterAll
  static void closeScope() {
    scope.close();
  }

  // ---------------------------------------------------------------------
  // row-gather replacement (full-float32)
  // ---------------------------------------------------------------------

  /** A single three-row image in the middle of the sequence replaces exactly its rows. */
  @Test
  @Tag("full-float32")
  void singleImageMidSequenceReplacesExactlyItsRows() {
    try (MLXScope run = scope.newChild()) {
      int length = 8;
      int[] destinations = {3, 4, 5};
      float[][] embedded = rows(length, 0f);
      float[][] features = rows(3, 10f);
      MLXArray replaced =
          VisualFeatureInjection.replaceImageRows(
              array(run, embedded), array(run, features), destinations);
      assertArrayEquals(new int[] {1, length, HIDDEN}, replaced.shape());
      assertEquals(run, replaced.scope());
      assertArrayEquals(
          flat(replacementRows(embedded, features, destinations)), replaced.toFloatArray());
    }
  }

  /** An image starting at row 0 replaces the leading rows. */
  @Test
  @Tag("full-float32")
  void imageAtSequenceStartReplacesLeadingRows() {
    try (MLXScope run = scope.newChild()) {
      int length = 8;
      int[] destinations = {0, 1, 2};
      float[][] embedded = rows(length, 0f);
      float[][] features = rows(3, 10f);
      MLXArray replaced =
          VisualFeatureInjection.replaceImageRows(
              array(run, embedded), array(run, features), destinations);
      assertArrayEquals(
          flat(replacementRows(embedded, features, destinations)), replaced.toFloatArray());
    }
  }

  /** An image ending at the last row replaces the trailing rows. */
  @Test
  @Tag("full-float32")
  void imageAtSequenceEndReplacesTrailingRows() {
    try (MLXScope run = scope.newChild()) {
      int length = 8;
      int[] destinations = {5, 6, 7};
      float[][] embedded = rows(length, 0f);
      float[][] features = rows(3, 10f);
      MLXArray replaced =
          VisualFeatureInjection.replaceImageRows(
              array(run, embedded), array(run, features), destinations);
      assertArrayEquals(
          flat(replacementRows(embedded, features, destinations)), replaced.toFloatArray());
    }
  }

  /** Two differently sized images, placed in prompt order, both replace exactly their rows. */
  @Test
  @Tag("full-float32")
  void twoDifferentlySizedPlacementsReplaceBothImages() {
    try (MLXScope run = scope.newChild()) {
      int length = 10;
      int[] destinations = {1, 2, 6, 7, 8};
      float[][] embedded = rows(length, 0f);
      float[][] features = combine(rows(2, 10f), rows(3, 20f));
      MLXArray replaced =
          VisualFeatureInjection.replaceImageRows(
              array(run, embedded), array(run, features), destinations);
      assertArrayEquals(
          flat(replacementRows(embedded, features, destinations)), replaced.toFloatArray());
    }
  }

  // ---------------------------------------------------------------------
  // masked residual add (full-float32)
  // ---------------------------------------------------------------------

  /** One DeepStack level adds into exactly the destination rows; every other row is untouched. */
  @Test
  @Tag("full-float32")
  void singleLevelAddsFeaturesIntoExactlyTheImageRows() {
    try (MLXScope run = scope.newChild()) {
      int length = 8;
      int[] destinations = {2, 3, 4};
      float[][] hidden = rows(length, 0f);
      float[][] features = rows(3, 10f);
      MLXArray updated =
          VisualFeatureInjection.addMaskedFeatures(
              run, array(run, hidden), array(run, features), destinations);
      assertArrayEquals(new int[] {1, length, HIDDEN}, updated.shape());
      assertArrayEquals(flat(addRows(hidden, features, destinations)), updated.toFloatArray());
    }
  }

  /**
   * Two levels at different destinations applied sequentially, mirroring the reference's per-level
   * masked clone + add + write-back: level 0 into rows 1-2, then level 1 into rows 5-7, with the
   * second add running on the first add's output.
   */
  @Test
  @Tag("full-float32")
  void twoLevelsApplySequentiallyAtDifferentDestinations() {
    try (MLXScope run = scope.newChild()) {
      int length = 8;
      float[][] hidden = rows(length, 0f);
      float[][] level0 = rows(2, 10f);
      float[][] level1 = rows(3, 20f);
      MLXArray afterLevel0 =
          VisualFeatureInjection.addMaskedFeatures(
              run, array(run, hidden), array(run, level0), new int[] {1, 2});
      float[][] expectedAfterLevel0 = addRows(hidden, level0, new int[] {1, 2});
      assertArrayEquals(flat(expectedAfterLevel0), afterLevel0.toFloatArray());
      MLXArray afterLevel1 =
          VisualFeatureInjection.addMaskedFeatures(
              run, afterLevel0, array(run, level1), new int[] {5, 6, 7});
      assertArrayEquals(
          flat(addRows(expectedAfterLevel0, level1, new int[] {5, 6, 7})),
          afterLevel1.toFloatArray());
    }
  }

  /**
   * Zero-valued feature rows are an exact {@code + 0.0}: every row, image rows included, comes back
   * bit-identical. The hidden values are exactly representable (multiples of 0.5 around {@code
   * -1.5f}), so a non-exact addition would change bits and fail the comparison.
   */
  @Test
  @Tag("full-float32")
  void zeroFeatureRowsAreAnExactPlusZeroIdentity() {
    try (MLXScope run = scope.newChild()) {
      int length = 8;
      float[][] hidden = new float[length][HIDDEN];
      for (int r = 0; r < length; r++) {
        for (int c = 0; c < HIDDEN; c++) {
          hidden[r][c] = -1.5f + (r + c) * 0.5f;
        }
      }
      MLXArray features = MLX.zeros(run, new int[] {1, 2, HIDDEN}, DType.FLOAT32);
      MLXArray updated =
          VisualFeatureInjection.addMaskedFeatures(
              run, array(run, hidden), features, new int[] {2, 3});
      assertArrayEquals(flat(hidden), updated.toFloatArray());
    }
  }

  /**
   * The masked add's exact {@code + 0.0} on a non-image row is an identity numerically, not
   * bit-for-bit: an IEEE {@code -0.0} element is normalised to {@code +0.0} (raw bits {@code
   * 0x00000000}), while the row's non-zero values — negative non-zero included — keep their exact
   * bits.
   */
  @Test
  @Tag("full-float32")
  void maskedAddNormalisesNegativeZeroInAnUntouchedRow() {
    try (MLXScope run = scope.newChild()) {
      int length = 8;
      float[][] hidden = rows(length, 0f);
      hidden[5][0] = -0.0f;
      hidden[5][1] = -1.5f;
      hidden[5][2] = 2.25f;
      hidden[5][3] = -0.125f;
      assertEquals(0x80000000, Float.floatToIntBits(hidden[5][0]));
      float[][] features = rows(2, 10f);
      int[] destinations = {3, 4};
      MLXArray updated =
          VisualFeatureInjection.addMaskedFeatures(
              run, array(run, hidden), array(run, features), destinations);
      float[] out = updated.toFloatArray();
      float[] expected = new float[length * HIDDEN];
      for (int r = 0; r < length; r++) {
        boolean destination = destinations[0] == r || destinations[1] == r;
        for (int c = 0; c < HIDDEN; c++) {
          expected[r * HIDDEN + c] =
              destination ? hidden[r][c] + features[r - destinations[0]][c] : hidden[r][c] + 0.0f;
        }
      }
      assertArrayEquals(expected, out);
      // The -0.0 element is normalised to +0.0: raw bits 0x00000000, not 0x80000000.
      assertEquals(0x00000000, Float.floatToIntBits(out[5 * HIDDEN]));
      // The row's non-zero values keep their exact bits.
      assertEquals(Float.floatToIntBits(-1.5f), Float.floatToIntBits(out[5 * HIDDEN + 1]));
      assertEquals(Float.floatToIntBits(2.25f), Float.floatToIntBits(out[5 * HIDDEN + 2]));
      assertEquals(Float.floatToIntBits(-0.125f), Float.floatToIntBits(out[5 * HIDDEN + 3]));
    }
  }

  // ---------------------------------------------------------------------
  // rejections (ordinary test task, MLX_ENABLE_TF32=1)
  // ---------------------------------------------------------------------

  /**
   * Every invalid input is rejected with {@link IllegalArgumentException} by the helper's own
   * validation — destination semantics via the shared gather-index algorithm, count and shape
   * checks via the pair validation — before any MLX array is written; a native failure from a
   * mis-built index array would surface as an {@code MLXException}, not an {@code
   * IllegalArgumentException}.
   */
  @Test
  void rejectionsThrowBeforeAnyArrayIsWritten() {
    try (MLXScope run = scope.newChild()) {
      MLXArray embedded = array(run, rows(8, 0f));
      MLXArray features = array(run, rows(3, 10f));
      assertThrows(
          IllegalArgumentException.class,
          () -> VisualFeatureInjection.replaceImageRows(embedded, features, new int[] {3, 8}));
      assertThrows(
          IllegalArgumentException.class,
          () -> VisualFeatureInjection.replaceImageRows(embedded, features, new int[] {-1}));
      assertThrows(
          IllegalArgumentException.class,
          () -> VisualFeatureInjection.replaceImageRows(embedded, features, new int[] {4, 3}));
      assertThrows(
          IllegalArgumentException.class,
          () -> VisualFeatureInjection.replaceImageRows(embedded, features, null));
      assertThrows(
          IllegalArgumentException.class,
          () -> VisualFeatureInjection.replaceImageRows(embedded, features, new int[] {3, 4}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              VisualFeatureInjection.replaceImageRows(
                  embedded, array(run, rows(2, 10f)), new int[] {3, 4, 5}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              VisualFeatureInjection.replaceImageRows(
                  embedded,
                  array(run, flat(rows(3, 10f)), new int[] {3, HIDDEN}),
                  new int[] {3, 4, 5}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              VisualFeatureInjection.replaceImageRows(
                  embedded,
                  MLX.array(run, flat(rows(6, 10f)), new int[] {2, 3, HIDDEN}),
                  new int[] {3, 4, 5}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              VisualFeatureInjection.replaceImageRows(
                  embedded,
                  MLX.array(run, flat(rows(3, 10f)), new int[] {1, 3, HIDDEN + 1}),
                  new int[] {3, 4, 5}));
      MLXArray half = MLX.astype(features, DType.FLOAT16);
      assertThrows(
          IllegalArgumentException.class,
          () -> VisualFeatureInjection.replaceImageRows(embedded, half, new int[] {3, 4, 5}));

      assertThrows(
          IllegalArgumentException.class,
          () ->
              VisualFeatureInjection.addMaskedFeatures(
                  null, embedded, features, new int[] {3, 4, 5}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              VisualFeatureInjection.addMaskedFeatures(
                  run,
                  array(run, flat(rows(3, 10f)), new int[] {3, HIDDEN}),
                  features,
                  new int[] {3, 4, 5}));
      assertThrows(
          IllegalArgumentException.class,
          () -> VisualFeatureInjection.addMaskedFeatures(run, embedded, half, new int[] {3, 4, 5}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              VisualFeatureInjection.addMaskedFeatures(
                  run, embedded, array(run, rows(2, 10f)), new int[] {3, 4, 5}));
    }
  }

  // ---------------------------------------------------------------------
  // data builders (Java-side expectations)
  // ---------------------------------------------------------------------

  /**
   * The replacement result by hand: destination rows take the feature rows, all others keep theirs.
   */
  private static float[][] replacementRows(
      float[][] embedded, float[][] features, int[] destinations) {
    float[][] result = new float[embedded.length][];
    for (int row = 0; row < embedded.length; row++) {
      float[] source = embedded[row];
      for (int f = 0; f < destinations.length; f++) {
        if (destinations[f] == row) {
          source = features[f];
        }
      }
      result[row] = source.clone();
    }
    return result;
  }

  /**
   * The masked-add result by hand: destination rows are the exact float sums, all others keep
   * theirs.
   */
  private static float[][] addRows(float[][] hidden, float[][] features, int[] destinations) {
    float[][] result = new float[hidden.length][];
    for (int row = 0; row < hidden.length; row++) {
      float[] source = hidden[row];
      for (int f = 0; f < destinations.length; f++) {
        if (destinations[f] == row) {
          float[] sum = new float[HIDDEN];
          for (int c = 0; c < HIDDEN; c++) {
            sum[c] = hidden[row][c] + features[f][c];
          }
          source = sum;
        }
      }
      result[row] = source.clone();
    }
    return result;
  }

  /** The {@code [1, rows, HIDDEN]} float32 array holding {@code values}. */
  private static MLXArray array(MLXScope scope, float[][] values) {
    return MLX.array(scope, flat(values), new int[] {1, values.length, HIDDEN});
  }

  private static MLXArray array(MLXScope scope, float[] flat, int[] shape) {
    return MLX.array(scope, flat, shape);
  }

  /**
   * Deterministic values at {@code (r, c)} of {@code offset + (r+1)/4 + (c+1)/16} — every value
   * exactly representable in float32.
   */
  private static float[][] rows(int rows, float offset) {
    float[][] values = new float[rows][];
    for (int r = 0; r < rows; r++) {
      values[r] = new float[HIDDEN];
      for (int c = 0; c < HIDDEN; c++) {
        values[r][c] = offset + (r + 1) * 0.25f + (c + 1) * 0.0625f;
      }
    }
    return values;
  }

  private static float[][] combine(float[][] first, float[][] second) {
    float[][] values = new float[first.length + second.length][];
    System.arraycopy(first, 0, values, 0, first.length);
    System.arraycopy(second, 0, values, first.length, second.length);
    return values;
  }

  private static float[] flat(float[][] values) {
    float[] flat = new float[values.length * HIDDEN];
    int i = 0;
    for (float[] row : values) {
      System.arraycopy(row, 0, flat, i, HIDDEN);
      i += HIDDEN;
    }
    return flat;
  }
}
