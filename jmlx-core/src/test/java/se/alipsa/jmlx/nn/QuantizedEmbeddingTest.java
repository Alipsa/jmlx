package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Fixture: {@code [4, 64]} float table ({@code numEmbeddings=4}, {@code dim=64} -- the smallest
 * shape divisible by the {@code groupSize=32} used throughout), built with {@code (i % 7 - 3) *
 * 0.3f}; indices fixture {@code [3, 0]} (deliberately non-identity order, so a gather-wiring
 * mistake is visible rather than hidden). {@code EPS_QUANT = 0.12f} bounds the 4-bit/group-32
 * dequantization error against the float reference: the pinned runtime's affine group quantization
 * uses step {@code (max - min) / 2^bits} with {@code bias = max} (confirmed empirically: this
 * fixture's per-group step is 1.8/16 = 0.1125, and the group minimum itself -- a full step below
 * the lowest representable level -- is the worst case), plus float rounding. Same-input
 * recomparisons use {@code 0f} (a deterministic native op on the same arrays).
 */
@EnabledIfNativeAvailable
class QuantizedEmbeddingTest {

  private static final float EPS_QUANT = 0.12f;
  private static final int EMBEDDINGS = 4;
  private static final int DIM = 64;
  private static final int GROUP_SIZE = 32;
  private static final int BITS = 4;

  private static float[] tableFixture() {
    float[] w = new float[EMBEDDINGS * DIM];
    for (int i = 0; i < w.length; i++) {
      w[i] = (i % 7 - 3) * 0.3f;
    }
    return w;
  }

  private static MLXArray[] quantizedTable(MLXScope scope, float[] table) {
    MLXArray w = MLX.array(scope, table, new int[] {EMBEDDINGS, DIM});
    return MLXQuant.quantize(w, GROUP_SIZE, BITS, "affine", null);
  }

  private static MLXArray indices(MLXScope scope) {
    return MLX.array(scope, new int[] {3, 0}, new int[] {2});
  }

  @Test
  void forwardMatchesTheFloatTableRowsWithinQuantizationError() {
    try (MLXScope scope = new MLXScope()) {
      float[] table = tableFixture();
      MLXArray[] q = quantizedTable(scope, table);
      QuantizedEmbedding embedding =
          new QuantizedEmbedding(scope, q[0], q[1], q[2], GROUP_SIZE, BITS);

      MLXArray result = embedding.forward(indices(scope));

      // Rows 3 then 0, per the indices fixture; both are within the 4-bit/group-32 error bound of
      // the original float rows.
      float[] expected = new float[2 * DIM];
      System.arraycopy(table, 3 * DIM, expected, 0, DIM);
      System.arraycopy(table, 0, expected, DIM, DIM);
      assertArrayEquals(new int[] {2, DIM}, result.shape());
      assertArrayEquals(expected, result.toFloatArray(), EPS_QUANT);
    }
  }

  /**
   * {@link QuantizedEmbedding#onParametersUpdated(java.util.Set)} rejects any write touching {@code
   * weight}: a shape check alone cannot verify a replacement was quantized under this table's own
   * {@code groupSize}/{@code bits} rather than a different pair sharing the same product, and the
   * fields {@code update} cannot see would stay stale. A {@code weight} write is rejected and
   * rolled back even when accompanied by a shape-consistent {@code scales}/{@code biases}
   * replacement, exactly as for {@link QuantizedLinear}.
   */
  @Test
  void updateOfWeightViaModuleUpdateThrowsAndRollsBack() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedTable(scope, tableFixture());
      QuantizedEmbedding embedding =
          new QuantizedEmbedding(scope, q[0], q[1], q[2], GROUP_SIZE, BITS);

      float[] otherTable = new float[EMBEDDINGS * DIM];
      for (int i = 0; i < otherTable.length; i++) {
        otherTable[i] = (i % 5 - 2) * 0.4f;
      }
      MLXArray w2 = MLX.array(scope, otherTable, new int[] {EMBEDDINGS, DIM});
      MLXArray[] q2 = MLXQuant.quantize(w2, GROUP_SIZE, BITS, "affine", null);

      assertThrows(
          IllegalStateException.class,
          () -> embedding.update(Map.of("weight", q2[0], "scales", q2[1], "biases", q2[2])));

      // weight is UINT32 (the packed dtype), so toFloatArray() does not apply here -- reference
      // identity is both sufficient and exact: rollback restores the same MLXArray.
      assertSame(q[0], embedding.parameters().get("weight"));
      assertSame(q[1], embedding.parameters().get("scales"));
      assertSame(q[2], embedding.parameters().get("biases"));
    }
  }

  /**
   * A {@code scales}/{@code biases}-only write is accepted and validated against this table's
   * current, unchanged {@code weight}/{@code groupSize}/{@code bits} (the gradient-step shape a
   * generic training loop must be able to write), with {@code forward} reflecting the new values.
   */
  @Test
  void updateOfScalesAndBiasesAloneViaModuleUpdateSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedTable(scope, tableFixture());
      QuantizedEmbedding embedding =
          new QuantizedEmbedding(scope, q[0], q[1], q[2], GROUP_SIZE, BITS);
      MLXArray ix = indices(scope);

      MLXArray newScales = MLXOps.multiply(q[1], MLX.array(scope, new float[] {2f}, new int[] {}));
      MLXArray newBiases = MLXOps.add(q[2], MLX.array(scope, new float[] {0.1f}, new int[] {}));

      embedding.update(Map.of("scales", newScales, "biases", newBiases));
      MLXArray result = embedding.forward(ix);

      QuantizedEmbedding rebuilt =
          new QuantizedEmbedding(scope, q[0], newScales, newBiases, GROUP_SIZE, BITS);
      assertArrayEquals(rebuilt.forward(ix).toFloatArray(), result.toFloatArray(), 0f);
      assertSame(newScales, embedding.parameters().get("scales"));
      assertSame(newBiases, embedding.parameters().get("biases"));
      assertSame(q[0], embedding.parameters().get("weight"));
    }
  }

  /**
   * A single-parameter write ({@code scales} alone) goes through the same hook: the new {@code
   * scales} is validated together with the unchanged {@code biases} against the current {@code
   * weight}/{@code groupSize}/{@code bits}, so the mixed pair can never be left inconsistent.
   */
  @Test
  void updateOfScalesAloneViaModuleUpdateSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedTable(scope, tableFixture());
      QuantizedEmbedding embedding =
          new QuantizedEmbedding(scope, q[0], q[1], q[2], GROUP_SIZE, BITS);
      MLXArray ix = indices(scope);

      MLXArray newScales = MLXOps.multiply(q[1], MLX.array(scope, new float[] {2f}, new int[] {}));

      embedding.update(Map.of("scales", newScales));
      MLXArray result = embedding.forward(ix);

      QuantizedEmbedding rebuilt =
          new QuantizedEmbedding(scope, q[0], newScales, q[2], GROUP_SIZE, BITS);
      assertArrayEquals(rebuilt.forward(ix).toFloatArray(), result.toFloatArray(), 0f);
      assertSame(newScales, embedding.parameters().get("scales"));
      assertSame(q[2], embedding.parameters().get("biases"));
    }
  }

  /**
   * The validation the success tests rely on actually runs: a replacement whose shape is
   * inconsistent with this table's unchanged {@code weight} (wrong row count) is rejected and
   * rolled back, exactly like the constructor's own check.
   */
  @Test
  void updateOfScalesAndBiasesWithAMismatchedShapeViaModuleUpdateThrowsAndRollsBack() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedTable(scope, tableFixture());
      QuantizedEmbedding embedding =
          new QuantizedEmbedding(scope, q[0], q[1], q[2], GROUP_SIZE, BITS);
      MLXArray wrongRowsScales = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray wrongRowsBiases = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});

      assertThrows(
          IllegalArgumentException.class,
          () -> embedding.update(Map.of("scales", wrongRowsScales, "biases", wrongRowsBiases)));

      assertSame(q[1], embedding.parameters().get("scales"));
      assertSame(q[2], embedding.parameters().get("biases"));
    }
  }

  /**
   * The safe replacement path {@link QuantizedEmbedding#updateQuantization}: {@code weight}/{@code
   * scales}/{@code biases} and {@code groupSize}/{@code bits} change together as one atomic unit.
   * Constructed at {@code groupSize=32, bits=4}; replaced with a payload actually quantized at
   * {@code groupSize=64, bits=2} and {@code forward} must compute the result for the new
   * configuration, matching a fresh table built directly from the replacement triple.
   */
  @Test
  void updateQuantizationReplacesTableAndGroupSizeBitsTogetherAndForwardIsCorrect() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedTable(scope, tableFixture());
      QuantizedEmbedding embedding =
          new QuantizedEmbedding(scope, q[0], q[1], q[2], GROUP_SIZE, BITS);

      float[] otherTable = new float[EMBEDDINGS * DIM];
      for (int i = 0; i < otherTable.length; i++) {
        otherTable[i] = (i % 5 - 2) * 0.4f;
      }
      MLXArray w2 = MLX.array(scope, otherTable, new int[] {EMBEDDINGS, DIM});
      MLXArray[] q2 = MLXQuant.quantize(w2, 64, 2, "affine", null);

      embedding.updateQuantization(q2[0], q2[1], q2[2], 64, 2);
      MLXArray ix = indices(scope);
      MLXArray result = embedding.forward(ix);

      QuantizedEmbedding rebuilt = new QuantizedEmbedding(scope, q2[0], q2[1], q2[2], 64, 2);
      assertArrayEquals(rebuilt.forward(ix).toFloatArray(), result.toFloatArray(), 0f);
      assertSame(q2[0], embedding.parameters().get("weight"));
    }
  }

  /**
   * {@code updateQuantization} re-runs the constructor's validation before writing anything: a
   * claimed {@code groupSize}/{@code bits} whose implied packed-column count disagrees with the
   * replacement triple is rejected, and no parameter or field is changed (deliberately not a
   * same-product pair such as {@code (32, 4)} for a {@code (64, 2)} triple -- that one satisfies
   * every shape check and is the pairing ambiguity the constructor's own javadoc documents).
   */
  @Test
  void updateQuantizationRejectsInconsistentPackingAndChangesNothing() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[] q = quantizedTable(scope, tableFixture());
      QuantizedEmbedding embedding =
          new QuantizedEmbedding(scope, q[0], q[1], q[2], GROUP_SIZE, BITS);

      float[] otherTable = new float[EMBEDDINGS * DIM];
      for (int i = 0; i < otherTable.length; i++) {
        otherTable[i] = (i % 5 - 2) * 0.4f;
      }
      MLXArray w2 = MLX.array(scope, otherTable, new int[] {EMBEDDINGS, DIM});
      MLXArray[] q2 = MLXQuant.quantize(w2, 64, 2, "affine", null);

      assertThrows(
          IllegalArgumentException.class,
          () -> embedding.updateQuantization(q2[0], q2[1], q2[2], 32, 2));

      assertSame(q[0], embedding.parameters().get("weight"));
      assertSame(q[1], embedding.parameters().get("scales"));
      assertSame(q[2], embedding.parameters().get("biases"));
    }
  }
}
