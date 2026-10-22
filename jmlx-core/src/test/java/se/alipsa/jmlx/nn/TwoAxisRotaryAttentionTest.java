package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Strict tests for the WP5 {@link TwoAxisRotaryAttention}
 * (req/plans/phase7-3b-reference-findings.md §11). Runs in the ordinary {@code :jmlx-core:test}
 * task. The random case compares the module's {@code scaledDotProductAttention} path against an
 * explicit matmul/softmax/matmul composition in the test (same projection, same rotation tables) to
 * validate the SDPA call and the reshapes without re-deriving the math.
 */
@EnabledIfNativeAvailable
class TwoAxisRotaryAttentionTest {

  private static final int N = 6;
  private static final int HEADS = 2;
  private static final int HEAD_DIM = 4;
  private static final int HIDDEN = HEADS * HEAD_DIM;

  @Test
  void zeroWeightsAndBiasesGiveZeroOutput() {
    try (MLXScope scope = new MLXScope()) {
      TwoAxisRotaryAttention attention =
          new TwoAxisRotaryAttention(
              scope,
              HEADS,
              HEAD_DIM,
              10_000f,
              MLX.zeros(scope, new int[] {3 * HIDDEN, HIDDEN}, DType.FLOAT32),
              MLX.zeros(scope, new int[] {3 * HIDDEN}, DType.FLOAT32),
              MLX.zeros(scope, new int[] {HIDDEN, HIDDEN}, DType.FLOAT32),
              MLX.zeros(scope, new int[] {HIDDEN}, DType.FLOAT32));
      float[] input = randomValues(new Random(0), N * HIDDEN);
      MLXArray arrayInput = MLX.array(scope, input, new int[] {N, HIDDEN});
      TwoAxisRotation rotation =
          new TwoAxisRope(10_000f, HEAD_DIM / 2).prepare(scope, N, row(N), row(N).clone());
      float[] out = attention.forward(arrayInput, rotation).toFloatArray();
      assertEquals(N * HIDDEN, out.length);
      for (float value : out) {
        assertEquals(0.0f, value, "zero weights/biases must give an exactly zero output");
      }
    }
  }

  @Test
  void constantBiasWithIdentityProjectionGivesUniformMeanOfValueRows() {
    try (MLXScope scope = new MLXScope()) {
      float c = 0.5f;
      float[] qkvBias = new float[3 * HIDDEN];
      java.util.Arrays.fill(qkvBias, c);
      float[] identity = new float[HIDDEN * HIDDEN];
      for (int i = 0; i < HIDDEN; i++) {
        identity[i * HIDDEN + i] = 1.0f;
      }
      TwoAxisRotaryAttention attention =
          new TwoAxisRotaryAttention(
              scope,
              HEADS,
              HEAD_DIM,
              10_000f,
              MLX.zeros(scope, new int[] {3 * HIDDEN, HIDDEN}, DType.FLOAT32),
              MLX.array(scope, qkvBias, new int[] {3 * HIDDEN}),
              MLX.array(scope, identity, new int[] {HIDDEN, HIDDEN}),
              MLX.zeros(scope, new int[] {HIDDEN}, DType.FLOAT32));
      float[] input = randomValues(new Random(1), N * HIDDEN);
      MLXArray arrayInput = MLX.array(scope, input, new int[] {N, HIDDEN});
      // Zero positions: the rotation is the identity, so q = k = v = the constant bias.
      int[] zero = new int[N];
      TwoAxisRotation rotation =
          new TwoAxisRope(10_000f, HEAD_DIM / 2).prepare(scope, N, zero, zero.clone());
      MLXArray outArray = attention.forward(arrayInput, rotation);
      assertArrayEquals(new int[] {N, HIDDEN}, outArray.shape());
      // Every k row is constant, so every score row is constant and the softmax is the uniform
      // 1/N; the attention output is then exactly the uniform mean of the value rows. The
      // identity proj passes it through, so the expected output is that mean (c here, computed
      // from the actual value rows for the assertion).
      MLXArray v = valueRows(scope, qkvBias);
      float[] values = v.toFloatArray();
      double expected = 0;
      for (float value : values) {
        expected += value;
      }
      expected /= values.length;
      float[] out = outArray.toFloatArray();
      for (int i = 0; i < out.length; i++) {
        // 1e-3, not 1e-4: this task runs in the default TF32 mode, where the SDPA softmax weights
        // (the uniform 1/N here) pick up an ~11-bit-mantissa rounding -- a constant-0.5 input
        // produced 0.5 * (1 - 2^-11) locally on 2026-10-21. Structural errors are O(1), so the
        // looser bound still discriminates.
        assertTrue(
            Float.isFinite(out[i]) && Math.abs(out[i] - expected) <= 1e-3,
            "element " + i + ": expected uniform mean " + expected + ", got " + out[i]);
      }
    }
  }

  @Test
  void randomSmallCaseMatchesExplicitMatmulSoftmaxComposition() {
    Random random = new Random(20_261_021);
    float[] qkvWeight = randomValues(random, 3 * HIDDEN * HIDDEN);
    float[] qkvBias = randomValues(random, 3 * HIDDEN);
    float[] projWeight = randomValues(random, HIDDEN * HIDDEN);
    float[] projBias = randomValues(random, HIDDEN);
    float[] input = randomValues(random, N * HIDDEN);
    int[] axis0 = row(N);
    int[] axis1 = {0, 1, 2, 0, 1, 2};
    try (MLXScope scope = new MLXScope()) {
      MLXArray qkvWeightArr = MLX.array(scope, qkvWeight, new int[] {3 * HIDDEN, HIDDEN});
      MLXArray qkvBiasArr = MLX.array(scope, qkvBias, new int[] {3 * HIDDEN});
      MLXArray projWeightArr = MLX.array(scope, projWeight, new int[] {HIDDEN, HIDDEN});
      MLXArray projBiasArr = MLX.array(scope, projBias, new int[] {HIDDEN});
      MLXArray arrayInput = MLX.array(scope, input, new int[] {N, HIDDEN});
      TwoAxisRotaryAttention attention =
          new TwoAxisRotaryAttention(
              scope,
              HEADS,
              HEAD_DIM,
              10_000f,
              qkvWeightArr,
              qkvBiasArr,
              projWeightArr,
              projBiasArr);
      TwoAxisRope rope = new TwoAxisRope(10_000f, HEAD_DIM / 2);
      TwoAxisRotation rotation = rope.prepare(scope, N, axis0, axis1);
      float[] actual = attention.forward(arrayInput, rotation).toFloatArray();

      // The explicit composition: the same qkv projection, the same rotation tables applied by
      // hand, then scores = scale * q @ k^T, softmax, @ v -- no SDPA call.
      MLXArray qkvOut =
          MLXOps.add(
              MLXOps.matmul(arrayInput, MLXShape.transpose(qkvWeightArr, new int[] {1, 0})),
              qkvBiasArr);
      MLXArray projected =
          MLXShape.transpose(
              MLXShape.reshape(qkvOut, new int[] {N, 3, HEADS, HEAD_DIM}), new int[] {1, 0, 2, 3});
      MLXArray[] parts = MLXShape.split(projected, 3, 0);
      MLXArray cos = rotation.cos();
      MLXArray sin = rotation.sin();
      // mlx_split keeps the split axis: each part is [1, N, heads, headDim], so squeeze axis 0 to
      // the [N, heads, headDim] layout the rotation tables and batched() expect.
      MLXArray q = manualRotate(scope, MLXShape.squeeze(parts[0], new int[] {0}), cos, sin);
      MLXArray k = manualRotate(scope, MLXShape.squeeze(parts[1], new int[] {0}), cos, sin);
      MLXArray v = MLXShape.squeeze(parts[2], new int[] {0});
      MLXArray q4 = batched(q, N);
      MLXArray k4 = batched(k, N);
      MLXArray v4 = batched(v, N);
      MLXArray scores =
          MLXOps.multiply(
              MLXOps.matmul(q4, MLXShape.transpose(k4, new int[] {0, 1, 3, 2})),
              MLX.full(scope, new int[] {1}, attention.scale(), DType.FLOAT32));
      MLXArray attended = MLXOps.matmul(MLXOps.softmaxAxis(scores, -1, true), v4);
      MLXArray merged =
          MLXShape.reshape(
              MLXShape.transpose(
                  MLXShape.reshape(attended, new int[] {HEADS, N, HEAD_DIM}), new int[] {1, 0, 2}),
              new int[] {N, HIDDEN});
      float[] expected =
          MLXOps.add(
                  MLXOps.matmul(merged, MLXShape.transpose(projWeightArr, new int[] {1, 0})),
                  projBiasArr)
              .toFloatArray();
      double maxAbs = 0;
      int worst = -1;
      for (int i = 0; i < expected.length; i++) {
        double diff = Math.abs(actual[i] - expected[i]);
        if (diff > maxAbs) {
          maxAbs = diff;
          worst = i;
        }
      }
      assertTrue(
          maxAbs <= 1e-4, "max abs err " + maxAbs + " at element " + worst + " exceeds 1e-4");
    }
  }

  @Test
  void forwardRejectsMismatchedRotationAndInputs() {
    try (MLXScope scope = new MLXScope()) {
      float[] qkvWeight = randomValues(new Random(2), 3 * HIDDEN * HIDDEN);
      float[] qkvBias = randomValues(new Random(3), 3 * HIDDEN);
      float[] projWeight = randomValues(new Random(4), HIDDEN * HIDDEN);
      float[] projBias = randomValues(new Random(5), HIDDEN);
      MLXArray qkvWeightArr = MLX.array(scope, qkvWeight, new int[] {3 * HIDDEN, HIDDEN});
      MLXArray qkvBiasArr = MLX.array(scope, qkvBias, new int[] {3 * HIDDEN});
      MLXArray projWeightArr = MLX.array(scope, projWeight, new int[] {HIDDEN, HIDDEN});
      MLXArray projBiasArr = MLX.array(scope, projBias, new int[] {HIDDEN});
      MLXArray arrayInput =
          MLX.array(scope, randomValues(new Random(6), N * HIDDEN), new int[] {N, HIDDEN});
      TwoAxisRotaryAttention attention =
          new TwoAxisRotaryAttention(
              scope,
              HEADS,
              HEAD_DIM,
              10_000f,
              qkvWeightArr,
              qkvBiasArr,
              projWeightArr,
              projBiasArr);
      // headDim 4 against a rotation whose 2 * rotaryDims is 8
      TwoAxisRotation wrongWidth =
          new TwoAxisRope(10_000f, 4).prepare(scope, N, row(N), row(N).clone());
      IllegalArgumentException width =
          assertThrows(
              IllegalArgumentException.class, () -> attention.forward(arrayInput, wrongWidth));
      assertTrue(width.getMessage().contains("2 * rotaryDims"));
      // sequence mismatch: a 4-row rotation against 6 rows
      TwoAxisRotation shortRotation =
          new TwoAxisRope(10_000f, HEAD_DIM / 2).prepare(scope, 4, row(4), row(4).clone());
      assertThrows(
          IllegalArgumentException.class, () -> attention.forward(arrayInput, shortRotation));
      // input with the wrong last dimension
      MLXArray wrongHidden =
          MLX.array(
              scope, randomValues(new Random(7), N * (HIDDEN + 1)), new int[] {N, HIDDEN + 1});
      assertThrows(
          IllegalArgumentException.class, () -> attention.forward(wrongHidden, shortRotation));
      // input with the wrong rank
      MLXArray rank3 =
          MLX.array(scope, randomValues(new Random(8), N * HIDDEN), new int[] {1, N, HIDDEN});
      assertThrows(IllegalArgumentException.class, () -> attention.forward(rank3, shortRotation));
      assertThrows(IllegalArgumentException.class, () -> attention.forward(null, shortRotation));
      assertThrows(IllegalArgumentException.class, () -> attention.forward(arrayInput, null));
    }
  }

  @Test
  void constructorRejectsInvalidGeometry() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray qkv =
          MLX.array(
              scope,
              randomValues(new Random(9), 3 * HIDDEN * HIDDEN),
              new int[] {3 * HIDDEN, HIDDEN});
      MLXArray qkvBias =
          MLX.array(scope, randomValues(new Random(10), 3 * HIDDEN), new int[] {3 * HIDDEN});
      MLXArray proj =
          MLX.array(
              scope, randomValues(new Random(11), HIDDEN * HIDDEN), new int[] {HIDDEN, HIDDEN});
      MLXArray projBias =
          MLX.array(scope, randomValues(new Random(12), HIDDEN), new int[] {HIDDEN});
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new TwoAxisRotaryAttention(
                  scope, 0, HEAD_DIM, 10_000f, qkv, qkvBias, proj, projBias));
      assertThrows(
          IllegalArgumentException.class,
          () -> new TwoAxisRotaryAttention(scope, HEADS, 0, 10_000f, qkv, qkvBias, proj, projBias));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new TwoAxisRotaryAttention(
                  scope, HEADS, HEAD_DIM, Float.NaN, qkv, qkvBias, proj, projBias));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new TwoAxisRotaryAttention(
                  scope, HEADS, HEAD_DIM, 10_000f, null, qkvBias, proj, projBias));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new TwoAxisRotaryAttention(
                  scope, HEADS, HEAD_DIM, 10_000f, qkv, null, proj, projBias));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new TwoAxisRotaryAttention(
                  scope, HEADS, HEAD_DIM, 10_000f, qkv, qkvBias, proj, null));
      // qkv weight with the wrong output width
      MLXArray qkvWrong =
          MLX.array(
              scope,
              randomValues(new Random(13), 5 * HIDDEN * HIDDEN),
              new int[] {5 * HIDDEN, HIDDEN});
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new TwoAxisRotaryAttention(
                  scope, HEADS, HEAD_DIM, 10_000f, qkvWrong, qkvBias, proj, projBias));
      // proj weight with the wrong width
      MLXArray projWrong =
          MLX.array(
              scope,
              randomValues(new Random(14), HIDDEN * (HIDDEN + 1)),
              new int[] {HIDDEN, HIDDEN + 1});
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new TwoAxisRotaryAttention(
                  scope, HEADS, HEAD_DIM, 10_000f, qkv, qkvBias, projWrong, projBias));
      float expectedScale = attentionScaleOf(HEAD_DIM);
      TwoAxisRotaryAttention ok =
          new TwoAxisRotaryAttention(scope, HEADS, HEAD_DIM, 10_000f, qkv, qkvBias, proj, projBias);
      assertEquals(expectedScale, ok.scale(), 0.0f);
      assertEquals(10_000f, ok.theta());
    }
  }

  /**
   * The constant-bias value rows: with a zero qkv weight, v is the broadcast v block of the qkv
   * bias.
   */
  private static MLXArray valueRows(MLXScope scope, float[] qkvBias) {
    float[] valueBlock = java.util.Arrays.copyOfRange(qkvBias, 2 * HIDDEN, 3 * HIDDEN);
    MLXArray v = MLX.array(scope, valueBlock, new int[] {HEADS, HEAD_DIM});
    return MLXShape.broadcastTo(v, new int[] {N, HEADS, HEAD_DIM});
  }

  /** The half-split rotate_half composition in float32, applied with hand-written ops. */
  private static MLXArray manualRotate(MLXScope scope, MLXArray x, MLXArray cos, MLXArray sin) {
    int half = HEAD_DIM / 2;
    int[] fullStop = {N, HEADS, HEAD_DIM};
    int[] frontStop = {N, HEADS, half};
    MLXArray front = MLXShape.slice(x, new int[] {0, 0, 0}, frontStop);
    MLXArray back = MLXShape.slice(x, new int[] {0, 0, half}, fullStop);
    MLXArray rotated = MLXShape.concatenate(new MLXArray[] {MLXOps.negative(back), front}, 2);
    return MLXOps.add(MLXOps.multiply(x, cos), MLXOps.multiply(rotated, sin));
  }

  /** The {@code [1, heads, N, headDim]} batched layout the explicit composition uses. */
  private static MLXArray batched(MLXArray x, int n) {
    return MLXShape.reshape(
        MLXShape.transpose(x, new int[] {1, 0, 2}), new int[] {1, HEADS, n, HEAD_DIM});
  }

  private static float[] randomValues(Random random, int n) {
    float[] data = new float[n];
    for (int i = 0; i < n; i++) {
      data[i] = random.nextFloat() * 0.2f - 0.1f;
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

  private static float attentionScaleOf(int headDim) {
    return (float) Math.pow(headDim, -0.5);
  }
}
