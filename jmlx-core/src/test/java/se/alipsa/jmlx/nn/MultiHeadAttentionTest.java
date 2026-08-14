package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class MultiHeadAttentionTest {

  private static final float EPS = 1e-3f;
  private static final int BATCH = 1;
  private static final int HEADS = 2;
  private static final int HEAD_DIM = 4;
  private static final int SEQ = 3;
  private static final int EMBED_DIM = HEADS * HEAD_DIM; // 8

  private static final float[] X_DATA = {
    1, 2, 3, 4, 5, 6, 7, 8,
    8, 7, 6, 5, 4, 3, 2, 1,
    2, 4, 6, 8, 1, 3, 5, 7
  };

  private static MLXArray stackedIdentityQkvWeight(MLXScope scope) {
    float[] data = new float[3 * EMBED_DIM * EMBED_DIM];
    for (int block = 0; block < 3; block++) {
      for (int i = 0; i < EMBED_DIM; i++) {
        data[(block * EMBED_DIM + i) * EMBED_DIM + i] = 1f;
      }
    }
    return MLX.array(scope, data, new int[] {3 * EMBED_DIM, EMBED_DIM});
  }

  private static MLXArray identityWeight(MLXScope scope) {
    float[] data = new float[EMBED_DIM * EMBED_DIM];
    for (int i = 0; i < EMBED_DIM; i++) {
      data[i * EMBED_DIM + i] = 1f;
    }
    return MLX.array(scope, data, new int[] {EMBED_DIM, EMBED_DIM});
  }

  /**
   * Per-head causal attention computed independently of {@link MultiHeadAttention#forward}: a
   * separate per-position rope call per head/position, and a manual matmul+triu+where+softmaxAxis
   * causal mask instead of the built-in {@code causal} flag -- Decision 8's composition-identity
   * fallback, since a literal hand-derived golden for a full multi-head attention pass is
   * impractical.
   */
  private static float[] composedReference(MLXScope scope, MLXArray x) {
    float[] expected = new float[BATCH * SEQ * EMBED_DIM];
    for (int h = 0; h < HEADS; h++) {
      MLXArray head =
          MLXShape.slice(
              x, new int[] {0, 0, h * HEAD_DIM}, new int[] {BATCH, SEQ, (h + 1) * HEAD_DIM});
      MLXArray[] roped = new MLXArray[SEQ];
      for (int t = 0; t < SEQ; t++) {
        MLXArray pos =
            MLXShape.slice(head, new int[] {0, t, 0}, new int[] {BATCH, t + 1, HEAD_DIM});
        roped[t] = MLXFast.rope(pos, HEAD_DIM, false, 10000f, 1.0f, t, null);
      }
      MLXArray ropedHead = MLXShape.concatenate(roped, 1); // [BATCH, SEQ, HEAD_DIM]
      MLXArray scoresRaw =
          MLXOps.matmul(ropedHead, MLXShape.transpose(ropedHead, new int[] {0, 2, 1}));
      MLXArray scaled =
          MLXOps.multiply(
              scoresRaw,
              MLX.full(scope, new int[0], 1f / (float) Math.sqrt(HEAD_DIM), DType.FLOAT32));
      MLXArray maskBool =
          MLXShape.triu(MLX.ones(scope, new int[] {BATCH, SEQ, SEQ}, DType.BOOL), 1);
      MLXArray negInf = MLX.full(scope, new int[] {BATCH, SEQ, SEQ}, -1e9f, DType.FLOAT32);
      MLXArray zero = MLX.full(scope, new int[] {BATCH, SEQ, SEQ}, 0f, DType.FLOAT32);
      MLXArray additiveMask = MLXOps.where(maskBool, negInf, zero);
      MLXArray weights = MLXOps.softmaxAxis(MLXOps.add(scaled, additiveMask), 2, true);
      MLXArray headOut = MLXOps.matmul(weights, head); // [BATCH, SEQ, HEAD_DIM]
      float[] headOutData = headOut.toFloatArray();
      for (int t = 0; t < SEQ; t++) {
        for (int d = 0; d < HEAD_DIM; d++) {
          expected[t * EMBED_DIM + h * HEAD_DIM + d] = headOutData[t * HEAD_DIM + d];
        }
      }
    }
    return expected;
  }

  @Test
  void forwardWithoutCacheMatchesPerHeadComposedReferenceWithCausalMasking() {
    try (MLXScope scope = new MLXScope()) {
      MultiHeadAttention mha =
          new MultiHeadAttention(
              scope, HEADS, stackedIdentityQkvWeight(scope), null, identityWeight(scope), null);
      MLXArray x = MLX.array(scope, X_DATA, new int[] {BATCH, SEQ, EMBED_DIM});

      float[] expected = composedReference(scope, x);
      MLXArray actual = mha.forward(x, null, true);

      assertArrayEquals(expected, actual.toFloatArray(), EPS);
    }
  }

  /**
   * Decodes the same three tokens one at a time through a shared {@link KVCache} and asserts the
   * newest token's output matches the equivalent row of a fresh full-sequence prefill -- the
   * defining correctness property of KV caching. Each step's activations live in their own child
   * scope of the cache's scope, matching the real decode-loop shape {@link KVCache}'s javadoc
   * describes.
   */
  @Test
  void incrementalDecodeWithKvCacheMatchesFullPrefillForTheNewestPosition() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray qkvWeight = stackedIdentityQkvWeight(scope);
      MLXArray outWeight = identityWeight(scope);

      MultiHeadAttention fullMha =
          new MultiHeadAttention(scope, HEADS, qkvWeight, null, outWeight, null);
      MLXArray x = MLX.array(scope, X_DATA, new int[] {BATCH, SEQ, EMBED_DIM});
      float[] fullOut = fullMha.forward(x, null, true).toFloatArray();
      float[] lastRowExpected = Arrays.copyOfRange(fullOut, 2 * EMBED_DIM, 3 * EMBED_DIM);

      MultiHeadAttention decodeMha =
          new MultiHeadAttention(scope, HEADS, qkvWeight, null, outWeight, null);
      KVCache cache = new KVCache(scope);
      for (int t = 0; t < 2; t++) {
        try (MLXScope step = scope.newChild()) {
          MLXArray xt =
              MLX.array(
                  step,
                  Arrays.copyOfRange(X_DATA, t * EMBED_DIM, (t + 1) * EMBED_DIM),
                  new int[] {BATCH, 1, EMBED_DIM});
          decodeMha.forward(xt, cache, true);
        }
      }
      float[] lastActual;
      try (MLXScope step = scope.newChild()) {
        MLXArray lastX =
            MLX.array(
                step,
                Arrays.copyOfRange(X_DATA, 2 * EMBED_DIM, 3 * EMBED_DIM),
                new int[] {BATCH, 1, EMBED_DIM});
        lastActual = decodeMha.forward(lastX, cache, true).toFloatArray();
      }

      assertArrayEquals(lastRowExpected, lastActual, EPS);
    }
  }

  @Test
  void constructorRejectsANonSquareOutWeight() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray outWeight = MLX.array(scope, new float[EMBED_DIM * 3], new int[] {EMBED_DIM, 3});
      MLXArray qkvWeight = stackedIdentityQkvWeight(scope);
      assertThrows(
          IllegalArgumentException.class,
          () -> new MultiHeadAttention(scope, HEADS, qkvWeight, null, outWeight, null));
    }
  }

  @Test
  void constructorRejectsANumHeadsThatDoesNotDivideEmbedDim() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray outWeight = identityWeight(scope);
      MLXArray qkvWeight = stackedIdentityQkvWeight(scope);
      assertThrows(
          IllegalArgumentException.class,
          () -> new MultiHeadAttention(scope, 3, qkvWeight, null, outWeight, null));
    }
  }

  @Test
  void constructorRejectsAQkvWeightShapedForADifferentEmbedDim() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray outWeight = identityWeight(scope);
      MLXArray badQkvWeight =
          MLX.array(
              scope, new float[2 * EMBED_DIM * EMBED_DIM], new int[] {2 * EMBED_DIM, EMBED_DIM});
      assertThrows(
          IllegalArgumentException.class,
          () -> new MultiHeadAttention(scope, HEADS, badQkvWeight, null, outWeight, null));
    }
  }
}
