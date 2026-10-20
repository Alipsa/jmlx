package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Seam tests for the frozen WP4 decoder rotation integration (req/plans/phase7-3b-reference-
 * findings.md §6.9): the batch-aware {@link DecoderAttention} forward that rotates the projected
 * query and key heads from a request-supplied {@link MultiAxisRotation}, and the {@link
 * DecoderBlock} passthrough. Every float32 case compares the seam against an operation-identical
 * manual composition built from the same retained projection and normalization instances -- the
 * same operations, the same order, the same shapes, the same scopes -- so those comparisons run at
 * zero tolerance, and the cases are tagged {@code full-float32} so they select into {@code
 * :jmlx-core:float32GoldenTest} with {@code MLX_ENABLE_TF32=0}. The single 1e-5 exception is the
 * decode-versus-fresh-prefill comparison in {@link #rotationPrefillAndDecodeBitMatchFreshPrefill},
 * where the two runs use different kernel shapes. The rejection cases run in the ordinary {@code
 * :jmlx-core:test} task.
 */
@EnabledIfNativeAvailable
class DecoderRotationSeamTest {

  private static final int HIDDEN = 8;
  private static final int HEADS = 1;
  private static final int KEY_VALUE_HEADS = 1;
  private static final int HEAD_DIM = 8;
  private static final int[] SECTIONS = {2, 1, 1}; // rotaryDims 8 == HEAD_DIM
  private static final float SCALE = (float) (1.0 / Math.sqrt(HEAD_DIM));

  // ---------------------------------------------------------------------
  // float32-comparison cases (full-float32 tag, MLX_ENABLE_TF32=0)
  // ---------------------------------------------------------------------

  /**
   * A five-token prefill through the rotation seam must be bit-identical to the operation-identical
   * reference composition on a second cache, the retained keys must equal the rotated projected
   * keys (pinning rotate-before-append), and the one-token decode at position five must match the
   * last row of a fresh six-token prefill at the cross-shape tolerance. The tolerance is needed
   * because the decode (one query over a retained six-key cache) and the fresh prefill (six queries
   * over six keys) drive different native attention and matmul kernel shapes: the last row's
   * arithmetic is the same, but MLX does not guarantee bit-equal results across shapes.
   */
  @Test
  @Tag("full-float32")
  void rotationPrefillAndDecodeBitMatchFreshPrefill() {
    try (MLXScope scope = new MLXScope()) {
      AttentionFixture fixture = new AttentionFixture(scope, null, null);
      MultiAxisRope rope = new MultiAxisRope(10_000f, SECTIONS);
      try (MLXScope run = scope.newChild()) {
        KVCache cache = new KVCache(run);
        KVCache referenceCache = new KVCache(run);
        float[] prefill = values(1 * 5 * HIDDEN, 1);
        MLXArray x = MLX.array(run, prefill, new int[] {1, 5, HIDDEN});
        MultiAxisRotation rotation =
            rope.prepare(
                run, 1, 5, new int[][] {{0, 1, 2, 3, 4}, {0, 2, 4, 2, 4}, {1, 1, 3, 3, 5}});
        MLXArray mask =
            AttentionMask.batched(run, new int[] {0}, new int[] {0}, new int[] {5}, 5, 5, 0);
        MLXArray actual = fixture.attention.forward(x, cache, mask, new int[] {5}, rotation);
        MLXArray reference =
            referenceAttention(fixture, x, referenceCache, mask, new int[] {5}, rotation);
        assertArrayEquals(reference.toFloatArray(), actual.toFloatArray());
        // The retained keys are the rotated projected keys: rotation lands before the append.
        MLXArray expectedKeys =
            rotation.rotate(
                AttentionHeads.toHeads(
                    fixture.keyProj.forward(x), 1, 5, KEY_VALUE_HEADS, HEAD_DIM));
        assertArrayEquals(expectedKeys.toFloatArray(), cache.keys().toFloatArray());

        float[] decodeInput = values(1 * 1 * HIDDEN, 7);
        MLXArray x1 = MLX.array(run, decodeInput, new int[] {1, 1, HIDDEN});
        MultiAxisRotation decodeRotation = rope.prepare(run, 1, 1, new int[][] {{5}, {6}, {4}});
        MLXArray decodeMask =
            AttentionMask.batched(run, new int[] {5}, new int[] {0}, new int[] {1}, 1, 6, 0);
        MLXArray decoded =
            fixture.attention.forward(x1, cache, decodeMask, new int[] {1}, decodeRotation);
        assertEquals(6, cache.nextPosition());
        MLXArray freshLast = freshSixTokenPrefillLastRow(fixture, rope, run, prefill, decodeInput);
        assertArrayEquals(freshLast.toFloatArray(), decoded.toFloatArray(), 1e-5f);
      }
    }
  }

  /**
   * With non-unit QK normalization the seam must normalize before rotating: the reference applies
   * the norms at the same position (after projection, before the rotation), so a different order
   * would change both the retained keys and the output at zero tolerance.
   */
  @Test
  @Tag("full-float32")
  void qkNormRunsBeforeRotation() {
    try (MLXScope scope = new MLXScope()) {
      float[] queryWeight = new float[HIDDEN];
      float[] keyWeight = new float[HIDDEN];
      for (int i = 0; i < HIDDEN; i++) {
        queryWeight[i] = 2f;
        keyWeight[i] = 0.5f;
      }
      AttentionFixture fixture =
          new AttentionFixture(
              scope,
              new RMSNorm(scope, MLX.array(scope, queryWeight, new int[] {HIDDEN}), 1e-6f),
              new RMSNorm(scope, MLX.array(scope, keyWeight, new int[] {HIDDEN}), 1e-6f));
      MultiAxisRope rope = new MultiAxisRope(10_000f, SECTIONS);
      try (MLXScope run = scope.newChild()) {
        KVCache cache = new KVCache(run);
        KVCache referenceCache = new KVCache(run);
        MLXArray x = MLX.array(run, values(1 * 3 * HIDDEN, 3), new int[] {1, 3, HIDDEN});
        MultiAxisRotation rotation =
            rope.prepare(run, 1, 3, new int[][] {{0, 1, 2}, {1, 3, 5}, {2, 4, 6}});
        MLXArray mask =
            AttentionMask.batched(run, new int[] {0}, new int[] {0}, new int[] {3}, 3, 3, 0);
        MLXArray actual = fixture.attention.forward(x, cache, mask, new int[] {3}, rotation);
        MLXArray reference =
            referenceAttention(fixture, x, referenceCache, mask, new int[] {3}, rotation);
        assertArrayEquals(reference.toFloatArray(), actual.toFloatArray());
      }
    }
  }

  /**
   * A rotation prepared at all-zero coordinates is the bitwise identity: the angles are exactly
   * zero, so the cos table is exactly 1 and the sin table exactly 0, and multiplying by 1 plus
   * adding 0 is exact for every finite float. The test values never contain {@code -0.0} (the
   * {@link #values} helper produces {@code 0/6.0f} for its zero, a positive zero), so the identity
   * holds element for element, sign bits included.
   */
  @Test
  @Tag("full-float32")
  void zeroPositionRotationIsBitwiseIdentity() {
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(10_000f, SECTIONS);
      try (MLXScope run = scope.newChild()) {
        float[] x = values(1 * 1 * 4 * HEAD_DIM, 0);
        MultiAxisRotation rotation =
            rope.prepare(run, 1, 4, new int[][] {zeros(4), zeros(4), zeros(4)});
        MLXArray array = MLX.array(run, x, new int[] {1, 1, 4, HEAD_DIM});
        assertArrayEquals(x, rotation.rotate(array).toFloatArray());
      }
    }
  }

  /**
   * A ragged batch (valid lengths {2, 4} on a width-4 prefill, then {1, 1} on a width-1 decode)
   * must bit-match the reference composition on a second cache and keep the per-row bookkeeping:
   * the decode advances each row from its own position (row 0 from 2, row 1 from 4), and the
   * rotation's per-row coordinates address the right valid suffix of every row.
   */
  @Test
  @Tag("full-float32")
  void raggedBatchKeepsPerRowPositionsAndBitMatchesReference() {
    try (MLXScope scope = new MLXScope()) {
      AttentionFixture fixture = new AttentionFixture(scope, null, null);
      MultiAxisRope rope = new MultiAxisRope(10_000f, SECTIONS);
      try (MLXScope run = scope.newChild()) {
        KVCache cache = new KVCache(run);
        KVCache referenceCache = new KVCache(run);
        // B=2 prefill, valid {2, 4}, width 4; row-major B*T positions, rows unequal on all axes.
        MLXArray x = MLX.array(run, values(2 * 4 * HIDDEN, 2), new int[] {2, 4, HIDDEN});
        MultiAxisRotation rotation =
            rope.prepare(
                run,
                2,
                4,
                new int[][] {
                  {0, 1, 0, 0, 2, 3, 4, 5}, {0, 2, 0, 0, 0, 1, 2, 3}, {1, 0, 0, 0, 1, 1, 2, 2}
                });
        MLXArray mask =
            AttentionMask.batched(
                run, new int[] {0, 0}, new int[] {0, 0}, new int[] {2, 4}, 4, 4, 0);
        MLXArray actual = fixture.attention.forward(x, cache, mask, new int[] {2, 4}, rotation);
        MLXArray reference =
            referenceAttention(fixture, x, referenceCache, mask, new int[] {2, 4}, rotation);
        assertArrayEquals(reference.toFloatArray(), actual.toFloatArray());
        assertEquals(2, cache.nextPosition(0));
        assertEquals(4, cache.nextPosition(1));
        assertEquals(2, cache.rowLength(0));
        assertEquals(4, cache.rowLength(1));

        // B=2 decode, valid {1, 1}, width 1; each row decodes from its own position.
        MLXArray x1 = MLX.array(run, values(2 * 1 * HIDDEN, 5), new int[] {2, 1, HIDDEN});
        MultiAxisRotation decodeRotation =
            rope.prepare(run, 2, 1, new int[][] {{2, 4}, {1, 2}, {3, 0}});
        MLXArray decodeMask =
            AttentionMask.batched(
                run, new int[] {2, 4}, new int[] {0, 0}, new int[] {1, 1}, 1, 5, 0);
        MLXArray decoded =
            fixture.attention.forward(x1, cache, decodeMask, new int[] {1, 1}, decodeRotation);
        MLXArray referenceDecode =
            referenceAttention(
                fixture, x1, referenceCache, decodeMask, new int[] {1, 1}, decodeRotation);
        assertArrayEquals(referenceDecode.toFloatArray(), decoded.toFloatArray());
        assertEquals(3, cache.nextPosition(0));
        assertEquals(5, cache.nextPosition(1));
      }
    }
  }

  /**
   * The block passthrough must compose exactly input plus the rotated attention output, plus the
   * MLP of the post-attention normalization -- bit-identical to the manual composition from the
   * same retained instances.
   */
  @Test
  @Tag("full-float32")
  void blockRotationForwardMatchesManualComposition() {
    try (MLXScope scope = new MLXScope()) {
      AttentionFixture fixture = new AttentionFixture(scope, null, null);
      RMSNorm inputNorm =
          new RMSNorm(scope, MLX.array(scope, values(HIDDEN, 10), new int[] {HIDDEN}), 1e-6f);
      RMSNorm postNorm =
          new RMSNorm(scope, MLX.array(scope, values(HIDDEN, 11), new int[] {HIDDEN}), 1e-6f);
      SwiGLU mlp =
          new SwiGLU(
              scope,
              MLX.array(scope, values(HIDDEN * 32, 12), new int[] {32, HIDDEN}),
              null,
              MLX.array(scope, values(HIDDEN * 32, 13), new int[] {32, HIDDEN}),
              null,
              MLX.array(scope, values(32 * HIDDEN, 14), new int[] {HIDDEN, 32}),
              null);
      DecoderBlock block = new DecoderBlock(scope, inputNorm, fixture.attention, postNorm, mlp);
      MultiAxisRope rope = new MultiAxisRope(10_000f, SECTIONS);
      try (MLXScope run = scope.newChild()) {
        KVCache cache = new KVCache(run);
        KVCache referenceCache = new KVCache(run);
        MLXArray x = MLX.array(run, values(1 * 4 * HIDDEN, 8), new int[] {1, 4, HIDDEN});
        MultiAxisRotation rotation =
            rope.prepare(run, 1, 4, new int[][] {{1, 2, 3, 4}, {0, 1, 2, 3}, {2, 0, 1, 2}});
        MLXArray mask =
            AttentionMask.batched(run, new int[] {0}, new int[] {0}, new int[] {4}, 4, 4, 0);
        MLXArray actual = block.forward(x, cache, mask, new int[] {4}, rotation);
        MLXArray attention =
            referenceAttention(
                fixture, inputNorm.forward(x), referenceCache, mask, new int[] {4}, rotation);
        MLXArray afterAttention = MLXOps.add(x, attention);
        MLXArray reference =
            MLXOps.add(afterAttention, mlp.forward(postNorm.forward(afterAttention)));
        assertArrayEquals(reference.toFloatArray(), actual.toFloatArray());
        assertEquals(4, cache.nextPosition());
      }
    }
  }

  // ---------------------------------------------------------------------
  // rejection cases (ordinary test task, TF32=1)
  // ---------------------------------------------------------------------

  /**
   * The rotation overload must reject a missing {@code validLengths} (it always runs the batch
   * bookkeeping), a null rotation, a wrong number of lengths and a rotation prepared for a
   * different sequence or batch or with a prefix wider than the head -- all {@link
   * IllegalArgumentException} before any array is written, the cache left untouched -- and a
   * rotation whose prefix fits the head but differs from the attention's configured rotary width
   * (the case the prefix-fit check alone would let through).
   */
  @Test
  void rotationForwardRejectsInvalidArguments() {
    try (MLXScope scope = new MLXScope()) {
      AttentionFixture fixture = new AttentionFixture(scope, null, null);
      MultiAxisRope rope = new MultiAxisRope(10_000f, SECTIONS);
      try (MLXScope run = scope.newChild()) {
        KVCache cache = new KVCache(run);
        MLXArray x = MLX.array(run, values(1 * 3 * HIDDEN, 4), new int[] {1, 3, HIDDEN});
        MultiAxisRotation rotation =
            rope.prepare(run, 1, 3, new int[][] {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}});
        IllegalArgumentException missingLengths =
            assertThrows(
                IllegalArgumentException.class,
                () -> fixture.attention.forward(x, cache, null, null, rotation));
        assertTrue(missingLengths.getMessage().contains("valid lengths"));
        assertThrows(
            IllegalArgumentException.class,
            () -> fixture.attention.forward(x, cache, null, new int[] {3}, null));
        assertThrows(
            IllegalArgumentException.class,
            () -> fixture.attention.forward(x, cache, null, new int[] {3, 3}, rotation));
        // A rotation prepared for a different sequence or batch, or with a prefix wider than the
        // head, is rejected before any projection builds a graph node.
        MultiAxisRotation shortRotation =
            rope.prepare(run, 1, 2, new int[][] {{0, 1}, {1, 2}, {2, 3}});
        IllegalArgumentException wrongSequence =
            assertThrows(
                IllegalArgumentException.class,
                () -> fixture.attention.forward(x, cache, null, new int[] {3}, shortRotation));
        assertTrue(wrongSequence.getMessage().contains("must match the prepared rotation"));
        MultiAxisRotation wideBatch =
            rope.prepare(
                run,
                2,
                3,
                new int[][] {{0, 1, 0, 1, 2, 3}, {1, 2, 0, 1, 2, 3}, {2, 3, 0, 1, 2, 3}});
        assertThrows(
            IllegalArgumentException.class,
            () -> fixture.attention.forward(x, cache, null, new int[] {3}, wideBatch));
        MultiAxisRope wideRope = new MultiAxisRope(10_000f, new int[] {8, 1, 1}); // rotaryDims 10
        MultiAxisRotation wideRotation =
            wideRope.prepare(run, 1, 3, new int[][] {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}});
        IllegalArgumentException wideHead =
            assertThrows(
                IllegalArgumentException.class,
                () -> fixture.attention.forward(x, cache, null, new int[] {3}, wideRotation));
        assertTrue(wideHead.getMessage().contains("rotaryDims exceeds head dimension"));
        // A rotation prepared from a different mrope-section sum: its prefix (6) fits the head
        // (8) but differs from the attention's configured rotary width (HEAD_DIM == 8), so only
        // the width-mismatch check rejects it, and it does so before any projection is built.
        MultiAxisRope narrowRope = new MultiAxisRope(10_000f, new int[] {1, 1, 1}); // rotaryDims 6
        MultiAxisRotation narrowRotation =
            narrowRope.prepare(run, 1, 3, new int[][] {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}});
        IllegalArgumentException wrongWidth =
            assertThrows(
                IllegalArgumentException.class,
                () -> fixture.attention.forward(x, cache, null, new int[] {3}, narrowRotation));
        assertTrue(wrongWidth.getMessage().contains("does not match the configured rotary width"));
        assertEquals(0, cache.nextPosition());
      }
    }
  }

  /**
   * The block rotation overload must reject a block whose attention child is not a {@link
   * DecoderAttention} (the only cached attention with the rotation overload) before any array is
   * written, leaving the cache untouched.
   */
  @Test
  void blockRotationForwardRejectsNonDecoderAttentionChild() {
    try (MLXScope scope = new MLXScope()) {
      int embed = 16;
      GroupedQueryAttention gqa =
          new GroupedQueryAttention(
              scope,
              2,
              1,
              10_000f,
              MLX.array(scope, values(embed * embed, 0), new int[] {embed, embed}),
              null,
              MLX.array(scope, values(8 * embed, 1), new int[] {8, embed}),
              null,
              MLX.array(scope, values(8 * embed, 2), new int[] {8, embed}),
              null,
              MLX.array(scope, values(embed * embed, 3), new int[] {embed, embed}),
              null);
      SwiGLU mlp =
          new SwiGLU(
              scope,
              MLX.array(scope, values(embed * 32, 4), new int[] {32, embed}),
              null,
              MLX.array(scope, values(embed * 32, 5), new int[] {32, embed}),
              null,
              MLX.array(scope, values(32 * embed, 6), new int[] {embed, 32}),
              null);
      DecoderBlock block =
          new DecoderBlock(
              scope,
              new RMSNorm(scope, MLX.array(scope, values(embed, 7), new int[] {embed}), 1e-6f),
              gqa,
              new RMSNorm(scope, MLX.array(scope, values(embed, 8), new int[] {embed}), 1e-6f),
              mlp);
      MultiAxisRope rope = new MultiAxisRope(10_000f, SECTIONS);
      try (MLXScope run = scope.newChild()) {
        KVCache cache = new KVCache(run);
        MLXArray x = MLX.array(run, values(1 * 3 * embed, 9), new int[] {1, 3, embed});
        MultiAxisRotation rotation =
            rope.prepare(run, 1, 3, new int[][] {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}});
        IllegalArgumentException failure =
            assertThrows(
                IllegalArgumentException.class,
                () -> block.forward(x, cache, null, new int[] {3}, rotation));
        assertTrue(
            failure
                .getMessage()
                .startsWith(
                    "DecoderBlock.forward with a MultiAxisRotation requires a DecoderAttention"
                        + " child"));
        assertEquals(0, cache.nextPosition());
      }
    }
  }

  // ---------------------------------------------------------------------
  // helpers
  // ---------------------------------------------------------------------

  /**
   * The operation-identical manual composition of {@link DecoderAttention#forward(MLXArray,
   * KVCache, MLXArray, int[], MultiAxisRotation)} from the retained instances: projection to heads,
   * the optional QK norms, the rotation, the cache append, the scaled dot-product attention over
   * the retained keys, and the output projection -- the same order, shapes and scopes the seam
   * itself runs.
   */
  private static MLXArray referenceAttention(
      AttentionFixture fixture,
      MLXArray x,
      KVCache cache,
      MLXArray mask,
      int[] validLengths,
      MultiAxisRotation rotation) {
    int batch = x.shape()[0];
    int sequence = x.shape()[1];
    MLXArray q =
        AttentionHeads.toHeads(fixture.queryProj.forward(x), batch, sequence, HEADS, HEAD_DIM);
    MLXArray k =
        AttentionHeads.toHeads(
            fixture.keyProj.forward(x), batch, sequence, KEY_VALUE_HEADS, HEAD_DIM);
    if (fixture.queryNorm != null) {
      q = fixture.queryNorm.forward(q);
    }
    if (fixture.keyNorm != null) {
      k = fixture.keyNorm.forward(k);
    }
    q = rotation.rotate(q);
    k = rotation.rotate(k);
    MLXArray v =
        AttentionHeads.toHeads(
            fixture.valueProj.forward(x), batch, sequence, KEY_VALUE_HEADS, HEAD_DIM);
    cache.append(k, v, validLengths);
    MLXArray attended =
        MLXFast.scaledDotProductAttention(
            q, cache.keys(), cache.values(), SCALE, mask == null, mask, null);
    MLXArray merged = MLXShape.flatten(MLXShape.transpose(attended, new int[] {0, 2, 1, 3}), 2, 3);
    return fixture.outProj.forward(merged);
  }

  /**
   * The cross-shape reference for {@link #rotationPrefillAndDecodeBitMatchFreshPrefill}: a fresh
   * six-token prefill (the five prefill tokens followed by the decode token) through the rotation
   * seam on its own cache under {@code run}, returning the output's last row.
   */
  private static MLXArray freshSixTokenPrefillLastRow(
      AttentionFixture fixture,
      MultiAxisRope rope,
      MLXScope run,
      float[] prefill,
      float[] decodeInput) {
    float[] fresh = new float[1 * 6 * HIDDEN];
    System.arraycopy(prefill, 0, fresh, 0, 5 * HIDDEN);
    System.arraycopy(decodeInput, 0, fresh, 5 * HIDDEN, HIDDEN);
    MLXArray x6 = MLX.array(run, fresh, new int[] {1, 6, HIDDEN});
    MultiAxisRotation freshRotation =
        rope.prepare(
            run, 1, 6, new int[][] {{0, 1, 2, 3, 4, 5}, {0, 2, 4, 2, 4, 6}, {1, 1, 3, 3, 5, 4}});
    MLXArray freshMask =
        AttentionMask.batched(run, new int[] {0}, new int[] {0}, new int[] {6}, 6, 6, 0);
    KVCache freshCache = new KVCache(run);
    MLXArray freshPrefill =
        fixture.attention.forward(x6, freshCache, freshMask, new int[] {6}, freshRotation);
    return MLXShape.slice(freshPrefill, new int[] {0, 5, 0}, new int[] {1, 6, HIDDEN});
  }

  /** The shared attention fixture: identity projections plus the optional norms, retained. */
  private static final class AttentionFixture {
    final DecoderAttention attention;
    final Linear queryProj;
    final Linear keyProj;
    final Linear valueProj;
    final Linear outProj;
    final RMSNorm queryNorm;
    final RMSNorm keyNorm;

    AttentionFixture(MLXScope scope, RMSNorm queryNorm, RMSNorm keyNorm) {
      this.queryProj = new Linear(scope, identity(scope), null);
      this.keyProj = new Linear(scope, identity(scope), null);
      this.valueProj = new Linear(scope, identity(scope), null);
      this.outProj = new Linear(scope, identity(scope), null);
      this.queryNorm = queryNorm;
      this.keyNorm = keyNorm;
      this.attention =
          new DecoderAttention(
              scope,
              HEADS,
              KEY_VALUE_HEADS,
              HEAD_DIM,
              new RopeSpec.Base(10_000f),
              HEAD_DIM,
              null,
              null,
              queryProj,
              keyProj,
              valueProj,
              outProj,
              queryNorm,
              keyNorm);
    }
  }

  /** The {@code [HIDDEN, HIDDEN]} identity projection weight. */
  private static MLXArray identity(MLXScope scope) {
    float[] weight = new float[HIDDEN * HIDDEN];
    for (int i = 0; i < HIDDEN; i++) {
      weight[i * HIDDEN + i] = 1f;
    }
    return MLX.array(scope, weight, new int[] {HIDDEN, HIDDEN});
  }

  /** Distinct deterministic values in [-1, 1] with a per-case seed offset. */
  private static float[] values(int n, int seed) {
    float[] data = new float[n];
    for (int j = 0; j < n; j++) {
      data[j] = (((j + seed) % 13) - 6) / 6.0f;
    }
    return data;
  }

  private static int[] zeros(int n) {
    return new int[n];
  }
}
