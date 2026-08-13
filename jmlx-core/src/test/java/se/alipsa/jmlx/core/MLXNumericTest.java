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
 * hand-computed values. reshape/transpose additionally assert element values, not just shape -- the
 * only thing that catches the contiguity bug in {@link MLXArray#toFloatArray()}.
 */
@EnabledIfNativeAvailable
class MLXNumericTest {

  private static final float EPS = 1e-5f;

  @Test
  void add() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      assertArrayEquals(new float[] {11, 22, 33}, MLXOps.add(a, b).toFloatArray(), EPS);
    }
  }

  /**
   * See req/phase4-plan.md §2: {@code binaryOp} now resolves its target via {@code scopeOf}/{@code
   * innermost} rather than rejecting any pair of operands that are not from the exact same scope --
   * this is "it stops rejecting parent/child pairs, which is the entire point" (§2), checked in
   * both operand orders since picking "the first operand's scope" would silently pass one order and
   * fail the other.
   */
  @Test
  void addAcrossParentAndChildScopeAllocatesIntoTheChildRegardlessOfOperandOrder() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray w = MLX.array(parent, new float[] {1, 2, 3}, new int[] {3});
      try (MLXScope child = parent.newChild()) {
        MLXArray x = MLX.array(child, new float[] {10, 20, 30}, new int[] {3});
        MLXArray r1 = MLXOps.add(x, w);
        MLXArray r2 = MLXOps.add(w, x);
        assertSame(child, r1.scope());
        assertSame(child, r2.scope());
        assertArrayEquals(new float[] {11, 22, 33}, r1.toFloatArray(), EPS);
        assertArrayEquals(new float[] {11, 22, 33}, r2.toFloatArray(), EPS);
      }
    }
  }

  /**
   * Proves the ancestor relaxation did not become "anything goes": two independent root scopes are
   * still rejected, even though they are no longer rejected merely for being unequal.
   */
  @Test
  void addAcrossTwoUnrelatedRootScopesThrows() {
    try (MLXScope scopeA = new MLXScope();
        MLXScope scopeB = new MLXScope()) {
      MLXArray a = MLX.array(scopeA, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scopeB, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> MLXOps.add(a, b));
    }
  }

  @Test
  void subtract() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertArrayEquals(new float[] {9, 18, 27}, MLXOps.subtract(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void multiply() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {4, 5, 6}, new int[] {3});
      assertArrayEquals(new float[] {4, 10, 18}, MLXOps.multiply(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void divide() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {2, 4, 5}, new int[] {3});
      assertArrayEquals(new float[] {5, 5, 6}, MLXOps.divide(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void matmul() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray b = MLX.array(scope, new float[] {5, 6, 7, 8}, new int[] {2, 2});
      assertArrayEquals(new float[] {19, 22, 43, 50}, MLXOps.matmul(a, b).toFloatArray(), 1e-3f);
    }
  }

  @Test
  void addBroadcastsRowVectorAgainstMatrix() {
    try (MLXScope scope = new MLXScope()) {
      // [2,3] + [3]: the row vector broadcasts against every row of the matrix.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray b = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      MLXArray result = MLXOps.add(a, b);
      assertArrayEquals(new int[] {2, 3}, result.shape());
      assertArrayEquals(new float[] {11, 22, 33, 14, 25, 36}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void addBroadcastsSingletonLeadingDimension() {
    try (MLXScope scope = new MLXScope()) {
      // [2,3] + [1,3]: same broadcast as the row-vector case, expressed with an explicit size-1
      // axis.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray b = MLX.array(scope, new float[] {100, 200, 300}, new int[] {1, 3});
      MLXArray result = MLXOps.add(a, b);
      assertArrayEquals(new int[] {2, 3}, result.shape());
      assertArrayEquals(new float[] {101, 202, 303, 104, 205, 306}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void addBroadcastsRankZeroScalarAgainstMatrix() {
    try (MLXScope scope = new MLXScope()) {
      // [2,2] + a rank-0 scalar: rank-0 is compatible with anything (the compare loop never runs).
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray scalar = MLX.array(scope, new float[] {10}, new int[] {});
      MLXArray result = MLXOps.add(a, scalar);
      assertArrayEquals(new int[] {2, 2}, result.shape());
      assertArrayEquals(new float[] {11, 12, 13, 14}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void addRejectsIncompatibleShapes() {
    try (MLXScope scope = new MLXScope()) {
      // [2,3] and [4]: last dims 3 vs 4, neither equal nor 1 -- the Java guard must fire
      // before ever reaching native, unlike MLXNativeErrorTest's deliberate addUnchecked bypass.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray b = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      assertThrows(IllegalArgumentException.class, () -> MLXOps.add(a, b));
    }
  }

  @Test
  void matmulOfTwoRank1VectorsIsADotProduct() {
    try (MLXScope scope = new MLXScope()) {
      // Both operands rank-1: mlx promotes a->[1,3], b->[3,1], matmuls, then drops both
      // promoted axes back out -- a rank-0 scalar, matching NumPy's 1-D matmul convention.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {4, 5, 6}, new int[] {3});
      MLXArray result = MLXOps.matmul(a, b);
      assertArrayEquals(new int[] {}, result.shape());
      assertArrayEquals(new float[] {32}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void matmulOfRank1VectorAndRank2MatrixDropsThePromotedAxis() {
    try (MLXScope scope = new MLXScope()) {
      // a rank-1 [3] promoted to [1,3]; b rank-2 [3,2] unchanged. Result [1,2] drops
      // back to rank-1 [2], matching NumPy's vector-times-matrix convention.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {3, 2});
      MLXArray result = MLXOps.matmul(a, b);
      assertArrayEquals(new int[] {2}, result.shape());
      assertArrayEquals(new float[] {22, 28}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void matmulOfRank2MatrixAndRank1VectorDropsThePromotedAxis() {
    try (MLXScope scope = new MLXScope()) {
      // a rank-2 [2,2] unchanged; b rank-1 [2] promoted to [2,1]. Result [2,1] drops
      // back to rank-1 [2], matching NumPy's matrix-times-vector convention.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray b = MLX.array(scope, new float[] {5, 6}, new int[] {2});
      MLXArray result = MLXOps.matmul(a, b);
      assertArrayEquals(new int[] {2}, result.shape());
      assertArrayEquals(new float[] {17, 39}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void matmulBatchBroadcastsOverLeadingDimensions() {
    try (MLXScope scope = new MLXScope()) {
      // [2,3,4] x [4,5] -> [4,5] broadcasts across the batch dimension implied by [2,3,4]'s
      // leading axes, giving [2,3,5]. All-ones operands make every output element
      // the shared inner dimension's length (4), so the batched shape is what's under test.
      float[] onesA = new float[2 * 3 * 4];
      java.util.Arrays.fill(onesA, 1f);
      float[] onesB = new float[4 * 5];
      java.util.Arrays.fill(onesB, 1f);
      MLXArray a = MLX.array(scope, onesA, new int[] {2, 3, 4});
      MLXArray b = MLX.array(scope, onesB, new int[] {4, 5});
      MLXArray result = MLXOps.matmul(a, b);
      assertArrayEquals(new int[] {2, 3, 5}, result.shape());
      float[] expected = new float[2 * 3 * 5];
      java.util.Arrays.fill(expected, 4f);
      assertArrayEquals(expected, result.toFloatArray(), EPS);
    }
  }

  @Test
  void matmulRejectsRankZeroOperand() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray scalar = MLX.array(scope, new float[] {1}, new int[] {});
      MLXArray matrix = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      assertThrows(IllegalArgumentException.class, () -> MLXOps.matmul(scalar, matrix));
      assertThrows(IllegalArgumentException.class, () -> MLXOps.matmul(matrix, scalar));
    }
  }

  @Test
  void matmulSucceedsWithOneExactAndOneInexactOperand() {
    try (MLXScope scope = new MLXScope()) {
      // promote_types(float32, int32) == float32, so native accepts this mixed pair even though
      // requireMatmulCompatible's Java guard only requires ONE operand to be inexact
      // (req/phase4-plan.md §4).
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray identity =
          MLX.astype(MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {2, 2}), DType.INT32);
      assertEquals(DType.INT32, identity.dtype());
      assertArrayEquals(new float[] {1, 2, 3, 4}, MLXOps.matmul(a, identity).toFloatArray(), EPS);
    }
  }

  @Test
  void matmulRejectsTwoExactOperands() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a =
          MLX.astype(MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2}), DType.INT32);
      MLXArray b =
          MLX.astype(MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {2, 2}), DType.INT32);
      assertThrows(IllegalArgumentException.class, () -> MLXOps.matmul(a, b));
    }
  }

  @Test
  void astypeRoundTripsThroughInt32AndBackToFloat32() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray f = MLX.array(scope, new float[] {1.75f, -2.25f, 3f}, new int[] {3});
      MLXArray i = MLX.astype(f, DType.INT32);
      assertEquals(DType.INT32, i.dtype());
      assertArrayEquals(new int[] {1, -2, 3}, i.toIntArray());
      MLXArray back = MLX.astype(i, DType.FLOAT32);
      assertArrayEquals(new float[] {1, -2, 3}, back.toFloatArray(), EPS);
    }
  }

  @Test
  void toFloatArrayReadsBackAFloat16ArrayThroughTheAstypeStep() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray f32 = MLX.array(scope, new float[] {1.5f, 2.5f, -3.5f}, new int[] {3});
      MLXArray f16 = MLX.astype(f32, DType.FLOAT16);
      assertEquals(DType.FLOAT16, f16.dtype());
      assertArrayEquals(new float[] {1.5f, 2.5f, -3.5f}, f16.toFloatArray(), EPS);
    }
  }

  @Test
  void toFloatArrayReadsBackABfloat16ArrayThroughTheAstypeStep() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray f32 = MLX.array(scope, new float[] {1.5f, 2.5f, -3.5f}, new int[] {3});
      MLXArray bf16 = MLX.astype(f32, DType.BFLOAT16);
      assertEquals(DType.BFLOAT16, bf16.dtype());
      assertArrayEquals(new float[] {1.5f, 2.5f, -3.5f}, bf16.toFloatArray(), EPS);
    }
  }

  @Test
  void toFloatArrayRejectsAnExactDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray i = MLX.astype(MLX.array(scope, new float[] {1, 2}, new int[] {2}), DType.INT32);
      assertThrows(IllegalStateException.class, i::toFloatArray);
    }
  }

  @Test
  void toIntArrayRejectsANonInt32Dtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray f = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      assertThrows(IllegalStateException.class, f::toIntArray);
    }
  }

  @Test
  void astypeToBoolAndUint32RoundTripTheirDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray f = MLX.array(scope, new float[] {0, 1, 2}, new int[] {3});
      assertEquals(DType.BOOL, MLX.astype(f, DType.BOOL).dtype());
      assertEquals(DType.UINT32, MLX.astype(f, DType.UINT32).dtype());
    }
  }

  @Test
  void arrayIntOverloadCreatesAnInt32ArrayFromRowMajorData() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new int[] {1, 2, 3, 4}, new int[] {2, 2});
      assertEquals(DType.INT32, a.dtype());
      assertArrayEquals(new int[] {1, 2, 3, 4}, a.toIntArray());
    }
  }

  @Test
  void zerosAndOnesFillEveryElement() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray z = MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32);
      assertArrayEquals(new float[] {0, 0, 0, 0}, z.toFloatArray(), EPS);
      MLXArray o = MLX.ones(scope, new int[] {3}, DType.INT32);
      assertEquals(DType.INT32, o.dtype());
      assertArrayEquals(new int[] {1, 1, 1}, o.toIntArray());
    }
  }

  @Test
  void fullFillsFloat32WithTheGivenValue() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray f = MLX.full(scope, new int[] {2, 2}, 7f, DType.FLOAT32);
      assertArrayEquals(new float[] {7, 7, 7, 7}, f.toFloatArray(), EPS);
    }
  }

  @Test
  void fullTruncatesTheFillValueForAnInt32Target() {
    try (MLXScope scope = new MLXScope()) {
      // 5.9f -> mlx_array_new_int((int) 5.9f) == 5, matching Java's own truncating cast.
      MLXArray f = MLX.full(scope, new int[] {3}, 5.9f, DType.INT32);
      assertArrayEquals(new int[] {5, 5, 5}, f.toIntArray());
    }
  }

  @Test
  void fullBuildsABoolFillViaTheBoolScalarConstructor() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray f = MLX.full(scope, new int[] {2}, 1f, DType.BOOL);
      assertEquals(DType.BOOL, f.dtype());
      assertArrayEquals(new int[] {1, 1}, MLX.astype(f, DType.INT32).toIntArray());
    }
  }

  @Test
  void fullRejectsANegativeValueForAUint32Target() {
    try (MLXScope scope = new MLXScope()) {
      // UINT32 falls to the float32 scalar branch and mlx_full then astypes
      // that scalar to uint32 -- static_cast<uint32_t>(-1.0f) is UB in C++,
      // so this is rejected in Java before it ever reaches native.
      assertThrows(
          IllegalArgumentException.class, () -> MLX.full(scope, new int[] {2}, -1f, DType.UINT32));
    }
  }

  @Test
  void fullRejectsNanForAUint32Target() {
    try (MLXScope scope = new MLXScope()) {
      // NaN < 0 is false, so a negative-only guard would let this through;
      // static_cast<uint32_t>(NaN) is UB regardless of sign.
      assertThrows(
          IllegalArgumentException.class,
          () -> MLX.full(scope, new int[] {2}, Float.NaN, DType.UINT32));
    }
  }

  @Test
  void fullRejectsAnOutOfRangeValueForAUint32Target() {
    try (MLXScope scope = new MLXScope()) {
      // 1e20f is positive and would pass a negative-only guard, but it is
      // far above UINT32_MAX -- static_cast<uint32_t>(1e20f) is UB.
      assertThrows(
          IllegalArgumentException.class,
          () -> MLX.full(scope, new int[] {2}, 1e20f, DType.UINT32));
    }
  }

  @Test
  void arangeWithAnIntegerStepProducesIntegerCounts() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.arange(scope, 0, 5, 1, DType.INT32);
      assertArrayEquals(new int[] {0, 1, 2, 3, 4}, a.toIntArray());
    }
  }

  @Test
  void arangeRejectsAZeroStep() {
    try (MLXScope scope = new MLXScope()) {
      // Without this guard, start == stop with step == 0 reaches native's
      // static_cast<int>(0.0 / 0.0) -- a NaN operand, C++ undefined
      // behaviour neither the isnan(start/step/stop) nor the real_size >
      // INT_MAX checks upstream catch.
      IllegalArgumentException ex =
          assertThrows(
              IllegalArgumentException.class, () -> MLX.arange(scope, 5, 5, 0, DType.FLOAT32));
      assertEquals("arange: step must not be 0", ex.getMessage());
    }
  }

  @Test
  void arangeWithAFractionalStepProducesFractionalCounts() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.arange(scope, 0, 1, 0.25, DType.FLOAT32);
      assertArrayEquals(new float[] {0, 0.25f, 0.5f, 0.75f}, a.toFloatArray(), EPS);
    }
  }

  @Test
  void stopGradientPreservesTheForwardValueAndDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray blocked = MLXOps.stopGradient(a);
      assertEquals(a.dtype(), blocked.dtype());
      assertArrayEquals(new float[] {1, 2, 3}, blocked.toFloatArray(), EPS);
    }
  }

  @Test
  void sum() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      assertArrayEquals(new float[] {10}, MLXOps.sum(a).toFloatArray(), EPS);
    }
  }

  @Test
  void exp() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {0f, 1f}, new int[] {2});
      float[] result = MLXOps.exp(a).toFloatArray();
      assertEquals(1.0f, result[0], EPS);
      assertEquals((float) Math.E, result[1], 1e-3f);
    }
  }

  @Test
  void reshapePreservesFlatOrderAndUpdatesShape() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray reshaped = MLXShape.reshape(a, new int[] {3, 2});
      assertArrayEquals(new int[] {3, 2}, reshaped.shape());
      assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6}, reshaped.toFloatArray(), EPS);
    }
  }

  @Test
  void transposeReordersElementsNotJustShape() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2,3],[4,5,6]] transposed -> [[1,4],[2,5],[3,6]]
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray transposed = MLXShape.transpose(a);
      assertArrayEquals(new int[] {3, 2}, transposed.shape());
      // Exactly the case that breaks without the mlx_contiguous fix in
      // toFloatArray(): a naive strided read would return the
      // pre-transpose flat order {1,2,3,4,5,6} instead.
      assertArrayEquals(new float[] {1, 4, 2, 5, 3, 6}, transposed.toFloatArray(), EPS);
    }
  }

  @Test
  void transposeAxesPermutesWithoutFullyReversing() {
    try (MLXScope scope = new MLXScope()) {
      // shape [2,3,2], values 1..12 in row-major order. axes=[1,0,2] swaps only the
      // first two axes -- unlike transpose(a)'s full reversal, axis 2 stays put -- so
      // this is the permutation transpose(a) alone can never exercise.
      float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
      MLXArray a = MLX.array(scope, data, new int[] {2, 3, 2});
      MLXArray transposed = MLXShape.transpose(a, new int[] {1, 0, 2});
      assertArrayEquals(new int[] {3, 2, 2}, transposed.shape());
      assertArrayEquals(
          new float[] {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}, transposed.toFloatArray(), EPS);
    }
  }

  @Test
  void transposeAxesRejectsANonPermutation() {
    try (MLXScope scope = new MLXScope()) {
      // {0, 0, 2} repeats axis 0 and omits axis 1: not a permutation of
      // 0..ndim()-1. Unlike the Java-side guards elsewhere in this class,
      // there is no Java guard here -- native's own validation rejects
      // this, so it surfaces as MLXException, not IllegalArgumentException.
      // Pinned here so that stays a deliberate fact, not an accident nobody
      // noticed changed.
      MLXArray a =
          MLX.array(
              scope, new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[] {2, 3, 2});
      assertThrows(MLXException.class, () -> MLXShape.transpose(a, new int[] {0, 0, 2}));
    }
  }

  @Test
  void log() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1f, (float) Math.E}, new int[] {2});
      float[] result = MLXOps.log(a).toFloatArray();
      assertEquals(0f, result[0], EPS);
      assertEquals(1f, result[1], EPS);
    }
  }

  @Test
  void sin() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {0f, (float) (Math.PI / 2)}, new int[] {2});
      float[] result = MLXOps.sin(a).toFloatArray();
      assertEquals(0f, result[0], EPS);
      assertEquals(1f, result[1], EPS);
    }
  }

  @Test
  void cos() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {0f, (float) (Math.PI / 2)}, new int[] {2});
      float[] result = MLXOps.cos(a).toFloatArray();
      assertEquals(1f, result[0], EPS);
      assertEquals(0f, result[1], 1e-6f);
    }
  }

  @Test
  void sigmoid() {
    try (MLXScope scope = new MLXScope()) {
      // irrational values; use 1e-4f tolerance
      MLXArray a = MLX.array(scope, new float[] {0f, 1f, -1f}, new int[] {3});
      float[] result = MLXOps.sigmoid(a).toFloatArray();
      assertEquals(0.5f, result[0], 1e-4f);
      assertEquals(0.7310586f, result[1], 1e-4f);
      assertEquals(0.2689414f, result[2], 1e-4f);
    }
  }

  @Test
  void erf() {
    try (MLXScope scope = new MLXScope()) {
      // irrational values; use 1e-4f tolerance
      MLXArray a = MLX.array(scope, new float[] {0f, 1f}, new int[] {2});
      float[] result = MLXOps.erf(a).toFloatArray();
      assertEquals(0.0f, result[0], 1e-4f);
      assertEquals(0.8427008f, result[1], 1e-4f);
    }
  }

  @Test
  void tanh() {
    try (MLXScope scope = new MLXScope()) {
      // irrational values; use 1e-4f tolerance
      MLXArray a = MLX.array(scope, new float[] {0f, 1f, -1f}, new int[] {3});
      float[] result = MLXOps.tanh(a).toFloatArray();
      assertEquals(0.0f, result[0], 1e-4f);
      assertEquals(0.7615942f, result[1], 1e-4f);
      assertEquals(-0.7615942f, result[2], 1e-4f);
    }
  }

  @Test
  void sqrt() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {4, 9, 16}, new int[] {3});
      assertArrayEquals(new float[] {2, 3, 4}, MLXOps.sqrt(a).toFloatArray(), EPS);
    }
  }

  @Test
  void rsqrt() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {4, 16}, new int[] {2});
      assertArrayEquals(new float[] {0.5f, 0.25f}, MLXOps.rsqrt(a).toFloatArray(), EPS);
    }
  }

  @Test
  void square() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {2, -3, 4}, new int[] {3});
      assertArrayEquals(new float[] {4, 9, 16}, MLXOps.square(a).toFloatArray(), EPS);
    }
  }

  @Test
  void negative() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {2, -3, 0}, new int[] {3});
      assertArrayEquals(new float[] {-2, 3, 0}, MLXOps.negative(a).toFloatArray(), EPS);
    }
  }

  @Test
  void maximum() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 5, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {4, 2, 3}, new int[] {3});
      assertArrayEquals(new float[] {4, 5, 3}, MLXOps.maximum(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void maximumRejectsIncompatibleShapes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      assertThrows(IllegalArgumentException.class, () -> MLXOps.maximum(a, b));
    }
  }

  @Test
  void power() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {2, 3, 4}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {3, 2, 1}, new int[] {3});
      assertArrayEquals(new float[] {8, 9, 4}, MLXOps.power(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void powerRejectsIncompatibleShapes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {2, 3}, new int[] {2});
      MLXArray b = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> MLXOps.power(a, b));
    }
  }

  @Test
  void sumWithAxesAndKeepdimsFalse() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray result = MLXOps.sum(a, new int[] {1}, false);
      assertArrayEquals(new int[] {2}, result.shape());
      assertArrayEquals(new float[] {6, 15}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void sumWithAxesAndKeepdimTrue() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray result = MLXOps.sum(a, new int[] {0}, true);
      assertArrayEquals(new int[] {1, 3}, result.shape());
      assertArrayEquals(new float[] {5, 7, 9}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void meanWithAxesAndKeepdimsFalse() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray result = MLXOps.mean(a, new int[] {1}, false);
      assertArrayEquals(new int[] {2}, result.shape());
      assertArrayEquals(new float[] {2, 5}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void meanWithAxesAndKeepdimTrue() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray result = MLXOps.mean(a, new int[] {0}, true);
      assertArrayEquals(new int[] {1, 3}, result.shape());
      assertArrayEquals(new float[] {2.5f, 3.5f, 4.5f}, result.toFloatArray(), EPS);
    }
  }

  /**
   * All other sum/mean-with-axes tests above pass a single-element {@code axes} array (e.g. {@code
   * {0}} or {@code {1}}) -- nothing in those would catch a bug where the implementation passed a
   * hardcoded {@code 1} instead of {@code axes.length} as the native axes count. A genuinely
   * multi-element {@code axes} array on a rank-3 input is the only way to distinguish the two.
   */
  @Test
  void sumAndMeanWithMultiElementAxesReduceOverBothGivenAxes() {
    try (MLXScope scope = new MLXScope()) {
      // shape [2,3,2], values 1..12 in row-major order. Reducing over axes
      // {0,1} leaves only the last axis: for k=0, sum of {1,3,5,7,9,11} = 36;
      // for k=1, sum of {2,4,6,8,10,12} = 42. Each set has 6 elements, so
      // mean is 6 and 7 respectively.
      float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
      MLXArray a = MLX.array(scope, data, new int[] {2, 3, 2});
      MLXArray sumResult = MLXOps.sum(a, new int[] {0, 1}, false);
      assertArrayEquals(new int[] {2}, sumResult.shape());
      assertArrayEquals(new float[] {36, 42}, sumResult.toFloatArray(), EPS);

      MLXArray meanResult = MLXOps.mean(a, new int[] {0, 1}, false);
      assertArrayEquals(new int[] {2}, meanResult.shape());
      assertArrayEquals(new float[] {6, 7}, meanResult.toFloatArray(), EPS);
    }
  }

  @Test
  void innerOfTwoRank1VectorsIsADotProduct() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {4, 5, 6}, new int[] {3});
      MLXArray result = MLXOps.inner(a, b);
      assertArrayEquals(new int[] {}, result.shape());
      assertArrayEquals(new float[] {32}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void innerRejectsMismatchedLastDimension() {
    try (MLXScope scope = new MLXScope()) {
      // Both operands rank-1 (>= 1), so the rank-0 skip in the guard cannot mask this:
      // last dimensions 3 vs 4 genuinely disagree and must be rejected before native ever runs.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      assertThrows(IllegalArgumentException.class, () -> MLXOps.inner(a, b));
    }
  }

  @Test
  void outerComputesShapeAndValues() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      MLXArray b = MLX.array(scope, new float[] {3, 4, 5}, new int[] {3});
      MLXArray result = MLXOps.outer(a, b);
      assertArrayEquals(new int[] {2, 3}, result.shape());
      assertArrayEquals(new float[] {3, 4, 5, 6, 8, 10}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void broadcastToSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray result = MLXShape.broadcastTo(a, new int[] {2, 3});
      assertArrayEquals(new int[] {2, 3}, result.shape());
      assertArrayEquals(new float[] {1, 2, 3, 1, 2, 3}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void broadcastToRejectsShapeThatFailsTheDirectionalCheck() {
    try (MLXScope scope = new MLXScope()) {
      // [3] -> [1] satisfies the symmetric broadcast_shapes rule (1 == 1 after
      // right-alignment padding never applies here) but native's directional
      // check for broadcastTo specifically rejects shrinking a real dimension.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> MLXShape.broadcastTo(a, new int[] {1}));
    }
  }

  @Test
  void broadcastToRejectsNegativeTargetDimension() {
    try (MLXScope scope = new MLXScope()) {
      // Without the non-negative check on targetShape, 1 == 1 would make the
      // directional check itself pass, and this would instead throw MLXException
      // from native -- assert the Java-side exception type specifically.
      MLXArray a = MLX.array(scope, new float[] {1}, new int[] {1});
      assertThrows(IllegalArgumentException.class, () -> MLXShape.broadcastTo(a, new int[] {-1}));
    }
  }

  @Test
  void squeezeRemovesAllSizeOneAxes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3, 1});
      MLXArray result = MLXShape.squeeze(a);
      assertArrayEquals(new int[] {3}, result.shape());
      assertArrayEquals(new float[] {1, 2, 3}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void squeezeAxesRemovesOnlyTheGivenAxes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3, 1});
      MLXArray result = MLXShape.squeeze(a, new int[] {0});
      assertArrayEquals(new int[] {3, 1}, result.shape());
      assertArrayEquals(new float[] {1, 2, 3}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void squeezeAxesRejectsAnAxisWithSizeNotOne() {
    try (MLXScope scope = new MLXScope()) {
      // Axis 1 has size 3, not 1, so it cannot be squeezed. As with
      // transposeAxesRejectsANonPermutation, there is no Java-side guard
      // here -- native validates and rejects it, surfacing as
      // MLXException rather than IllegalArgumentException. Pinned so this
      // stays a deliberate, known fact rather than an untested assumption.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3, 1});
      assertThrows(MLXException.class, () -> MLXShape.squeeze(a, new int[] {1}));
    }
  }

  @Test
  void sliceRejectsLengthMismatchWithNdim() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      assertThrows(
          IllegalArgumentException.class, () -> MLXShape.slice(a, new int[] {0}, new int[] {2, 3}));
    }
  }

  @Test
  void sliceWithNegativeStartSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      // start=-3 on a length-5 axis normalizes to 5-3=2, NumPy-style -- must not throw,
      // and must select elements [2,3,4] (values 3,4,5), not some clamped/rejected result.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5}, new int[] {5});
      MLXArray result = MLXShape.slice(a, new int[] {-3}, new int[] {5});
      assertArrayEquals(new int[] {3}, result.shape());
      assertArrayEquals(new float[] {3, 4, 5}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void sliceWithNegativeStrideReversesADimension() {
    try (MLXScope scope = new MLXScope()) {
      // start=3, stop=1, stride=-1 on [1,2,3,4,5]: walks index 3 down to (but excluding)
      // index 1, i.e. indices {3,2} -> values {4,3}. Must not throw: native's reverse form.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5}, new int[] {5});
      MLXArray result = MLXShape.slice(a, new int[] {3}, new int[] {1}, new int[] {-1});
      assertArrayEquals(new int[] {2}, result.shape());
      assertArrayEquals(new float[] {4, 3}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void sliceRejectsZeroStride() {
    try (MLXScope scope = new MLXScope()) {
      // Without this guard, native does not throw at all -- it silently returns a
      // zero-sized dimension (aarch64 sdiv-by-zero returns 0 rather than trapping).
      // A test asserting only "does not succeed" would pass against that broken
      // behaviour too, so assert both the exception type and the message.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      IllegalArgumentException ex =
          assertThrows(
              IllegalArgumentException.class,
              () -> MLXShape.slice(a, new int[] {0}, new int[] {4}, new int[] {0}));
      assertEquals("slice: strides[0] must not be 0 (slice step cannot be zero)", ex.getMessage());
    }
  }

  @Test
  void sliceContiguousSelectsExpectedSubrange() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2,3],[4,5,6]], columns 1..2 of both rows -> [[2,3],[5,6]].
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray result = MLXShape.slice(a, new int[] {0, 1}, new int[] {2, 3});
      assertArrayEquals(new int[] {2, 2}, result.shape());
      assertArrayEquals(new float[] {2, 3, 5, 6}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void sliceStridedSelectsEveryOtherElementInOrder() {
    try (MLXScope scope = new MLXScope()) {
      // Stride 2 over 8 elements yields a non-contiguous view: exactly the case that
      // exercises toFloatArray()'s mlx_contiguous path, the way
      // transposeReordersElementsNotJustShape does for transpose.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6, 7, 8}, new int[] {8});
      MLXArray result = MLXShape.slice(a, new int[] {0}, new int[] {8}, new int[] {2});
      assertArrayEquals(new int[] {4}, result.shape());
      assertArrayEquals(new float[] {1, 3, 5, 7}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void swapaxesSwapsTheGivenAxes() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2,3],[4,5,6]], swapaxes(0,1) -> [[1,4],[2,5],[3,6]]
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray result = MLXShape.swapaxes(a, 0, 1);
      assertArrayEquals(new int[] {3, 2}, result.shape());
      assertArrayEquals(new float[] {1, 4, 2, 5, 3, 6}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void takeFlattenedArrayAtIndices() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2],[3,4]] flattened is [1,2,3,4]; take indices [0,3,1] -> [1,4,2]
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray indices = MLX.array(scope, new int[] {0, 3, 1}, new int[] {3});
      MLXArray result = MLXShape.take(a, indices);
      assertArrayEquals(new int[] {3}, result.shape());
      assertArrayEquals(new float[] {1, 4, 2}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void takeAxisSelectsSlicesAlongAnAxis() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2],[3,4],[5,6]], takeAxis along axis 0 with indices [2,0] -> [[5,6],[1,2]]
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {3, 2});
      MLXArray indices = MLX.array(scope, new int[] {2, 0}, new int[] {2});
      MLXArray result = MLXShape.takeAxis(a, indices, 0);
      assertArrayEquals(new int[] {2, 2}, result.shape());
      assertArrayEquals(new float[] {5, 6, 1, 2}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void transposeExplicitTargetAllocatesIntoTargetScope() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray result;
      try (MLXScope child = parent.newChild()) {
        MLXArray a = MLX.array(child, new float[] {1, 2, 3, 4}, new int[] {2, 2});
        result = MLXShape.transpose(a, parent);
        assertSame(parent, result.scope());
      }
      // Close the child scope; now verify the result is still readable after the child is
      // closed. This proves the view actually escaped the child before it closed, not just
      // that the call didn't throw.
      assertArrayEquals(new float[] {1, 3, 2, 4}, result.toFloatArray(), EPS);
    }
  }

  /**
   * The other side of {@link #transposeExplicitTargetAllocatesIntoTargetScope}: an explicit target
   * unrelated to {@code a.scope()} (two independent roots, neither an ancestor nor a descendant of
   * the other) must be rejected -- {@code NativeOps.unaryOp}'s relatedness check exists
   * specifically so a result never references an operand from a scope that could close before or
   * long after it.
   */
  @Test
  void transposeExplicitTargetRejectsAnUnrelatedScope() {
    try (MLXScope s1 = new MLXScope();
        MLXScope s2 = new MLXScope()) {
      MLXArray a = MLX.array(s1, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      assertThrows(IllegalArgumentException.class, () -> MLXShape.transpose(a, s2));
    }
  }
}
