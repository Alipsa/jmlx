package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/initial-plan.md, Testing approach, "Numeric correctness": every facade op against hand-computed values.
 * reshape/transpose additionally assert element values, not just shape -- the only thing that catches the contiguity
 * bug in {@link MLXArray#toFloatArray()}.
 */
@EnabledIfNativeAvailable
class MLXNumericTest {

  private static final float EPS = 1e-5f;

  @Test
  void add() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      assertArrayEquals(new float[] {11, 22, 33}, MLX.add(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void subtract() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertArrayEquals(new float[] {9, 18, 27}, MLX.subtract(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void multiply() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {4, 5, 6}, new int[] {3});
      assertArrayEquals(new float[] {4, 10, 18}, MLX.multiply(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void divide() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {2, 4, 5}, new int[] {3});
      assertArrayEquals(new float[] {5, 5, 6}, MLX.divide(a, b).toFloatArray(), EPS);
    }
  }

  @Test
  void matmul() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray b = MLX.array(scope, new float[] {5, 6, 7, 8}, new int[] {2, 2});
      assertArrayEquals(new float[] {19, 22, 43, 50}, MLX.matmul(a, b).toFloatArray(), 1e-3f);
    }
  }

  @Test
  void addBroadcastsRowVectorAgainstMatrix() {
    try (MLXScope scope = new MLXScope()) {
      // [2,3] + [3]: the row vector broadcasts against every row of the matrix.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray b = MLX.array(scope, new float[] {10, 20, 30}, new int[] {3});
      MLXArray result = MLX.add(a, b);
      assertArrayEquals(new int[] {2, 3}, result.shape());
      assertArrayEquals(new float[] {11, 22, 33, 14, 25, 36}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void addBroadcastsSingletonLeadingDimension() {
    try (MLXScope scope = new MLXScope()) {
      // [2,3] + [1,3]: same broadcast as the row-vector case, expressed with an explicit size-1 axis.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray b = MLX.array(scope, new float[] {100, 200, 300}, new int[] {1, 3});
      MLXArray result = MLX.add(a, b);
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
      MLXArray result = MLX.add(a, scalar);
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
      assertThrows(IllegalArgumentException.class, () -> MLX.add(a, b));
    }
  }

  @Test
  void matmulOfTwoRank1VectorsIsADotProduct() {
    try (MLXScope scope = new MLXScope()) {
      // Both operands rank-1: mlx promotes a->[1,3], b->[3,1], matmuls, then drops both
      // promoted axes back out -- a rank-0 scalar, matching NumPy's 1-D matmul convention.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {4, 5, 6}, new int[] {3});
      MLXArray result = MLX.matmul(a, b);
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
      MLXArray result = MLX.matmul(a, b);
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
      MLXArray result = MLX.matmul(a, b);
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
      MLXArray result = MLX.matmul(a, b);
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
      assertThrows(IllegalArgumentException.class, () -> MLX.matmul(scalar, matrix));
      assertThrows(IllegalArgumentException.class, () -> MLX.matmul(matrix, scalar));
    }
  }

  @Test
  void sum() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {4});
      assertArrayEquals(new float[] {10}, MLX.sum(a).toFloatArray(), EPS);
    }
  }

  @Test
  void exp() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {0f, 1f}, new int[] {2});
      float[] result = MLX.exp(a).toFloatArray();
      assertEquals(1.0f, result[0], EPS);
      assertEquals((float) Math.E, result[1], 1e-3f);
    }
  }

  @Test
  void reshapePreservesFlatOrderAndUpdatesShape() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray reshaped = MLX.reshape(a, new int[] {3, 2});
      assertArrayEquals(new int[] {3, 2}, reshaped.shape());
      assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6}, reshaped.toFloatArray(), EPS);
    }
  }

  @Test
  void transposeReordersElementsNotJustShape() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2,3],[4,5,6]] transposed -> [[1,4],[2,5],[3,6]]
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray transposed = MLX.transpose(a);
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
      MLXArray transposed = MLX.transpose(a, new int[] {1, 0, 2});
      assertArrayEquals(new int[] {3, 2, 2}, transposed.shape());
      assertArrayEquals(new float[] {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}, transposed.toFloatArray(), EPS);
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
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[] {2, 3, 2});
      assertThrows(MLXException.class, () -> MLX.transpose(a, new int[] {0, 0, 2}));
    }
  }

  @Test
  void log() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1f, (float) Math.E}, new int[] {2});
      float[] result = MLX.log(a).toFloatArray();
      assertEquals(0f, result[0], EPS);
      assertEquals(1f, result[1], EPS);
    }
  }

  @Test
  void sin() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {0f, (float) (Math.PI / 2)}, new int[] {2});
      float[] result = MLX.sin(a).toFloatArray();
      assertEquals(0f, result[0], EPS);
      assertEquals(1f, result[1], EPS);
    }
  }

  @Test
  void cos() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {0f, (float) (Math.PI / 2)}, new int[] {2});
      float[] result = MLX.cos(a).toFloatArray();
      assertEquals(1f, result[0], EPS);
      assertEquals(0f, result[1], 1e-6f);
    }
  }

  @Test
  void innerOfTwoRank1VectorsIsADotProduct() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {4, 5, 6}, new int[] {3});
      MLXArray result = MLX.inner(a, b);
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
      assertThrows(IllegalArgumentException.class, () -> MLX.inner(a, b));
    }
  }

  @Test
  void outerComputesShapeAndValues() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      MLXArray b = MLX.array(scope, new float[] {3, 4, 5}, new int[] {3});
      MLXArray result = MLX.outer(a, b);
      assertArrayEquals(new int[] {2, 3}, result.shape());
      assertArrayEquals(new float[] {3, 4, 5, 6, 8, 10}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void broadcastToSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray result = MLX.broadcastTo(a, new int[] {2, 3});
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
      assertThrows(IllegalArgumentException.class, () -> MLX.broadcastTo(a, new int[] {1}));
    }
  }

  @Test
  void broadcastToRejectsNegativeTargetDimension() {
    try (MLXScope scope = new MLXScope()) {
      // Without the non-negative check on targetShape, 1 == 1 would make the
      // directional check itself pass, and this would instead throw MLXException
      // from native -- assert the Java-side exception type specifically.
      MLXArray a = MLX.array(scope, new float[] {1}, new int[] {1});
      assertThrows(IllegalArgumentException.class, () -> MLX.broadcastTo(a, new int[] {-1}));
    }
  }

  @Test
  void squeezeRemovesAllSizeOneAxes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3, 1});
      MLXArray result = MLX.squeeze(a);
      assertArrayEquals(new int[] {3}, result.shape());
      assertArrayEquals(new float[] {1, 2, 3}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void squeezeAxesRemovesOnlyTheGivenAxes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {1, 3, 1});
      MLXArray result = MLX.squeeze(a, new int[] {0});
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
      assertThrows(MLXException.class, () -> MLX.squeeze(a, new int[] {1}));
    }
  }

  @Test
  void sliceRejectsLengthMismatchWithNdim() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      assertThrows(IllegalArgumentException.class, () -> MLX.slice(a, new int[] {0}, new int[] {2, 3}));
    }
  }

  @Test
  void sliceWithNegativeStartSucceeds() {
    try (MLXScope scope = new MLXScope()) {
      // start=-3 on a length-5 axis normalizes to 5-3=2, NumPy-style -- must not throw,
      // and must select elements [2,3,4] (values 3,4,5), not some clamped/rejected result.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5}, new int[] {5});
      MLXArray result = MLX.slice(a, new int[] {-3}, new int[] {5});
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
      MLXArray result = MLX.slice(a, new int[] {3}, new int[] {1}, new int[] {-1});
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
          assertThrows(IllegalArgumentException.class, () -> MLX.slice(a, new int[] {0}, new int[] {4}, new int[] {0}));
      assertEquals("slice: strides[0] must not be 0 (slice step cannot be zero)", ex.getMessage());
    }
  }

  @Test
  void sliceContiguousSelectsExpectedSubrange() {
    try (MLXScope scope = new MLXScope()) {
      // [[1,2,3],[4,5,6]], columns 1..2 of both rows -> [[2,3],[5,6]].
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
      MLXArray result = MLX.slice(a, new int[] {0, 1}, new int[] {2, 3});
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
      MLXArray result = MLX.slice(a, new int[] {0}, new int[] {8}, new int[] {2});
      assertArrayEquals(new int[] {4}, result.shape());
      assertArrayEquals(new float[] {1, 3, 5, 7}, result.toFloatArray(), EPS);
    }
  }
}
