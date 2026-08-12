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
}
