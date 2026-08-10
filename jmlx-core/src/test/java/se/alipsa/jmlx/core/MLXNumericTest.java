package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * req/initial-plan.md, Testing approach, "Numeric correctness": every facade
 * op against hand-computed values. reshape/transpose additionally assert
 * element values, not just shape -- the only thing that catches the
 * contiguity bug in {@link MLXArray#toFloatArray()}.
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
