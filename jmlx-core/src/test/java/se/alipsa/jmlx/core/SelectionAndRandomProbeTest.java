package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Records the native selection and explicit-key behavior required to accept the Phase 6 contract.
 *
 * <p>The assertions record observations from the MLX pins in {@code scripts/bootstrap-native.sh}.
 */
@EnabledIfNativeAvailable
class SelectionAndRandomProbeTest {

  @Test
  void recordsPinnedSelectionAndExplicitKeyBehavior() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray logits =
          MLX.array(scope, new float[] {1f, 3f, 2f, 0f, 5f, 4f, 3f, 0f}, new int[] {2, 4});
      MLXArray argmax = rawArgmaxAxis(scope, logits, 1);
      MLXArray topk = rawTopkAxis(scope, logits, 2, 1);
      MLXArray sorted = rawSortAxis(scope, logits, 1);
      MLXArray argsorted = rawArgsortAxis(scope, logits, 1);
      MLXArray partitioned = rawPartitionAxis(scope, logits, 1, 1);
      MLXArray argpartitioned = rawArgpartitionAxis(scope, logits, 1, 1);
      MLXArray key = rawKey(scope, 42L);
      MLXArray split = rawSplitNum(scope, key, 3);
      MLXArray categorical = rawCategorical(scope, logits, 1, key);
      MLX.eval(
          argmax, topk, sorted, argsorted, partitioned, argpartitioned, key, split, categorical);

      assertExact(argmax, DType.UINT32, new int[] {2}, new int[] {1, 0});
      assertTopK(topk, logits, 2);
      assertFloating(sorted, new int[] {2, 4}, new float[] {0f, 1f, 2f, 3f, 0f, 3f, 4f, 5f});
      assertExact(argsorted, DType.UINT32, new int[] {2, 4}, new int[] {3, 0, 2, 1, 3, 2, 1, 0});
      assertPartition(partitioned, logits, 1);
      assertArgPartition(argpartitioned, logits, 1);
      assertExact(key, DType.UINT32, new int[] {2}, new int[] {0, 42});
      assertExact(
          split,
          DType.UINT32,
          new int[] {3, 2},
          new int[] {-1160419002, -561808247, -548466209, 894150801, 801545058, -1931765865});
      assertExact(categorical, DType.UINT32, new int[] {2}, new int[] {1, 0});
    }
  }

  @Test
  void recordsArgmaxTieBreaking() {
    try (MLXScope scope = new MLXScope()) {
      // Each row has a repeated maximum (row 0: value 3 at indices 0 and 2; row 1: value 5 at
      // indices 0 and 1), so this specifically exercises tie-breaking, unlike the distinct-valued
      // logits used above.
      MLXArray tiedLogits =
          MLX.array(scope, new float[] {3f, 1f, 3f, 0f, 5f, 5f, 4f, 0f}, new int[] {2, 4});
      MLXArray argmax = rawArgmaxAxis(scope, tiedLogits, 1);
      MLX.eval(argmax);

      assertExact(argmax, DType.UINT32, new int[] {2}, new int[] {0, 0});
    }
  }

  private static void assertTopK(MLXArray topk, MLXArray original, int k) {
    assertEquals(DType.FLOAT32, topk.dtype());
    int rows = original.shape()[0];
    assertArrayEquals(new int[] {rows, k}, topk.shape());
    float[] actual = topk.toFloatArray();
    float[] source = original.toFloatArray();
    int cols = original.shape()[1];
    for (int row = 0; row < rows; row++) {
      float[] rowActual = java.util.Arrays.copyOfRange(actual, row * k, row * k + k);
      float[] rowSource = java.util.Arrays.copyOfRange(source, row * cols, row * cols + cols);
      java.util.Arrays.sort(rowActual);
      java.util.Arrays.sort(rowSource);
      float[] expected = java.util.Arrays.copyOfRange(rowSource, cols - k, cols);
      assertArrayEquals(expected, rowActual);
    }
  }

  private static void assertFloating(MLXArray actual, int[] shape, float[] values) {
    assertEquals(DType.FLOAT32, actual.dtype());
    assertArrayEquals(shape, actual.shape());
    assertArrayEquals(values, actual.toFloatArray());
  }

  private static void assertExact(MLXArray actual, DType dtype, int[] shape, int[] values) {
    assertEquals(dtype, actual.dtype());
    assertArrayEquals(shape, actual.shape());
    MLXArray int32 = actual.dtype() == DType.INT32 ? actual : MLX.astype(actual, DType.INT32);
    assertArrayEquals(values, int32.toIntArray());
  }

  private static void assertPartition(MLXArray partitioned, MLXArray original, int kth) {
    assertEquals(DType.FLOAT32, partitioned.dtype());
    assertArrayEquals(original.shape(), partitioned.shape());
    float[] values = partitioned.toFloatArray();
    for (int row = 0; row < original.shape()[0]; row++) {
      int start = row * original.shape()[1];
      float pivot = values[start + kth];
      for (int index = 0; index < original.shape()[1]; index++) {
        if (index < kth) {
          assertTrue(values[start + index] <= pivot);
        } else if (index > kth) {
          assertTrue(values[start + index] >= pivot);
        }
      }
    }
  }

  private static void assertArgPartition(MLXArray indices, MLXArray original, int kth) {
    assertEquals(DType.UINT32, indices.dtype());
    assertArrayEquals(original.shape(), indices.shape());
    MLXArray int32 = MLX.astype(indices, DType.INT32);
    int[] values = int32.toIntArray();
    float[] source = original.toFloatArray();
    for (int row = 0; row < original.shape()[0]; row++) {
      int start = row * original.shape()[1];
      float pivot = source[start + values[start + kth]];
      for (int index = 0; index < original.shape()[1]; index++) {
        float value = source[start + values[start + index]];
        if (index < kth) {
          assertTrue(value <= pivot);
        } else if (index > kth) {
          assertTrue(value >= pivot);
        }
      }
    }
  }

  private static MLXArray rawArgmaxAxis(MLXScope scope, MLXArray array, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe argmax",
        () -> mlx_h.mlx_argmax_axis(result, array.handle(), axis, false, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawTopkAxis(MLXScope scope, MLXArray array, int k, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe topk",
        () -> mlx_h.mlx_topk_axis(result, array.handle(), k, axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawSortAxis(MLXScope scope, MLXArray array, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe sort",
        () -> mlx_h.mlx_sort_axis(result, array.handle(), axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawArgsortAxis(MLXScope scope, MLXArray array, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe argsort",
        () -> mlx_h.mlx_argsort_axis(result, array.handle(), axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawPartitionAxis(MLXScope scope, MLXArray array, int kth, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe partition",
        () ->
            mlx_h.mlx_partition_axis(result, array.handle(), kth, axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawArgpartitionAxis(MLXScope scope, MLXArray array, int kth, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe argpartition",
        () ->
            mlx_h.mlx_argpartition_axis(
                result, array.handle(), kth, axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawKey(MLXScope scope, long seed) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked("probe key", () -> mlx_h.mlx_random_key(result, seed));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawSplitNum(MLXScope scope, MLXArray key, int count) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe split_num",
        () -> mlx_h.mlx_random_split_num(result, key.handle(), count, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawCategorical(MLXScope scope, MLXArray logits, int axis, MLXArray key) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe categorical",
        () ->
            mlx_h.mlx_random_categorical(
                result, logits.handle(), axis, key.handle(), NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }
}
