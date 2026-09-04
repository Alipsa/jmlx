package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

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
          MLX.array(scope, new float[] {1f, 3f, 3f, 2f, 5f, 5f, 4f, 0f}, new int[] {2, 4});
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
      assertFloating(topk, new int[] {2, 2}, new float[] {3f, 3f, 5f, 5f});
      assertFloating(sorted, new int[] {2, 4}, new float[] {1f, 2f, 3f, 3f, 0f, 4f, 5f, 5f});
      assertExact(argsorted, DType.UINT32, new int[] {2, 4}, new int[] {0, 3, 1, 2, 3, 2, 0, 1});
      assertFloating(partitioned, new int[] {2, 4}, new float[] {1f, 2f, 3f, 3f, 0f, 4f, 5f, 5f});
      assertExact(
          argpartitioned, DType.UINT32, new int[] {2, 4}, new int[] {0, 3, 1, 2, 3, 2, 0, 1});
      assertExact(key, DType.UINT32, new int[] {2}, new int[] {0, 42});
      assertExact(
          split,
          DType.UINT32,
          new int[] {3, 2},
          new int[] {-1160419002, -561808247, -548466209, 894150801, 801545058, -1931765865});
      assertExact(categorical, DType.UINT32, new int[] {2}, new int[] {1, 0});
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
