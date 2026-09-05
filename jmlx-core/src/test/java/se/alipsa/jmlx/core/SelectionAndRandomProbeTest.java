package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
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

  private static final int[] SPLIT_TWO_GOLDEN = {-1829035798, -615737125, 255383827, 267815257};

  @Test
  void recordsFirstTiedIndexAwayFromZero() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray tied = MLX.array(scope, new float[] {1f, 3f, 3f, 0f}, new int[] {1, 4});
      MLXArray argmax = rawArgmaxAxis(scope, tied, 1);
      MLX.eval(argmax);

      assertExact(argmax, DType.UINT32, new int[] {1}, new int[] {1});
    }
  }

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
      MLXArray flatTopk = rawTopk(scope, logits, 3);
      MLXArray flatTopkBoundary = rawTopk(scope, logits, 8);
      MLXArray flatSorted = rawSort(scope, logits);
      MLXArray flatArgsorted = rawArgsort(scope, logits);
      MLXArray flatPartitioned = rawPartition(scope, logits, 3);
      MLXArray flatPartitionBoundary = rawPartition(scope, logits, 0);
      MLXArray flatArgpartitioned = rawArgpartition(scope, logits, 3);
      MLXArray flatArgpartitionBoundary = rawArgpartition(scope, logits, 7);
      MLXArray key = rawKey(scope, 42L);
      MLXArray split = rawSplitNum(scope, key, 3);
      MLXArray splitPairArray = rawSplitNum(scope, key, 2);
      MLXArray[] splitPair = rawSplit(scope, key);
      MLXArray categorical = rawCategorical(scope, logits, 1, key);
      MLXArray categoricalRepeat = rawCategorical(scope, logits, 1, key);
      MLX.eval(
          argmax,
          topk,
          sorted,
          argsorted,
          partitioned,
          argpartitioned,
          flatTopk,
          flatTopkBoundary,
          flatSorted,
          flatArgsorted,
          flatPartitioned,
          flatPartitionBoundary,
          flatArgpartitioned,
          flatArgpartitionBoundary,
          key,
          split,
          splitPairArray,
          splitPair[0],
          splitPair[1],
          categorical,
          categoricalRepeat);

      assertExact(argmax, DType.UINT32, new int[] {2}, new int[] {1, 0});
      assertTopK(topk, logits, 2);
      assertFloating(sorted, new int[] {2, 4}, new float[] {0f, 1f, 2f, 3f, 0f, 3f, 4f, 5f});
      assertExact(argsorted, DType.UINT32, new int[] {2, 4}, new int[] {3, 0, 2, 1, 3, 2, 1, 0});
      assertPartition(partitioned, logits, 1);
      assertArgPartition(argpartitioned, logits, 1);
      assertFlatTopK(flatTopk, logits, 3);
      assertFlatTopK(flatTopkBoundary, logits, 8);
      assertFloating(flatSorted, new int[] {8}, new float[] {0f, 0f, 1f, 2f, 3f, 3f, 4f, 5f});
      assertFlatArgsort(flatArgsorted, logits);
      assertFlatPartition(flatPartitioned, logits, 3);
      assertFlatPartition(flatPartitionBoundary, logits, 0);
      assertFlatArgPartition(flatArgpartitioned, logits, 3);
      assertFlatArgPartition(flatArgpartitionBoundary, logits, 7);
      assertExact(key, DType.UINT32, new int[] {2}, new int[] {0, 42});
      assertExact(
          split,
          DType.UINT32,
          new int[] {3, 2},
          new int[] {-1160419002, -561808247, -548466209, 894150801, 801545058, -1931765865});
      int[] splitPairValues = joinedKeys(splitPair);
      assertArrayEquals(SPLIT_TWO_GOLDEN, splitPairValues);
      assertExact(splitPairArray, DType.UINT32, new int[] {2, 2}, SPLIT_TWO_GOLDEN);
      assertFalse(
          Arrays.equals(
              Arrays.copyOfRange(splitPairValues, 0, 2),
              Arrays.copyOfRange(splitPairValues, 2, 4)));
      assertExact(key, DType.UINT32, new int[] {2}, new int[] {0, 42});
      assertExact(categorical, DType.UINT32, new int[] {2}, new int[] {1, 0});
      assertExact(categoricalRepeat, DType.UINT32, new int[] {2}, new int[] {1, 0});
      assertThrows(MLXException.class, () -> rawCategorical(scope, logits, 2, key));
    }
  }

  @Test
  void recordsLazyResultsRetainInputsAfterTheirScopeCloses() {
    try (MLXScope outputScope = new MLXScope()) {
      MLXArray sorted;
      MLXArray[] split;
      MLXArray categorical;
      try (MLXScope inputScope = outputScope.newChild()) {
        MLXArray input = MLX.array(inputScope, new float[] {3f, 1f, 2f, 0f}, new int[] {2, 2});
        MLXArray key = rawKey(inputScope, 42L);
        sorted = rawSort(outputScope, input);
        split = rawSplit(outputScope, key);
        categorical = rawCategorical(outputScope, input, 1, key);
      }

      // Neither result is evaluated until after its input wrappers and their child scope are gone.
      MLX.eval(sorted, split[0], split[1], categorical);
      assertFloating(sorted, new int[] {4}, new float[] {0f, 1f, 2f, 3f});
      int[] splitValues = joinedKeys(split);
      assertArrayEquals(SPLIT_TWO_GOLDEN, splitValues);
      assertFalse(
          Arrays.equals(
              Arrays.copyOfRange(splitValues, 0, 2), Arrays.copyOfRange(splitValues, 2, 4)));
      assertCategoricalIndices(categorical, 2, 2);
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
      float[] rowActual = Arrays.copyOfRange(actual, row * k, row * k + k);
      float[] rowSource = Arrays.copyOfRange(source, row * cols, row * cols + cols);
      Arrays.sort(rowActual);
      Arrays.sort(rowSource);
      float[] expected = Arrays.copyOfRange(rowSource, cols - k, cols);
      assertArrayEquals(expected, rowActual);
    }
  }

  private static void assertFlatTopK(MLXArray topk, MLXArray original, int k) {
    assertEquals(DType.FLOAT32, topk.dtype());
    assertArrayEquals(new int[] {k}, topk.shape());
    float[] actual = topk.toFloatArray();
    float[] source = original.toFloatArray();
    Arrays.sort(actual);
    Arrays.sort(source);
    assertArrayEquals(Arrays.copyOfRange(source, source.length - k, source.length), actual);
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

  private static void assertFlatArgsort(MLXArray indices, MLXArray original) {
    assertEquals(DType.UINT32, indices.dtype());
    assertArrayEquals(new int[] {(int) original.size()}, indices.shape());
    int[] values = MLX.astype(indices, DType.INT32).toIntArray();
    float[] source = original.toFloatArray();
    assertPermutation(values, source.length);
    for (int index = 1; index < values.length; index++) {
      assertTrue(source[values[index - 1]] <= source[values[index]]);
    }
  }

  private static void assertFlatPartition(MLXArray partitioned, MLXArray original, int kth) {
    assertEquals(DType.FLOAT32, partitioned.dtype());
    assertArrayEquals(new int[] {(int) original.size()}, partitioned.shape());
    assertPartitionInvariant(partitioned.toFloatArray(), kth);
  }

  private static void assertFlatArgPartition(MLXArray indices, MLXArray original, int kth) {
    assertEquals(DType.UINT32, indices.dtype());
    assertArrayEquals(new int[] {(int) original.size()}, indices.shape());
    int[] values = MLX.astype(indices, DType.INT32).toIntArray();
    float[] source = original.toFloatArray();
    assertPermutation(values, source.length);
    float[] partitioned = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      partitioned[index] = source[values[index]];
    }
    assertPartitionInvariant(partitioned, kth);
  }

  private static void assertPartitionInvariant(float[] values, int kth) {
    float pivot = values[kth];
    for (int index = 0; index < values.length; index++) {
      if (index < kth) {
        assertTrue(values[index] <= pivot);
      } else if (index > kth) {
        assertTrue(values[index] >= pivot);
      }
    }
  }

  private static void assertPermutation(int[] values, int size) {
    int[] sorted = values.clone();
    Arrays.sort(sorted);
    int[] expected = new int[size];
    for (int index = 0; index < size; index++) {
      expected[index] = index;
    }
    assertArrayEquals(expected, sorted);
  }

  private static int[] joinedKeys(MLXArray[] keys) {
    int[] result = new int[keys.length * 2];
    for (int index = 0; index < keys.length; index++) {
      assertEquals(DType.UINT32, keys[index].dtype());
      assertArrayEquals(new int[] {2}, keys[index].shape());
      int[] values = MLX.astype(keys[index], DType.INT32).toIntArray();
      System.arraycopy(values, 0, result, index * 2, 2);
    }
    return result;
  }

  private static void assertCategoricalIndices(MLXArray values, int count, int categories) {
    assertEquals(DType.UINT32, values.dtype());
    assertArrayEquals(new int[] {count}, values.shape());
    for (int value : MLX.astype(values, DType.INT32).toIntArray()) {
      assertTrue(value >= 0 && value < categories);
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

  private static MLXArray rawTopk(MLXScope scope, MLXArray array, int k) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe flat topk",
        () -> mlx_h.mlx_topk(result, array.handle(), k, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawSortAxis(MLXScope scope, MLXArray array, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe sort",
        () -> mlx_h.mlx_sort_axis(result, array.handle(), axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawSort(MLXScope scope, MLXArray array) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe flat sort", () -> mlx_h.mlx_sort(result, array.handle(), NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawArgsortAxis(MLXScope scope, MLXArray array, int axis) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe argsort",
        () -> mlx_h.mlx_argsort_axis(result, array.handle(), axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, result);
  }

  private static MLXArray rawArgsort(MLXScope scope, MLXArray array) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe flat argsort",
        () -> mlx_h.mlx_argsort(result, array.handle(), NativeOps.DEFAULT_STREAM));
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

  private static MLXArray rawPartition(MLXScope scope, MLXArray array, int kth) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe flat partition",
        () -> mlx_h.mlx_partition(result, array.handle(), kth, NativeOps.DEFAULT_STREAM));
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

  private static MLXArray rawArgpartition(MLXScope scope, MLXArray array, int kth) {
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe flat argpartition",
        () -> mlx_h.mlx_argpartition(result, array.handle(), kth, NativeOps.DEFAULT_STREAM));
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

  private static MLXArray[] rawSplit(MLXScope scope, MLXArray key) {
    MemorySegment first = mlx_h.mlx_array_new(scope);
    MemorySegment second = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "probe split",
        () -> mlx_h.mlx_random_split(first, second, key.handle(), NativeOps.DEFAULT_STREAM));
    return new MLXArray[] {new MLXArray(scope, first), new MLXArray(scope, second)};
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
