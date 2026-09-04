package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Records the native selection and explicit-key behavior required to accept the Phase 6 contract.
 *
 * <p>The first run intentionally reports observations rather than freezing unverified values as
 * goldens. Copy the {@code PHASE6_PROBE} lines from an Apple-Silicon run into {@code
 * req/plans/phase6-0b-probe-findings.md}; the subsequent change turns those observations into
 * stable assertions and removes the reporting-only wording.
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

      assertEquals(DType.UINT32, argmax.dtype(), "native argmax dtype before facade conversion");
      System.out.println("PHASE6_PROBE argmax=" + exact(argmax));
      System.out.println("PHASE6_PROBE topk=" + floating(topk));
      System.out.println("PHASE6_PROBE sort=" + floating(sorted));
      System.out.println("PHASE6_PROBE argsort=" + exact(argsorted));
      System.out.println("PHASE6_PROBE partition=" + floating(partitioned));
      System.out.println("PHASE6_PROBE argpartition=" + exact(argpartitioned));
      System.out.println("PHASE6_PROBE key=" + exact(key));
      System.out.println("PHASE6_PROBE split_num=" + exact(split));
      System.out.println("PHASE6_PROBE categorical=" + exact(categorical));
    }
  }

  private static String floating(MLXArray array) {
    return array.dtype()
        + " "
        + Arrays.toString(array.shape())
        + " "
        + Arrays.toString(array.toFloatArray());
  }

  private static String exact(MLXArray array) {
    MLXArray int32 = array.dtype() == DType.INT32 ? array : MLX.astype(array, DType.INT32);
    return array.dtype()
        + " "
        + Arrays.toString(array.shape())
        + " "
        + Arrays.toString(int32.toIntArray());
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
