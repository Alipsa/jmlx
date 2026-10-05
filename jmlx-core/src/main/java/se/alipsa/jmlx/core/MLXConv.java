package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/** Channels-last convolutions, with weights [out, spatial kernel..., in/groups]. */
public final class MLXConv {
  private MLXConv() {}

  private static void validate(
      MLXArray x,
      MLXArray w,
      int rank,
      int[] stride,
      int[] padding,
      int[] dilation,
      int[] outputPadding,
      int groups) {
    if (x.ndim() != rank + 2
        || w.ndim() != rank + 2
        || stride.length != rank
        || padding.length != rank
        || dilation.length != rank
        || groups <= 0
        || (outputPadding != null && outputPadding.length != rank)) {
      throw new IllegalArgumentException("convolution: invalid ranks, configuration or groups");
    }
    int[] xs = x.shape();
    int[] ws = w.shape();
    if (xs[rank + 1] % groups != 0
        || ws[0] % groups != 0
        || ws[rank + 1] != xs[rank + 1] / groups) {
      throw new IllegalArgumentException("convolution: incompatible channels/groups");
    }
    for (int i = 0; i < rank; i++) {
      if (stride[i] <= 0
          || dilation[i] <= 0
          || ws[i + 1] <= 0
          || (outputPadding != null && outputPadding[i] < 0)) {
        throw new IllegalArgumentException("convolution: invalid stride, dilation or kernel");
      }
      try {
        long kernel = Math.addExact(Math.multiplyExact(ws[i + 1] - 1L, dilation[i]), 1);
        long extent =
            outputPadding == null
                ? Math.floorDiv(xs[i + 1] + 2L * padding[i] - kernel, stride[i]) + 1
                : (xs[i + 1] - 1L) * stride[i] - 2L * padding[i] + kernel + outputPadding[i];
        if (extent < 0 || extent > Integer.MAX_VALUE) {
          throw new IllegalArgumentException("convolution: output extent out of range");
        }
      } catch (ArithmeticException error) {
        throw new IllegalArgumentException("convolution: size overflow", error);
      }
    }
  }

  /** Native conv1d, with every spatial option exposed. */
  public static MLXArray conv1d(
      MLXArray x, MLXArray weight, int stride, int padding, int dilation, int groups) {
    validate(
        x, weight, 1, new int[] {stride}, new int[] {padding}, new int[] {dilation}, null, groups);
    MLXScope scope = NativeOps.scopeOf("conv1d", x, weight);
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "conv1d",
        () ->
            mlx_h.mlx_conv1d(
                result,
                x.handle(),
                weight.handle(),
                stride,
                padding,
                dilation,
                groups,
                scope.stream()));
    return new MLXArray(scope, result);
  }

  /** Native conv2d, with every spatial option exposed. */
  public static MLXArray conv2d(
      MLXArray x, MLXArray weight, int[] stride, int[] padding, int[] dilation, int groups) {
    validate(x, weight, 2, stride, padding, dilation, null, groups);
    MLXScope scope = NativeOps.scopeOf("conv2d", x, weight);
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "conv2d",
        () ->
            mlx_h.mlx_conv2d(
                result,
                x.handle(),
                weight.handle(),
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                groups,
                scope.stream()));
    return new MLXArray(scope, result);
  }

  /** Native conv3d, with every spatial option exposed. */
  public static MLXArray conv3d(
      MLXArray x, MLXArray weight, int[] stride, int[] padding, int[] dilation, int groups) {
    validate(x, weight, 3, stride, padding, dilation, null, groups);
    MLXScope scope = NativeOps.scopeOf("conv3d", x, weight);
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "conv3d",
        () ->
            mlx_h.mlx_conv3d(
                result,
                x.handle(),
                weight.handle(),
                stride[0],
                stride[1],
                stride[2],
                padding[0],
                padding[1],
                padding[2],
                dilation[0],
                dilation[1],
                dilation[2],
                groups,
                scope.stream()));
    return new MLXArray(scope, result);
  }

  /** Native convTranspose1d, with every spatial option exposed. */
  public static MLXArray convTranspose1d(
      MLXArray x,
      MLXArray weight,
      int stride,
      int padding,
      int dilation,
      int outputPadding,
      int groups) {
    validate(
        x,
        weight,
        1,
        new int[] {stride},
        new int[] {padding},
        new int[] {dilation},
        new int[] {outputPadding},
        groups);
    MLXScope scope = NativeOps.scopeOf("convTranspose1d", x, weight);
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "convTranspose1d",
        () ->
            mlx_h.mlx_conv_transpose1d(
                result,
                x.handle(),
                weight.handle(),
                stride,
                padding,
                dilation,
                outputPadding,
                groups,
                scope.stream()));
    return new MLXArray(scope, result);
  }

  /** Native convTranspose2d, with every spatial option exposed. */
  public static MLXArray convTranspose2d(
      MLXArray x,
      MLXArray weight,
      int[] stride,
      int[] padding,
      int[] dilation,
      int[] outputPadding,
      int groups) {
    validate(x, weight, 2, stride, padding, dilation, outputPadding, groups);
    if (groups > 1 && (stride[0] != 1 || stride[1] != 1)) {
      throw new UnsupportedOperationException(
          "convTranspose2d: pinned MLX GPU grouped stride > 1 is incorrect; see Phase 7.1"
              + " findings");
    }

    MLXScope scope = NativeOps.scopeOf("convTranspose2d", x, weight);
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "convTranspose2d",
        () ->
            mlx_h.mlx_conv_transpose2d(
                result,
                x.handle(),
                weight.handle(),
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                outputPadding[0],
                outputPadding[1],
                groups,
                scope.stream()));
    return new MLXArray(scope, result);
  }

  /** Native convTranspose3d, with every spatial option exposed. */
  public static MLXArray convTranspose3d(
      MLXArray x,
      MLXArray weight,
      int[] stride,
      int[] padding,
      int[] dilation,
      int[] outputPadding,
      int groups) {
    validate(x, weight, 3, stride, padding, dilation, outputPadding, groups);
    MLXScope scope = NativeOps.scopeOf("convTranspose3d", x, weight);
    MemorySegment result = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "convTranspose3d",
        () ->
            mlx_h.mlx_conv_transpose3d(
                result,
                x.handle(),
                weight.handle(),
                stride[0],
                stride[1],
                stride[2],
                padding[0],
                padding[1],
                padding[2],
                dilation[0],
                dilation[1],
                dilation[2],
                outputPadding[0],
                outputPadding[1],
                outputPadding[2],
                groups,
                scope.stream()));
    return new MLXArray(scope, result);
  }

  /** General convolution exposes low/high padding, kernel/input dilation and kernel reversal. */
  public static MLXArray convGeneral(
      MLXArray x,
      MLXArray weight,
      int[] stride,
      int[] low,
      int[] high,
      int[] kernelDilation,
      int[] inputDilation,
      int groups,
      boolean flip) {
    int rank = x.ndim() - 2;
    if (rank < 1
        || rank > 3
        || weight.ndim() != x.ndim()
        || stride.length != rank
        || low.length != rank
        || high.length != rank
        || kernelDilation.length != rank
        || inputDilation.length != rank
        || groups <= 0) {
      throw new IllegalArgumentException("convGeneral: invalid spatial configuration");
    }
    int[] inputShape = x.shape();
    int[] weightShape = weight.shape();
    if (inputShape[rank + 1] % groups != 0
        || weightShape[0] % groups != 0
        || weightShape[rank + 1] != inputShape[rank + 1] / groups) {
      throw new IllegalArgumentException("convGeneral: incompatible channels/groups");
    }
    for (int i = 0; i < rank; i++) {
      if (stride[i] <= 0 || kernelDilation[i] <= 0 || inputDilation[i] <= 0) {
        throw new IllegalArgumentException("convGeneral: dilation and stride must be positive");
      }
    }
    try {
      for (int i = 0; i < rank; i++) {
        if (weightShape[i + 1] <= 0) {
          throw new IllegalArgumentException("convGeneral: kernel must be nonempty");
        }
        long inputExtent =
            Math.addExact(Math.multiplyExact(inputShape[i + 1] - 1L, inputDilation[i]), 1);
        long kernelExtent =
            Math.addExact(Math.multiplyExact(weightShape[i + 1] - 1L, kernelDilation[i]), 1);
        long padded = Math.addExact(Math.addExact(inputExtent, low[i]), high[i]);
        long outputExtent =
            Math.addExact(Math.floorDiv(Math.subtractExact(padded, kernelExtent), stride[i]), 1);
        if (outputExtent < 0 || outputExtent > Integer.MAX_VALUE) {
          throw new IllegalArgumentException("convGeneral: output extent out of range");
        }
      }
    } catch (ArithmeticException error) {
      throw new IllegalArgumentException("convGeneral: size overflow", error);
    }
    MLXScope scope = NativeOps.scopeOf("convGeneral", x, weight);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment ns = tmp.allocateFrom(ValueLayout.JAVA_INT, stride);
      MemorySegment nl = tmp.allocateFrom(ValueLayout.JAVA_INT, low);
      MemorySegment nh = tmp.allocateFrom(ValueLayout.JAVA_INT, high);
      MemorySegment nk = tmp.allocateFrom(ValueLayout.JAVA_INT, kernelDilation);
      MemorySegment ni = tmp.allocateFrom(ValueLayout.JAVA_INT, inputDilation);
      MemorySegment result = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "convGeneral",
          () ->
              mlx_h.mlx_conv_general(
                  result,
                  x.handle(),
                  weight.handle(),
                  ns,
                  rank,
                  nl,
                  rank,
                  nh,
                  rank,
                  nk,
                  rank,
                  ni,
                  rank,
                  groups,
                  flip,
                  scope.stream()));
      return new MLXArray(scope, result);
    }
  }
}
