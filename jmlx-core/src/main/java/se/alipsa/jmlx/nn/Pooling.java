package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

abstract class Pooling extends UnaryLayer {
  private final int[] kernel;
  private final int[] stride;
  private final int[] padding;
  private final boolean maximum;

  Pooling(MLXScope scope, int[] kernel, int[] stride, int[] padding, boolean maximum) {
    super(scope);
    if (kernel.length != stride.length || kernel.length != padding.length) {
      throw new IllegalArgumentException("pool: spatial configuration lengths differ");
    }
    this.kernel = kernel.clone();
    this.stride = stride.clone();
    this.padding = padding.clone();
    this.maximum = maximum;
    for (int i = 0; i < kernel.length; i++) {
      if (kernel[i] <= 0 || stride[i] <= 0 || padding[i] < 0) {
        throw new IllegalArgumentException("pool: kernel/stride positive and padding nonnegative");
      }
    }
  }

  @Override
  public final MLXArray forward(MLXArray x) {
    int rank = kernel.length;
    if (x.ndim() != rank + 2) {
      throw new IllegalArgumentException("pool: incorrect input rank");
    }
    int[] axes = new int[rank];
    for (int i = 0; i < rank; i++) {
      axes[i] = i + 1;
    }
    x =
        MLXShape.pad(
            x,
            axes,
            padding,
            padding,
            LayerOps.scalar(x, maximum ? Float.NEGATIVE_INFINITY : 0),
            MLXShape.PadMode.CONSTANT);
    int[] source = x.shape();
    int[] view = new int[2 * rank + 2];
    long[] strides = new long[view.length];
    long[] contiguous = new long[source.length];
    long step = 1;
    for (int i = source.length - 1; i >= 0; i--) {
      contiguous[i] = step;
      step = Math.multiplyExact(step, source[i]);
    }
    view[0] = source[0];
    strides[0] = contiguous[0];
    view[view.length - 1] = source[source.length - 1];
    strides[strides.length - 1] = 1;
    int[] reductions = new int[rank];
    for (int i = 0; i < rank; i++) {
      int output = Math.toIntExact(Math.floorDiv((long) source[i + 1] - kernel[i], stride[i]) + 1);
      if (output < 0) {
        throw new IllegalArgumentException(
            "pool: padded size "
                + source[i + 1]
                + ", window "
                + kernel[i]
                + ", stride "
                + stride[i]
                + " gives negative extent");
      }
      view[i + 1] = output;
      strides[i + 1] = Math.multiplyExact(contiguous[i + 1], stride[i]);
      view[rank + i + 1] = kernel[i];
      strides[rank + i + 1] = contiguous[i + 1];
      reductions[i] = rank + i + 1;
    }
    MLXArray windows = MLXShape.asStrided(x, view, strides, 0);
    return maximum
        ? MLXOps.maxAxes(windows, reductions, false)
        : MLXOps.mean(windows, reductions, false);
  }
}
