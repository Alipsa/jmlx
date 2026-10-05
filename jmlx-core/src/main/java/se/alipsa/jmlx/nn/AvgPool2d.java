package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.memory.MLXScope;

/** Channels-last AvgPool2d; padded averages include zeros in their denominator. */
public final class AvgPool2d extends Pooling {
  /** Kernel-sized stride and zero padding. */
  public AvgPool2d(MLXScope scope, int[] kernel) {
    this(scope, kernel, kernel, new int[] {0, 0});
  }

  /** Explicit window, stride and symmetric spatial padding. */
  public AvgPool2d(MLXScope scope, int[] kernel, int[] stride, int[] padding) {
    super(scope, kernel, stride, padding, false);
    if (kernel.length != 2) {
      throw new IllegalArgumentException("pool: expected two spatial dimensions");
    }
  }
}
