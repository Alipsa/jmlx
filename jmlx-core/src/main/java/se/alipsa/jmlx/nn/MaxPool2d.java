package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.memory.MLXScope;

/** Channels-last MaxPool2d; padding uses negative infinity. */
public final class MaxPool2d extends Pooling {
  /** Kernel-sized stride and zero padding. */
  public MaxPool2d(MLXScope scope, int[] kernel) {
    this(scope, kernel, kernel, new int[] {0, 0});
  }

  /** Explicit window, stride and symmetric spatial padding. */
  public MaxPool2d(MLXScope scope, int[] kernel, int[] stride, int[] padding) {
    super(scope, kernel, stride, padding, true);
    if (kernel.length != 2) {
      throw new IllegalArgumentException("pool: expected two spatial dimensions");
    }
  }
}
