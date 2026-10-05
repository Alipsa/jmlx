package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.memory.MLXScope;

/** Channels-last MaxPool1d; padded averages include zeros in their denominator. */
public final class MaxPool1d extends Pooling {
  /** Kernel-sized stride and zero padding. */
  public MaxPool1d(MLXScope scope, int kernel) {
    this(scope, kernel, kernel, 0);
  }

  /** Explicit window, stride and symmetric spatial padding. */
  public MaxPool1d(MLXScope scope, int kernel, int stride, int padding) {
    super(scope, new int[] {kernel}, new int[] {stride}, new int[] {padding}, true);
  }
}
