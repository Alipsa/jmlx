package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.memory.MLXScope;

/** Channels-last AvgPool1d; padded averages include zeros in their denominator. */
public final class AvgPool1d extends Pooling {
  /** Kernel-sized stride and zero padding. */
  public AvgPool1d(MLXScope scope, int kernel) {
    this(scope, kernel, kernel, 0);
  }

  /** Explicit window, stride and symmetric spatial padding. */
  public AvgPool1d(MLXScope scope, int kernel, int stride, int padding) {
    super(scope, new int[] {kernel}, new int[] {stride}, new int[] {padding}, false);
  }
}
