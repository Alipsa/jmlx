package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXConv;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Channels-last checkpoint-backed Conv2d; configuration is copied and weights are read fresh. */
public final class Conv2d extends UnaryLayer {
  private final int[] stride;
  private final int[] padding;
  private final int[] dilation;
  private final int groups;
  private final boolean hasBias;

  /** Unit stride/dilation, zero padding and one group. */
  public Conv2d(MLXScope scope, MLXArray weight, MLXArray bias) {
    this(scope, weight, bias, new int[] {1, 1}, new int[] {0, 0}, new int[] {1, 1}, 1);
  }

  /** Creates a layer with explicit immutable spatial options. */
  public Conv2d(
      MLXScope scope,
      MLXArray weight,
      MLXArray bias,
      int[] stride,
      int[] padding,
      int[] dilation,
      int groups) {
    super(scope);
    if (weight.ndim() != 4 || groups <= 0) {
      throw new IllegalArgumentException("Conv2d: invalid weight rank/groups");
    }
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException("Conv2d: bias must match output channels");
    }
    this.stride = stride.clone();
    this.padding = padding.clone();
    this.dilation = dilation.clone();
    if (this.stride.length != 2 || this.padding.length != 2 || this.dilation.length != 2) {
      throw new IllegalArgumentException("convolution: spatial configuration length mismatch");
    }
    for (int i = 0; i < this.stride.length; i++) {
      if (this.stride[i] <= 0 || this.dilation[i] <= 0) {
        throw new IllegalArgumentException("convolution: stride and dilation must be positive");
      }
    }
    this.groups = groups;
    param("weight", weight);
    hasBias = bias != null;
    if (hasBias) {
      param("bias", bias);
    }
  }

  @Override
  public MLXArray forward(MLXArray x) {
    MLXArray result = MLXConv.conv2d(x, param("weight"), stride, padding, dilation, groups);
    return hasBias ? MLXOps.add(result, param("bias")) : result;
  }
}
