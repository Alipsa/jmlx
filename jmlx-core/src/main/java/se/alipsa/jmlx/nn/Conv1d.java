package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXConv;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Channels-last checkpoint-backed Conv1d; configuration is copied and weights are read fresh. */
public final class Conv1d extends UnaryLayer {
  private final int[] stride;
  private final int[] padding;
  private final int[] dilation;
  private final int groups;
  private final boolean hasBias;

  /** Unit stride/dilation, zero padding and one group. */
  public Conv1d(MLXScope scope, MLXArray weight, MLXArray bias) {
    this(scope, weight, bias, 1, 0, 1, 1);
  }

  /** Creates a layer with explicit immutable spatial options. */
  public Conv1d(
      MLXScope scope,
      MLXArray weight,
      MLXArray bias,
      int stride,
      int padding,
      int dilation,
      int groups) {
    super(scope);
    if (weight.ndim() != 3 || groups <= 0) {
      throw new IllegalArgumentException("Conv1d: invalid weight rank/groups");
    }
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException("Conv1d: bias must match output channels");
    }
    this.stride = new int[] {stride};
    this.padding = new int[] {padding};
    this.dilation = new int[] {dilation};
    if (this.stride.length != 1 || this.padding.length != 1 || this.dilation.length != 1) {
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
    MLXArray result =
        MLXConv.conv1d(x, param("weight"), stride[0], padding[0], dilation[0], groups);
    return hasBias ? MLXOps.add(result, param("bias")) : result;
  }
}
