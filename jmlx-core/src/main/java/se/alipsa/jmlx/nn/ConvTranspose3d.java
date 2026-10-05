package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXConv;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Channels-last checkpoint-backed ConvTranspose3d; configuration is copied and weights are read
 * fresh.
 */
public final class ConvTranspose3d extends UnaryLayer {
  private final int[] stride;
  private final int[] padding;
  private final int[] dilation;
  private final int[] outputPadding;
  private final int groups;
  private final boolean hasBias;

  /** Unit stride/dilation, zero padding and one group. */
  public ConvTranspose3d(MLXScope scope, MLXArray weight, MLXArray bias) {
    this(
        scope,
        weight,
        bias,
        new int[] {1, 1, 1},
        new int[] {0, 0, 0},
        new int[] {1, 1, 1},
        new int[] {0, 0, 0},
        1);
  }

  /** Creates a layer with explicit immutable spatial options. */
  public ConvTranspose3d(
      MLXScope scope,
      MLXArray weight,
      MLXArray bias,
      int[] stride,
      int[] padding,
      int[] dilation,
      int[] outputPadding,
      int groups) {
    super(scope);
    if (weight.ndim() != 5 || groups <= 0) {
      throw new IllegalArgumentException("ConvTranspose3d: invalid weight rank/groups");
    }
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException("ConvTranspose3d: bias must match output channels");
    }
    this.stride = stride.clone();
    this.padding = padding.clone();
    this.dilation = dilation.clone();
    this.outputPadding = outputPadding.clone();
    if (this.stride.length != 3
        || this.padding.length != 3
        || this.dilation.length != 3
        || this.outputPadding.length != 3) {
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
        MLXConv.convTranspose3d(
            x, param("weight"), stride, padding, dilation, outputPadding, groups);
    return hasBias ? MLXOps.add(result, param("bias")) : result;
  }
}
