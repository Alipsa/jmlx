package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Group population normalization. MLX defaults to interleaved channels. PyTorch checkpoint loaders
 * must explicitly select pytorchCompatible=true for contiguous channel groups.
 */
public final class GroupNorm extends Normalization {
  private final int groups;
  private final boolean pytorchCompatible;

  /** Interleaved non-affine normalization with epsilon 1e-5. */
  public GroupNorm(MLXScope scope, int groups, int channels) {
    this(scope, groups, channels, 1e-5f, false, null, null);
  }

  /** Explicit group ordering and optional checkpoint affine pair. */
  public GroupNorm(
      MLXScope scope,
      int groups,
      int channels,
      float epsilon,
      boolean pytorchCompatible,
      MLXArray weight,
      MLXArray bias) {
    super(scope, channels, epsilon, weight, bias);
    if (groups <= 0 || channels % groups != 0) {
      throw new IllegalArgumentException("GroupNorm: channels must be divisible by groups");
    }
    this.groups = groups;
    this.pytorchCompatible = pytorchCompatible;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    validate(x, 2, Integer.MAX_VALUE);
    int batch = x.shape()[0];
    int elements = Math.toIntExact(x.size() / (long) Math.max(1, batch) / groups);
    MLXArray grouped;
    int[] axes;
    if (pytorchCompatible) {
      int groupSize = channels / groups;
      int spatial = elements / groupSize;
      grouped =
          MLXShape.transpose(
              MLXShape.reshape(x, new int[] {batch, spatial, groups, groupSize}),
              new int[] {0, 2, 1, 3});
      axes = new int[] {2, 3};
    } else {
      grouped = MLXShape.reshape(x, new int[] {batch, elements, groups});
      axes = new int[] {1};
    }
    MLXArray y =
        normalize(
            grouped, MLXOps.mean(grouped, axes, true), MLXOps.varAxes(grouped, axes, true, 0));
    if (pytorchCompatible) {
      y = MLXShape.transpose(y, new int[] {0, 2, 1, 3});
    }
    return affine(MLXShape.reshape(y, x.shape()));
  }
}
