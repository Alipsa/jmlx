package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Unary selected-axis padding with copied widths and an input-dtype scalar value. */
public final class Pad extends UnaryLayer {
  private final int[] axes;
  private final int[] low;
  private final int[] high;
  private final float value;
  private final MLXShape.PadMode mode;

  /** Creates selected-axis padding. */
  public Pad(
      MLXScope scope, int[] axes, int[] low, int[] high, float value, MLXShape.PadMode mode) {
    super(scope);
    this.axes = axes.clone();
    this.low = low.clone();
    this.high = high.clone();
    this.value = value;
    this.mode = java.util.Objects.requireNonNull(mode);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXShape.pad(x, axes, low, high, LayerOps.scalar(x, value), mode);
  }
}
