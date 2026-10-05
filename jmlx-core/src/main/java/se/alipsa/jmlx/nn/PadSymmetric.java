package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Equal-width padding on every axis, not reflection padding. */
public final class PadSymmetric extends UnaryLayer {
  private final int width;
  private final float value;
  private final MLXShape.PadMode mode;

  /** Creates equal padding widths. */
  public PadSymmetric(MLXScope scope, int width, float value, MLXShape.PadMode mode) {
    super(scope);
    this.width = width;
    this.value = value;
    this.mode = java.util.Objects.requireNonNull(mode);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXShape.padSymmetric(x, width, LayerOps.scalar(x, value), mode);
  }
}
