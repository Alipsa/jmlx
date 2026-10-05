package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** SELU uses pinned MLX constants 1.0507 and 1.67326; temporary scalars live in the input scope. */
public final class SELU extends UnaryLayer {
  /** Creates the pinned MLX default activation. */
  public SELU(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.multiply(
        LayerOps.scalar(x, 1.0507f),
        MLXOps.where(
            MLXOps.greater(x, LayerOps.scalar(x, 0)),
            x,
            MLXOps.multiply(
                LayerOps.scalar(x, 1.67326f),
                MLXOps.subtract(MLXOps.exp(x), LayerOps.scalar(x, 1)))));
  }
}
