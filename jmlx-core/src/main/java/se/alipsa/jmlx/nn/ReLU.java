package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** ReLU activation; temporary scalars live in the input scope. */
public final class ReLU extends UnaryLayer {
  /** Creates the pinned MLX default activation. */
  public ReLU(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.maximum(x, LayerOps.scalar(x, 0));
  }
}
