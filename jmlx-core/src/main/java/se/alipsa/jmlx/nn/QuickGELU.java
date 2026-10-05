package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** QuickGELU activation; temporary scalars live in the input scope. */
public final class QuickGELU extends UnaryLayer {
  /** Creates the pinned MLX default activation. */
  public QuickGELU(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.multiply(x, MLXOps.sigmoid(MLXOps.multiply(x, LayerOps.scalar(x, 1.702f))));
  }
}
