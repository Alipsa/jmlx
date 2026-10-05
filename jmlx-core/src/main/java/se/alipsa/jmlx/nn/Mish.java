package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Mish activation; temporary scalars live in the input scope. */
public final class Mish extends UnaryLayer {
  /** Creates the pinned MLX default activation. */
  public Mish(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.multiply(x, MLXOps.tanh(LayerOps.softplus(x)));
  }
}
