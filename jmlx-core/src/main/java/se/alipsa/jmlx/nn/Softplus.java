package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** Softplus activation; temporary scalars live in the input scope. */
public final class Softplus extends UnaryLayer {
  /** Creates the pinned MLX default activation. */
  public Softplus(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return LayerOps.softplus(x);
  }
}
