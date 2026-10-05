package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** LeakyReLU activation; temporary scalars live in the input scope. */
public final class LeakyReLU extends UnaryLayer {
  private final float slope;

  /** Creates the pinned MLX default activation. */
  public LeakyReLU(MLXScope scope) {
    this(scope, 0.01f);
  }

  /** Creates the activation with explicit negative slope/alpha. */
  public LeakyReLU(MLXScope scope, float slope) {
    super(scope);
    this.slope = slope;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.maximum(x, MLXOps.multiply(x, LayerOps.scalar(x, slope)));
  }
}
