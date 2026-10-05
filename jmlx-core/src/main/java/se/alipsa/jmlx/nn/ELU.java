package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** ELU activation; temporary scalars live in the input scope. */
public final class ELU extends UnaryLayer {
  private final float slope;

  /** Creates the pinned MLX default activation. */
  public ELU(MLXScope scope) {
    this(scope, 1f);
  }

  /** Creates the activation with explicit negative slope/alpha. */
  public ELU(MLXScope scope, float slope) {
    super(scope);
    this.slope = slope;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.where(
        MLXOps.greater(x, LayerOps.scalar(x, 0)),
        x,
        MLXOps.multiply(
            LayerOps.scalar(x, slope), MLXOps.subtract(MLXOps.exp(x), LayerOps.scalar(x, 1))));
  }
}
