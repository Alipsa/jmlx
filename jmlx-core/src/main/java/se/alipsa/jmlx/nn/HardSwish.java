package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** HardSwish activation; temporary scalars live in the input scope. */
public final class HardSwish extends UnaryLayer {
  /** Creates the pinned MLX default activation. */
  public HardSwish(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.multiply(
        x,
        MLXOps.divide(
            MLXOps.clip(
                MLXOps.add(x, LayerOps.scalar(x, 3)), LayerOps.scalar(x, 0), LayerOps.scalar(x, 6)),
            LayerOps.scalar(x, 6)));
  }
}
