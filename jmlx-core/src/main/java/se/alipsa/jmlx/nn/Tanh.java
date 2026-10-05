package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Tanh activation; temporary scalars live in the input scope. */
public final class Tanh extends UnaryLayer {
  /** Creates the pinned MLX default activation. */
  public Tanh(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.tanh(x);
  }
}
