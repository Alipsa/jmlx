package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * The exact-form GELU activation: {@code 0.5 * x * (1 + erf(x / sqrt(2)))}. The tanh approximation is out of scope for
 * this layer. No parameters -- {@code scope} is accepted only to satisfy {@link Module}'s constructor contract; nothing
 * is registered.
 */
public final class GELU extends Module implements UnaryModule {

  private static final float SQRT2 = 1.4142135f;

  /** Creates a {@code GELU} activation layer. */
  public GELU(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    // Scalar constants below go into x's scope, NOT this.scope() -- §2's fifth sub-hazard:
    // a creation op called with the model scope inside forward() leaks once per call.
    MLXScope s = x.scope();
    MLXArray sqrt2 = MLX.full(s, new int[0], SQRT2, DType.FLOAT32);
    MLXArray half = MLX.full(s, new int[0], 0.5f, DType.FLOAT32);
    MLXArray one = MLX.full(s, new int[0], 1f, DType.FLOAT32);
    MLXArray erfTerm = MLXOps.erf(MLXOps.divide(x, sqrt2));
    return MLXOps.multiply(MLXOps.multiply(half, x), MLXOps.add(one, erfTerm));
  }
}
