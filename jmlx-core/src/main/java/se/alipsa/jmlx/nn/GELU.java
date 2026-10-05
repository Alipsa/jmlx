package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * The GELU activation: exact form {@code 0.5 * x * (1 + erf(x / sqrt(2)))} by default, or the tanh
 * approximation {@code 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))} when constructed
 * with {@code tanhApproximation}. Constants take {@code x}'s dtype so half-precision inputs stay
 * half precision. No parameters -- {@code scope} is accepted only to satisfy {@link Module}'s
 * constructor contract; nothing is registered.
 */
public final class GELU extends UnaryLayer {

  private static final float SQRT2 = 1.4142135f;
  private static final float SQRT_2_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);
  private static final float CUBIC_COEFFICIENT = 0.044715f;

  private final boolean tanhApproximation;

  /** Creates an exact-form {@code GELU} activation layer. */
  public GELU(MLXScope scope) {
    this(scope, false);
  }

  /** Creates a {@code GELU} layer using the tanh approximation when {@code tanhApproximation}. */
  public GELU(MLXScope scope, boolean tanhApproximation) {
    super(scope);
    this.tanhApproximation = tanhApproximation;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    // Scalar constants below go into x's scope, NOT this.scope() -- §2's fifth sub-hazard:
    // a creation op called with the model scope inside forward() leaks once per call.
    MLXScope s = x.scope();
    MLXArray half = MLX.full(s, new int[0], 0.5f, x.dtype());
    MLXArray one = MLX.full(s, new int[0], 1f, x.dtype());
    MLXArray inner;
    if (tanhApproximation) {
      MLXArray cubic = MLX.full(s, new int[0], CUBIC_COEFFICIENT, x.dtype());
      MLXArray scale = MLX.full(s, new int[0], SQRT_2_OVER_PI, x.dtype());
      MLXArray cube = MLXOps.multiply(MLXOps.multiply(x, x), x);
      inner = MLXOps.tanh(MLXOps.multiply(scale, MLXOps.add(x, MLXOps.multiply(cubic, cube))));
    } else {
      MLXArray sqrt2 = MLX.full(s, new int[0], SQRT2, x.dtype());
      inner = MLXOps.erf(MLXOps.divide(x, sqrt2));
    }
    return MLXOps.multiply(MLXOps.multiply(half, x), MLXOps.add(one, inner));
  }
}
