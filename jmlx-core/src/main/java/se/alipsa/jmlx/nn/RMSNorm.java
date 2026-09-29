package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Root Mean Square normalization, delegating to {@link MLXFast#rmsNorm(MLXArray, MLXArray, float)}.
 * {@code weight} is required, not nullable, for this layer -- matching how every real {@code
 * RMSNorm} usage trains one. {@link MLXFast#rmsNorm} itself still accepts a {@code null} {@code
 * weight}, unused by this layer, for a caller that wants the bare op.
 */
public final class RMSNorm extends Module implements UnaryModule {

  private final float eps;
  private final boolean weightOffset;

  /**
   * Creates an {@code RMSNorm} layer with the given {@code weight} and numerical-stability {@code
   * eps}.
   */
  public RMSNorm(MLXScope scope, MLXArray weight, float eps) {
    this(scope, weight, eps, false);
  }

  /** Creates an RMS norm whose stored weight is optionally offset by one at evaluation time. */
  public RMSNorm(MLXScope scope, MLXArray weight, float eps, boolean weightOffset) {
    super(scope);
    param("weight", weight);
    this.eps = eps;
    this.weightOffset = weightOffset;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    if (!weightOffset) {
      return MLXFast.rmsNorm(x, param("weight"), eps);
    }
    MLXArray input = MLX.astype(x, DType.FLOAT32);
    MLXArray variance = MLXOps.mean(MLXOps.square(input), new int[] {x.ndim() - 1}, true);
    MLXArray epsilon = MLX.full(x.scope(), new int[] {1}, eps, DType.FLOAT32);
    MLXArray normalized = MLXOps.multiply(input, MLXOps.rsqrt(MLXOps.add(variance, epsilon)));
    MLXArray result = MLXOps.multiply(normalized, offsetWeight(x));
    return MLX.astype(result, x.dtype());
  }

  MLXArray offsetWeight(MLXArray x) {
    // Module.scope() puts a unary cast of a parameter in the model scope; the activation-scope
    // float32 operand keeps the derived weight in the activation scope on every forward. Do not
    // cache this value: ModuleGrad can rebind traced parameters without onParametersUpdated.
    MLXArray one = MLX.full(x.scope(), new int[] {1}, 1f, DType.FLOAT32);
    return MLXOps.add(one, param("weight"));
  }
}
