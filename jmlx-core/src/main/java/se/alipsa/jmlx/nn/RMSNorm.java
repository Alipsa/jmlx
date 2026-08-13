package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Root Mean Square normalization, delegating to {@link MLXFast#rmsNorm(MLXArray, MLXArray, float)}.
 * {@code weight} is required, not nullable, for this layer -- matching how every real {@code
 * RMSNorm} usage trains one. {@link MLXFast#rmsNorm} itself still accepts a {@code null} {@code
 * weight}, unused by this layer, for a caller that wants the bare op.
 */
public final class RMSNorm extends Module implements UnaryModule {

  private final float eps;

  /**
   * Creates an {@code RMSNorm} layer with the given {@code weight} and numerical-stability {@code
   * eps}.
   */
  public RMSNorm(MLXScope scope, MLXArray weight, float eps) {
    super(scope);
    param("weight", weight);
    this.eps = eps;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXFast.rmsNorm(x, param("weight"), eps);
  }
}
