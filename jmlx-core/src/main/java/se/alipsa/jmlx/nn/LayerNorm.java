package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Layer normalization, delegating to {@link MLXFast#layerNorm(MLXArray, MLXArray, MLXArray, float)}. Both
 * {@code weight} and {@code bias} are required for this layer, same reasoning as {@link RMSNorm}. A caller needing
 * {@code affine=false} calls {@code MLXFast.layerNorm(x, null, null, eps)} directly -- that is what the bare op is for.
 */
public final class LayerNorm extends Module implements UnaryModule {

  private final float eps;

  /**
   * Creates a {@code LayerNorm} layer with the given {@code weight}, {@code bias} and numerical-stability {@code eps}.
   */
  public LayerNorm(MLXScope scope, MLXArray weight, MLXArray bias, float eps) {
    super(scope);
    param("weight", weight);
    param("bias", bias);
    this.eps = eps;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXFast.layerNorm(x, param("weight"), param("bias"), eps);
  }
}
