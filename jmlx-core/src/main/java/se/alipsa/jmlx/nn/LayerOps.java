package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;

final class LayerOps {
  private LayerOps() {}

  static MLXArray scalar(MLXArray x, float value) {
    return MLX.full(x.scope(), new int[0], value, x.dtype());
  }

  static MLXArray softplus(MLXArray x) {
    return MLXOps.add(
        MLXOps.maximum(x, scalar(x, 0)), MLXOps.log1p(MLXOps.exp(MLXOps.negative(MLXOps.abs(x)))));
  }
}
