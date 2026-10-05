package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

abstract class Normalization extends UnaryLayer {
  final int channels;
  final float epsilon;
  private final boolean affine;

  Normalization(MLXScope scope, int channels, float epsilon, MLXArray weight, MLXArray bias) {
    super(scope);
    if (channels <= 0
        || !(epsilon >= 0)
        || !Float.isFinite(epsilon)
        || (weight == null) != (bias == null)) {
      throw new IllegalArgumentException("normalization: invalid channels, epsilon or affine pair");
    }
    this.channels = channels;
    this.epsilon = epsilon;
    affine = weight != null;
    if (affine) {
      vector("weight", weight);
      vector("bias", bias);
    }
  }

  final void vector(String name, MLXArray value) {
    if (value.ndim() != 1 || value.shape()[0] != channels) {
      throw new IllegalArgumentException(name + " must be a channel vector of size " + channels);
    }
    param(name, value);
  }

  final void validate(MLXArray x, int minimumRank, int maximumRank) {
    if (x.ndim() < minimumRank || x.ndim() > maximumRank || x.shape()[x.ndim() - 1] != channels) {
      throw new IllegalArgumentException("normalization: invalid input rank/channel count");
    }
  }

  final MLXArray normalize(MLXArray x, MLXArray mean, MLXArray variance) {
    return MLXOps.multiply(
        MLXOps.subtract(x, mean), MLXOps.rsqrt(MLXOps.add(variance, LayerOps.scalar(x, epsilon))));
  }

  final MLXArray affine(MLXArray x) {
    return affine ? MLXOps.add(MLXOps.multiply(x, param("weight")), param("bias")) : x;
  }
}
