package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Returns position features, without adding them to an input embedding. Odd dims returns dims-1.
 */
public final class SinusoidalPositionalEncoding extends UnaryLayer {
  private final int dims;
  private final float minFrequency;
  private final float maxFrequency;
  private final float scale;
  private final boolean cosFirst;
  private final boolean fullTurns;

  /** Pinned default frequencies, scale sqrt(2/dims), sine first. */
  public SinusoidalPositionalEncoding(MLXScope scope, int dims) {
    this(scope, dims, 0.0001f, 1, (float) Math.sqrt(2.0 / dims), false, false);
  }

  /** Dims below four are rejected: the pinned frequency denominator would be undefined. */
  public SinusoidalPositionalEncoding(
      MLXScope scope,
      int dims,
      float minFrequency,
      float maxFrequency,
      float scale,
      boolean cosFirst,
      boolean fullTurns) {
    super(scope);
    if (dims < 4 || !(minFrequency > 0) || !(maxFrequency > 0)) {
      throw new IllegalArgumentException(
          "SinusoidalPositionalEncoding: dims >= 4 and positive frequencies required");
    }
    this.dims = dims;
    this.minFrequency = minFrequency;
    this.maxFrequency = maxFrequency;
    this.scale = scale == 0 ? (float) Math.sqrt(2.0 / dims) : scale;
    this.cosFirst = cosFirst;
    this.fullTurns = fullTurns;
  }

  @Override
  public MLXArray forward(MLXArray positions) {
    int half = dims / 2;
    float[] frequencies = new float[half];
    for (int i = 0; i < half; i++) {
      double fraction = 1 - (double) i / (half - 1);
      frequencies[i] =
          (float)
              (Math.exp(
                      fraction * (Math.log(maxFrequency) - Math.log(minFrequency))
                          + Math.log(minFrequency))
                  * (fullTurns ? 2 * Math.PI : 1));
    }
    MLXArray angles =
        MLXOps.multiply(
            MLXShape.expandDims(positions, -1),
            MLX.array(positions.scope(), frequencies, new int[] {half}));
    MLXArray sin = MLXOps.sin(angles);
    MLXArray cos = MLXOps.cos(angles);
    return MLXOps.multiply(
        MLXShape.concatenate(cosFirst ? new MLXArray[] {cos, sin} : new MLXArray[] {sin, cos}, -1),
        LayerOps.scalar(angles, scale));
  }
}
