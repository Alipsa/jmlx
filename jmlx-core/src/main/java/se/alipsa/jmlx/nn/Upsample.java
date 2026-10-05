package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Channels-last nearest/linear interpolation over any spatial rank. Output sizes truncate scale*N.
 * Nearest enlargement rounds ties to even; reduction samples floor(i*N/M). Linear uses the pinned
 * centered grid, or endpoint alignment. Single-element aligned linear outputs are rejected,
 * matching the pinned denominator failure. Every index is clipped before gather; an empty output is
 * created without gathering an empty source.
 */
public final class Upsample extends UnaryLayer {
  /** Supported interpolation modes. */
  public enum Mode {
    NEAREST,
    LINEAR
  }

  private final double[] scales;
  private final Mode mode;
  private final boolean alignCorners;

  /** One scale shared by every spatial axis, nearest mode. */
  public Upsample(MLXScope scope, double scale) {
    this(scope, new double[] {scale}, Mode.NEAREST, false);
  }

  /** Explicit spatial scales, mode and alignment. A single scale broadcasts across axes. */
  public Upsample(MLXScope scope, double[] scales, Mode mode, boolean alignCorners) {
    super(scope);
    this.scales = scales.clone();
    this.mode = java.util.Objects.requireNonNull(mode);
    this.alignCorners = alignCorners;
    if (scales.length == 0) {
      throw new IllegalArgumentException("Upsample: missing scales");
    }
    for (double scale : scales) {
      if (!(scale > 0) || !Double.isFinite(scale)) {
        throw new IllegalArgumentException("Upsample: invalid scale");
      }
    }
  }

  @Override
  public MLXArray forward(MLXArray x) {
    int rank = x.ndim() - 2;
    if (rank < 1 || (scales.length != 1 && scales.length != rank)) {
      throw new IllegalArgumentException("Upsample: rank/scale mismatch");
    }
    for (int axis = 1; axis <= rank; axis++) {
      int n = x.shape()[axis];
      double scale = scales[scales.length == 1 ? 0 : axis - 1];
      double extent = n * scale;
      if (extent > Integer.MAX_VALUE) {
        throw new IllegalArgumentException("Upsample: size overflow");
      }
      int m = (int) extent;
      if (m == 0 || n == 0) {
        int[] empty = x.shape();
        empty[axis] = 0;
        x = MLX.zeros(x.scope(), empty, x.dtype());
        continue;
      }
      if (mode == Mode.LINEAR && alignCorners && m == 1) {
        throw new IllegalArgumentException(
            "Upsample: aligned linear output size one has undefined denominator");
      }
      int[] left = new int[m];
      int[] right = new int[m];
      float[] weights = new float[m];
      for (int i = 0; i < m; i++) {
        double coordinate;
        if (mode == Mode.NEAREST) {
          coordinate = m > n ? Math.rint((i + 0.5) * n / m - 0.5) : Math.floor((double) i * n / m);
        } else {
          coordinate =
              alignCorners
                  ? ((double) i * (n - 1) / (m - 1))
                  : i / scale - ((m - 1) / scale - n + 1) / 2;
        }
        coordinate = Math.max(0, Math.min(n - 1, coordinate));
        left[i] = (int) Math.floor(coordinate);
        right[i] = (int) Math.ceil(coordinate);
        weights[i] = (float) (coordinate - left[i]);
      }
      MLXArray l = MLXShape.takeAxis(x, MLX.array(x.scope(), left, new int[] {m}), axis);
      if (mode == Mode.NEAREST) {
        x = l;
        continue;
      }
      MLXArray r = MLXShape.takeAxis(x, MLX.array(x.scope(), right, new int[] {m}), axis);
      int[] shape = new int[x.ndim()];
      java.util.Arrays.fill(shape, 1);
      shape[axis] = m;
      MLXArray weight = MLX.array(x.scope(), weights, shape);
      x = MLXOps.add(l, MLXOps.multiply(MLXOps.subtract(r, l), weight));
    }
    return x;
  }
}
