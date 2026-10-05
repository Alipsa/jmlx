package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Pinned MLX symmetric -abs(query-key) bias added to rank-four attention scores. */
public final class ALiBi extends Module {
  /** Creates a parameter-free helper. */
  public ALiBi(MLXScope scope) {
    super(scope);
  }

  /** Adds bias with zero query offset. */
  public MLXArray forward(MLXArray scores) {
    return forward(scores, 0);
  }

  /** Adds bias; query positions start at offset and keys at zero. Allocates in the score scope. */
  public MLXArray forward(MLXArray scores, int offset) {
    if (scores.ndim() != 4 || offset < 0) {
      throw new IllegalArgumentException("ALiBi: rank four and nonnegative offset required");
    }
    int[] shape = scores.shape();
    int heads = shape[1];
    int q = shape[2];
    int k = shape[3];
    if (heads < 1) {
      throw new IllegalArgumentException("ALiBi: at least one head required");
    }
    float[] values = new float[Math.multiplyExact(heads, Math.multiplyExact(q, k))];
    int power = Integer.highestOneBit(heads);
    for (int h = 0; h < heads; h++) {
      int n = h < power ? power : 2 * power;
      int index = h < power ? h : 2 * (h - power);
      double start = Math.pow(2, -Math.pow(2, -(Math.log(n) / Math.log(2) - 3)));
      float slope = (float) Math.pow(start, index + 1);
      for (int i = 0; i < q; i++) {
        for (int j = 0; j < k; j++) {
          values[(h * q + i) * k + j] = -Math.abs((long) offset + i - j) * slope;
        }
      }
    }
    return MLXOps.add(scores, MLX.array(scores.scope(), values, new int[] {1, heads, q, k}));
  }
}
