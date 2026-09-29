package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Builds bottom-aligned decoder masks. A query at position {@code p} sees key {@code j} when {@code
 * j <= p && p - j < window}. This matches Hugging Face Transformers' eager {@code
 * sliding_window_overlay} in {@code modeling_attn_mask_utils.py} (the revision pinned in {@code
 * tools/hf-reference/provenance.json}).
 */
public final class AttentionMask {

  private AttentionMask() {}

  /** Returns a BOOL {@code [queryLength, keyLength]} mask; true means attend. */
  public static MLXArray slidingWindow(MLXScope scope, int queryLength, int keyLength, int window) {
    if (window <= 0) {
      throw new IllegalArgumentException("slidingWindow: window must be positive");
    }
    if (queryLength <= 0 || keyLength < queryLength) {
      throw new IllegalArgumentException("slidingWindow: require 0 < queryLength <= keyLength");
    }
    MLXArray positions =
        MLXShape.reshape(
            MLX.arange(scope, keyLength - queryLength, keyLength, 1, DType.INT32),
            new int[] {queryLength, 1});
    MLXArray keys =
        MLXShape.reshape(MLX.arange(scope, 0, keyLength, 1, DType.INT32), new int[] {1, keyLength});
    MLXArray causal = MLXOps.lessEqual(keys, positions);
    MLXArray distance = MLXOps.subtract(positions, keys);
    MLXArray limit = MLX.array(scope, new int[] {window}, new int[] {1});
    return MLXOps.logicalAnd(causal, MLXOps.less(distance, limit));
  }
}
