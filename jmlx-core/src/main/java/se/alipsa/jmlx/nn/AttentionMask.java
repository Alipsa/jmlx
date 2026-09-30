package se.alipsa.jmlx.nn;

import java.util.Objects;
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
    return slidingWindow(scope, keyLength - queryLength, 0, queryLength, keyLength, window);
  }

  /**
   * Returns a BOOL {@code [queryLength, keyLength]} mask using absolute query and key positions.
   * This supports a retained cache whose first key is no longer at position zero.
   */
  public static MLXArray slidingWindow(
      MLXScope scope, int queryStart, int keyStart, int queryLength, int keyLength, int window) {
    if (window <= 0) {
      throw new IllegalArgumentException("slidingWindow: window must be positive");
    }
    if (queryStart < 0
        || keyStart < 0
        || queryLength <= 0
        || keyLength < queryLength
        || (long) queryStart + queryLength > Integer.MAX_VALUE
        || (long) keyStart + keyLength > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("slidingWindow: invalid absolute positions or lengths");
    }
    MLXArray positions =
        MLXShape.reshape(
            MLX.arange(scope, queryStart, queryStart + queryLength, 1, DType.INT32),
            new int[] {queryLength, 1});
    MLXArray keys =
        MLXShape.reshape(
            MLX.arange(scope, keyStart, keyStart + keyLength, 1, DType.INT32),
            new int[] {1, keyLength});
    MLXArray causal = MLXOps.lessEqual(keys, positions);
    MLXArray distance = MLXOps.subtract(positions, keys);
    MLXArray limit = MLX.array(scope, new int[] {window}, new int[] {1});
    return MLXOps.logicalAnd(causal, MLXOps.less(distance, limit));
  }

  /**
   * Builds a BOOL {@code [B,1,T,S]} causal mask for left-padded batch rows. {@code window == 0}
   * means full attention. Invalid padded queries are assigned one valid key so native attention
   * never sees an all-masked row; callers must discard their output positions.
   */
  public static MLXArray batched(
      MLXScope scope,
      int[] queryStarts,
      int[] keyStarts,
      int[] validQueryLengths,
      int queryWidth,
      int keyWidth,
      int window) {
    Objects.requireNonNull(queryStarts, "queryStarts");
    Objects.requireNonNull(keyStarts, "keyStarts");
    Objects.requireNonNull(validQueryLengths, "validQueryLengths");
    int batch = queryStarts.length;
    if (batch == 0
        || keyStarts.length != batch
        || validQueryLengths.length != batch
        || queryWidth <= 0
        || keyWidth <= 0
        || window < 0) {
      throw new IllegalArgumentException("invalid batched attention dimensions");
    }
    int[] data = new int[Math.multiplyExact(batch, Math.multiplyExact(queryWidth, keyWidth))];
    for (int row = 0; row < batch; row++) {
      int validQueries = validQueryLengths[row];
      long retainedLong = (long) queryStarts[row] - keyStarts[row] + validQueries;
      if (queryStarts[row] < 0
          || keyStarts[row] < 0
          || validQueries <= 0
          || validQueries > queryWidth
          || (long) queryStarts[row] + validQueries > Integer.MAX_VALUE
          || retainedLong <= 0
          || retainedLong > keyWidth) {
        throw new IllegalArgumentException("invalid row positions or left padding");
      }
      int retained = (int) retainedLong;
      int keyPadding = keyWidth - retained;
      int queryPadding = queryWidth - validQueries;
      for (int q = 0; q < queryWidth; q++) {
        if (q < queryPadding) {
          data[(row * queryWidth + q) * keyWidth + keyWidth - 1] = 1;
          continue;
        }
        int absoluteQuery = queryStarts[row] + q - queryPadding;
        for (int key = keyPadding; key < keyWidth; key++) {
          int absoluteKey = keyStarts[row] + key - keyPadding;
          if (absoluteKey <= absoluteQuery
              && (window == 0 || absoluteQuery - absoluteKey < window)) {
            data[(row * queryWidth + q) * keyWidth + key] = 1;
          }
        }
      }
    }
    return MLX.astype(
        MLX.array(scope, data, new int[] {batch, 1, queryWidth, keyWidth}), DType.BOOL);
  }
}
