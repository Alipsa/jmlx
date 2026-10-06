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

  /** Expands a binary key-padding mask to BOOL [B,1,T,S], including padded queries. */
  public static MLXArray bidirectional(MLXScope scope, int[][] padding, int queryLength) {
    Objects.requireNonNull(padding, "padding");
    if (padding.length == 0 || padding[0] == null || padding[0].length == 0 || queryLength <= 0) {
      throw new IllegalArgumentException("bidirectional mask requires positive dimensions");
    }
    int width = padding[0].length;
    int[] flat = new int[Math.multiplyExact(padding.length, width)];
    for (int row = 0; row < padding.length; row++) {
      if (padding[row] == null || padding[row].length != width) {
        throw new IllegalArgumentException("padding rows must have equal lengths");
      }
      int valid = 0;
      for (int col = 0; col < width; col++) {
        int value = padding[row][col];
        if (value != 0 && value != 1) {
          throw new IllegalArgumentException("padding must be binary");
        }
        valid += value;
        flat[row * width + col] = value;
      }
      if (valid == 0) {
        throw new IllegalArgumentException("bidirectional input has no valid key");
      }
    }
    MLXArray keys =
        MLX.astype(MLX.array(scope, flat, new int[] {padding.length, 1, 1, width}), DType.BOOL);
    return MLXShape.broadcastTo(keys, new int[] {padding.length, 1, queryLength, width});
  }

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
    int[] queryOffsets = new int[batch];
    int[] queryPaddings = new int[batch];
    int[] keyOffsets = new int[batch];
    int[] keyPaddings = new int[batch];
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
      queryPaddings[row] = queryWidth - validQueries;
      keyPaddings[row] = keyWidth - (int) retainedLong;
      queryOffsets[row] = queryStarts[row] - queryPaddings[row];
      keyOffsets[row] = keyStarts[row] - keyPaddings[row];
    }
    int[] column = {batch, 1, 1, 1};
    MLXArray queryIndex =
        MLXShape.reshape(
            MLX.arange(scope, 0, queryWidth, 1, DType.INT32), new int[] {1, 1, queryWidth, 1});
    MLXArray keyIndex =
        MLXShape.reshape(
            MLX.arange(scope, 0, keyWidth, 1, DType.INT32), new int[] {1, 1, 1, keyWidth});
    MLXArray absoluteQuery = MLXOps.add(queryIndex, MLX.array(scope, queryOffsets, column));
    MLXArray absoluteKey = MLXOps.add(keyIndex, MLX.array(scope, keyOffsets, column));
    MLXArray validKey = MLXOps.greaterEqual(keyIndex, MLX.array(scope, keyPaddings, column));
    MLXArray visible = MLXOps.logicalAnd(validKey, MLXOps.lessEqual(absoluteKey, absoluteQuery));
    if (window > 0) {
      MLXArray limit = MLX.array(scope, new int[] {window}, new int[] {1});
      visible =
          MLXOps.logicalAnd(
              visible, MLXOps.less(MLXOps.subtract(absoluteQuery, absoluteKey), limit));
    }
    MLXArray paddedQuery = MLXOps.less(queryIndex, MLX.array(scope, queryPaddings, column));
    MLXArray lastKey =
        MLXOps.equal(keyIndex, MLX.array(scope, new int[] {keyWidth - 1}, new int[] {1}));
    return MLXOps.where(paddedQuery, lastKey, visible);
  }
}
