package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Accumulates the key/value tensors of a multi-head-attention layer across decode steps. Not a
 * {@link Module}: the accumulated tensors are activations, not trainable parameters -- {@code
 * Module.rebind}'s value-swap contract has no meaning for them.
 *
 * <p>{@link #append}'s {@code k}/{@code v} may live in any scope that is this cache's own {@code
 * scope} or a <em>descendant</em> of it (a per-step child scope, in the normal decode-loop shape)
 * -- {@link MLX#hoist} enforces this and reports a clear {@link IllegalArgumentException} if
 * violated, so this class adds no separate check.
 *
 * <p>{@code append} explicitly closes the superseded {@code keys}/{@code values} handle after
 * hoisting each replacement -- safe because an mlx {@code ArrayDesc} owns its graph inputs by value
 * (req/phase4-plan.md §2, Research findings): the freshly concatenated array's graph holds its own
 * copy of the superseded array's wrapper, sharing the same descriptor, so closing this class's own
 * handle only decrements a refcount it does not solely own. Without this, active memory would grow
 * by one full copy of "everything appended so far" every step -- {@code O(N^2)} in total appended
 * length after {@code N} steps, not {@code O(N)} -- exactly the shape req/phase4-plan.md's Context
 * section opens with.
 *
 * <p>The {@code O(N)} guarantee above is for retained <em>data</em>, not for the work a caller who
 * never forces evaluation defers: if {@code N} {@code append}s happen with no intervening {@link
 * MLX#eval} (or other read of the accumulated tensors), the unevaluated {@code concatenate} chain
 * still does {@code O(N^2)} work, with all intermediates transiently live, at the eventual single
 * {@code eval} -- not violated by {@code KVCacheTest} or {@code MultiHeadAttentionTest}, both of
 * which evaluate every step, but not enforced by this class either.
 */
public final class KVCache {

  private final MLXScope scope;
  private final KVCachePolicy policy;
  private MLXArray keys;
  private MLXArray values;
  private int offset;
  private int startPosition;
  private int[] rowNextPositions;
  private int[] rowStartPositions;
  private boolean poisoned;
  private boolean ownsKeys; // false when the first hoist returned k unchanged (same scope as this)
  private boolean ownsValues; // same, for v -- tracked independently since hoist checks each alone

  /** Creates an empty cache whose accumulated tensors live in {@code scope}. */
  public KVCache(MLXScope scope) {
    this(scope, KVCachePolicy.full());
  }

  /** Creates an empty cache with a resolved retention policy. */
  public KVCache(MLXScope scope, KVCachePolicy policy) {
    this.scope = Objects.requireNonNull(scope, "KVCache: scope must not be null");
    this.policy = Objects.requireNonNull(policy, "policy");
  }

  /** Returns the resolved cache policy. */
  public KVCachePolicy policy() {
    return policy;
  }

  /** Absolute position of the first retained key; rejects unequal batch-row starts. */
  public int startPosition() {
    requireUniform(rowStartPositions, "startPosition");
    return rowStartPositions == null ? 0 : rowStartPositions[0];
  }

  /** Absolute position of one row's first valid retained key. */
  public int startPosition(int row) {
    return rowStartPositions[row];
  }

  /** Absolute position of the next key; rejects unequal batch-row positions. */
  public int nextPosition() {
    requireUniform(rowNextPositions, "nextPosition");
    return rowNextPositions == null ? 0 : rowNextPositions[0];
  }

  /** Absolute next position for one row. */
  public int nextPosition(int row) {
    return rowNextPositions[row];
  }

  /** Physical retained width; for a batch, shorter rows are left padded to this width. */
  public int length() {
    return keys == null ? 0 : keys.shape()[keys.ndim() - 2];
  }

  /** Number of batch rows after the first append, or zero while empty. */
  public int batchSize() {
    return rowNextPositions == null ? 0 : rowNextPositions.length;
  }

  /** Valid retained keys for one row, excluding left padding. */
  public int rowLength(int row) {
    return rowNextPositions[row] - rowStartPositions[row];
  }

  /** Whether all batch rows have the same position range. */
  public boolean isUniform() {
    return uniform(rowNextPositions) && uniform(rowStartPositions);
  }

  private static void requireUniform(int[] values, String name) {
    if (!uniform(values)) {
      throw new IllegalStateException(name + " is undefined for unequal batch positions");
    }
  }

  private static boolean uniform(int[] values) {
    if (values == null) {
      return true;
    }
    for (int value : values) {
      if (value != values[0]) {
        return false;
      }
    }
    return true;
  }

  /** Whether a failed forward requires this cache to be reset before reuse. */
  public boolean isPoisoned() {
    return poisoned;
  }

  /** Marks a cache whose lazy forward failed after mutation as unusable until reset. */
  public void poison() {
    poisoned = true;
  }

  /** Releases owned accumulated tensors and resets positions and failure state. */
  public void reset() {
    if (ownsKeys) {
      keys.close();
    }
    if (ownsValues) {
      values.close();
    }
    keys = null;
    values = null;
    offset = 0;
    startPosition = 0;
    rowNextPositions = null;
    rowStartPositions = null;
    ownsKeys = false;
    ownsValues = false;
    poisoned = false;
  }

  /** The number of positions already accumulated -- the position of the next appended row. */
  public int offset() {
    return nextPosition();
  }

  /**
   * The accumulated keys, shape {@code [..., length(), headDim]}, or {@code null} before the first
   * {@link #append}. Batched rows are left padded; use {@link #rowLength(int)} to exclude padding.
   */
  public MLXArray keys() {
    return keys;
  }

  /**
   * The accumulated values, shape {@code [..., length(), headDim]}, or {@code null} before the
   * first {@link #append}. Batched rows share the keys' left-padded layout.
   */
  public MLXArray values() {
    return values;
  }

  /**
   * Appends {@code k}/{@code v} (shape {@code [..., T, headDim]}, second-to-last axis the sequence
   * position -- matching {@code MLXFast.rope}'s own position-axis convention) and advances {@link
   * #offset()} by their sequence length. On the first call, hoists {@code k}/ {@code v} directly
   * (nothing to concatenate against); on every later call, concatenates against the existing
   * accumulated tensor, hoists the result into this cache's own scope, then closes the superseded
   * handle -- see this class's javadoc for why that close is both necessary and safe.
   *
   * @throws NullPointerException if {@code k} or {@code v} is {@code null}
   * @throws IllegalArgumentException if {@code k} has rank &lt; 2, if {@code v}'s rank disagrees
   *     with {@code k}'s, if any of their shared leading (batch) axes disagree, if their sequence
   *     lengths (second-to-last axis) disagree, or if {@code k}/{@code v}'s scope is neither this
   *     cache's own scope nor a descendant of it (via {@link MLX#hoist}), or (from {@link
   *     MLXShape#concatenate}) if their shape disagrees with the existing accumulated tensor on any
   *     axis but the sequence axis. {@code k}'s and {@code v}'s last axis (head dim) may differ
   *     from each other -- the two tensors are concatenated independently, so that is not a
   *     cross-tensor invariant this class needs.
   */
  public void append(MLXArray k, MLXArray v) {
    if (poisoned) {
      throw new IllegalStateException("KVCache is poisoned; reset before reuse");
    }
    Objects.requireNonNull(k, "KVCache.append: k must not be null");
    Objects.requireNonNull(v, "KVCache.append: v must not be null");
    int[] ks = k.shape();
    int[] vs = v.shape();
    if (ks.length < 2) {
      throw new IllegalArgumentException(
          "KVCache.append: k must have rank >= 2 (shape [..., T, headDim]), got shape "
              + Arrays.toString(ks));
    }
    if (vs.length != ks.length) {
      throw new IllegalArgumentException(
          "KVCache.append: v's rank must match k's ("
              + ks.length
              + "), got k shape "
              + Arrays.toString(ks)
              + ", v shape "
              + Arrays.toString(vs));
    }
    int seqAxis = ks.length - 2;
    for (int axis = 0; axis < ks.length - 1; axis++) {
      if (axis != seqAxis && ks[axis] != vs[axis]) {
        throw new IllegalArgumentException(
            "KVCache.append: v's leading (batch) axes must match k's, got k shape "
                + Arrays.toString(ks)
                + ", v shape "
                + Arrays.toString(vs));
      }
    }
    int newLength = ks[seqAxis];
    if (newLength <= 0) {
      throw new IllegalArgumentException("KVCache.append: sequence length must be positive");
    }
    long ending = (long) offset + newLength;
    if (ending > Integer.MAX_VALUE
        || (policy.mode() == KVCachePolicy.Mode.FULL
            && policy.limit() > 0
            && ending > policy.limit())) {
      throw new IllegalArgumentException("KVCache.append: absolute position exceeds capacity");
    }
    if (vs[seqAxis] != newLength) {
      throw new IllegalArgumentException(
          "KVCache.append: v's sequence length must match k's ("
              + newLength
              + "), got k shape "
              + Arrays.toString(ks)
              + ", v shape "
              + Arrays.toString(vs));
    }
    if (keys != null) {
      int[] oldKeyShape = keys.shape();
      int[] oldValueShape = values.shape();
      if (oldKeyShape.length != ks.length
          || oldValueShape.length != vs.length
          || keys.dtype() != k.dtype()
          || values.dtype() != v.dtype()) {
        throw new IllegalArgumentException("KVCache.append: rank or dtype changed");
      }
      for (int axis = 0; axis < ks.length; axis++) {
        if (axis != seqAxis && (oldKeyShape[axis] != ks[axis] || oldValueShape[axis] != vs[axis])) {
          throw new IllegalArgumentException(
              "KVCache.append: batch, head or feature shape changed");
        }
      }
    }
    if (keys == null) {
      // MLX.hoist is a documented no-copy optimization when its argument is already in the target
      // scope -- when that happens here, keys/values alias the caller's own arrays, and the next
      // append must not close() them. Tracked independently for keys/values: k and v may live in
      // different (though each individually valid) scopes.
      MLXArray firstKeys = MLX.hoist(k, scope);
      MLXArray firstValues;
      try {
        firstValues = MLX.hoist(v, scope);
      } catch (RuntimeException | Error failure) {
        if (firstKeys != k) {
          firstKeys.close();
        }
        throw failure;
      }
      keys = firstKeys;
      values = firstValues;
      ownsKeys = firstKeys != k;
      ownsValues = firstValues != v;
    } else if (length() == 0) {
      // A window of one retains no predecessor. Replace the empty tensors directly: native
      // concatenate need not support a zero-width input, and no old key must be copied.
      MLXArray firstKeys = MLX.hoist(k, scope);
      MLXArray firstValues;
      try {
        firstValues = MLX.hoist(v, scope);
      } catch (RuntimeException | Error failure) {
        if (firstKeys != k) {
          firstKeys.close();
        }
        throw failure;
      }
      replaceAccumulated(firstKeys, firstValues, firstKeys != k, firstValues != v);
    } else {
      MLXArray concatenatedKeys = MLXShape.concatenate(new MLXArray[] {keys, k}, seqAxis);
      MLXArray concatenatedValues = MLXShape.concatenate(new MLXArray[] {values, v}, seqAxis);
      // Both concatenates and both hoists must succeed before either superseded handle is closed:
      // if any of the four throws, keys/values still refer to open, valid arrays instead of a
      // handle this class already closed out from under itself, and neither field has been
      // reassigned yet -- a partial success (one tensor advanced, the other not) would desync
      // keys/values from each other and from offset. replaceAccumulated does the close+reassign as
      // a single step once both hoists are in hand, immediately after they're computed.
      MLXArray hoistedKeys = MLX.hoist(concatenatedKeys, scope);
      MLXArray hoistedValues = MLX.hoist(concatenatedValues, scope);
      replaceAccumulated(hoistedKeys, hoistedValues);
    }
    offset += newLength;
    if (rowNextPositions == null) {
      int rows = ks.length == 4 ? ks[0] : 1;
      rowNextPositions = new int[rows];
      rowStartPositions = new int[rows];
      Arrays.fill(rowNextPositions, offset);
    } else {
      for (int i = 0; i < rowNextPositions.length; i++) {
        rowNextPositions[i] += newLength;
      }
    }
  }

  /**
   * Appends a left-padded {@code [batch, heads, sequence, headDim]} chunk. Each row's valid tokens
   * occupy the rightmost {@code validLengths[row]} slots. The retained result is left padded to the
   * longest valid row; a one-token decode with all lengths one uses the uniform concatenate path.
   */
  public void append(MLXArray k, MLXArray v, int[] validLengths) {
    Objects.requireNonNull(k, "k");
    Objects.requireNonNull(v, "v");
    Objects.requireNonNull(validLengths, "validLengths");
    int[] ks = k.shape();
    int[] vs = v.shape();
    if (ks.length != 4
        || vs.length != 4
        || ks[0] <= 0
        || ks[2] <= 0
        || ks[0] != vs[0]
        || ks[1] != vs[1]
        || ks[2] != vs[2]
        || validLengths.length != ks[0]) {
      throw new IllegalArgumentException(
          "batched append requires matching [B,H,T,D] tensors and B lengths");
    }
    int maxValid = 0;
    for (int valid : validLengths) {
      if (valid <= 0 || valid > ks[2]) {
        throw new IllegalArgumentException("every batch row needs 1..T valid tokens");
      }
      maxValid = Math.max(maxValid, valid);
    }
    if (maxValid != ks[2]) {
      throw new IllegalArgumentException("left-padded width must equal the longest valid row");
    }
    if (rowNextPositions == null) {
      append(k, v);
      rowNextPositions = Arrays.copyOf(validLengths, validLengths.length);
      offset = maxValid;
      return;
    }
    boolean uniformChunk = true;
    for (int valid : validLengths) {
      uniformChunk &= valid == ks[2];
    }
    if (uniformChunk) {
      append(k, v);
      return;
    }
    if (ks[0] != rowNextPositions.length
        || ks[1] != keys.shape()[1]
        || ks[3] != keys.shape()[3]
        || vs[3] != values.shape()[3]
        || k.dtype() != keys.dtype()
        || v.dtype() != values.dtype()) {
      throw new IllegalArgumentException(
          "batched append changed cache batch, head, feature or dtype");
    }
    int nextWidth = 0;
    int maxNext = 0;
    for (int row = 0; row < validLengths.length; row++) {
      long next = (long) rowNextPositions[row] + validLengths[row];
      int width = rowLength(row) + validLengths[row];
      if (next > Integer.MAX_VALUE
          || (policy.mode() == KVCachePolicy.Mode.FULL
              && policy.limit() > 0
              && next > policy.limit())) {
        throw new IllegalArgumentException("batched append exceeds absolute position capacity");
      }
      nextWidth = Math.max(nextWidth, width);
      maxNext = Math.max(maxNext, (int) next);
    }
    if (policy.mode() == KVCachePolicy.Mode.FULL
        && policy.limit() > 0
        && nextWidth > policy.limit()) {
      throw new IllegalArgumentException("batched append exceeds padded-width capacity");
    }
    MLXArray nextKeys = repackRows(keys, k, validLengths, nextWidth);
    MLXArray nextValues = repackRows(values, v, validLengths, nextWidth);
    MLXArray hoistedKeys = MLX.hoist(nextKeys, scope);
    MLXArray hoistedValues = MLX.hoist(nextValues, scope);
    replaceAccumulated(hoistedKeys, hoistedValues);
    for (int row = 0; row < validLengths.length; row++) {
      rowNextPositions[row] += validLengths[row];
    }
    offset = maxNext;
  }

  private MLXArray repackRows(
      MLXArray old, MLXArray incoming, int[] validLengths, int paddedWidth) {
    MLXArray[] rows = new MLXArray[validLengths.length];
    int oldWidth = old.shape()[2];
    int inputWidth = incoming.shape()[2];
    for (int row = 0; row < rows.length; row++) {
      int retained = rowLength(row);
      int valid = validLengths[row];
      int[] oldStart = {row, 0, oldWidth - retained, 0};
      int[] oldStop = {row + 1, old.shape()[1], oldWidth, old.shape()[3]};
      int[] inputStart = {row, 0, inputWidth - valid, 0};
      int[] inputStop = {row + 1, incoming.shape()[1], inputWidth, incoming.shape()[3]};
      MLXArray addition = MLXShape.slice(incoming, inputStart, inputStop);
      MLXArray rowData =
          retained == 0
              ? addition
              : MLXShape.concatenate(
                  new MLXArray[] {MLXShape.slice(old, oldStart, oldStop), addition}, 2);
      int pad = paddedWidth - retained - valid;
      rows[row] =
          pad == 0
              ? rowData
              : MLXShape.concatenate(
                  new MLXArray[] {
                    MLX.zeros(
                        incoming.scope(),
                        new int[] {1, old.shape()[1], pad, old.shape()[3]},
                        old.dtype()),
                    rowData
                  },
                  2);
    }
    return MLXShape.concatenate(rows, 0);
  }

  /**
   * Keeps the last {@code count} keys after an attention operation. Only sliding caches may evict.
   * The absolute next position remains unchanged.
   */
  public void trimToLast(int count) {
    if (poisoned) {
      throw new IllegalStateException("KVCache is poisoned; reset before reuse");
    }
    if (!policy.evicts()) {
      throw new IllegalStateException("FULL cache must not evict keys");
    }
    if (count < 0 || count > length() || count >= policy.limit()) {
      throw new IllegalArgumentException("trim count outside retained cache length");
    }
    if (count == length()) {
      return;
    }
    MLXArray trimmedKeys;
    MLXArray trimmedValues;
    if (count == 0) {
      int[] keyShape = keys.shape();
      int[] valueShape = values.shape();
      keyShape[keyShape.length - 2] = 0;
      valueShape[valueShape.length - 2] = 0;
      trimmedKeys = MLX.zeros(scope, keyShape, keys.dtype());
      trimmedValues = MLX.zeros(scope, valueShape, values.dtype());
    } else {
      int[] positions = new int[count];
      for (int i = 0; i < count; i++) {
        positions[i] = length() - count + i;
      }
      MLXArray indices = MLX.array(scope, positions, new int[] {count});
      trimmedKeys = MLXShape.takeAxis(keys, indices, keys.ndim() - 2);
      trimmedValues = MLXShape.takeAxis(values, indices, values.ndim() - 2);
    }
    replaceAccumulated(trimmedKeys, trimmedValues);
    startPosition = offset - count;
    for (int i = 0; i < rowStartPositions.length; i++) {
      rowStartPositions[i] = Math.max(rowStartPositions[i], rowNextPositions[i] - count);
    }
  }

  /**
   * Copies this cache into {@code destinationScope}. The gathered tensors are evaluated before
   * return so the copy no longer depends on the source graph or its backing storage.
   */
  public KVCache fork(MLXScope destinationScope) {
    Objects.requireNonNull(destinationScope, "destinationScope");
    if (poisoned) {
      throw new IllegalStateException("KVCache is poisoned; reset before reuse");
    }
    if (keys == null) {
      return new KVCache(destinationScope, policy);
    }
    int axis = keys.ndim() == 4 ? 0 : keys.ndim() - 2;
    int[] identity = new int[keys.shape()[axis]];
    for (int i = 0; i < identity.length; i++) {
      identity[i] = i;
    }
    return copyAlongAxis(identity, axis, destinationScope);
  }

  /**
   * Returns an independently materialized batch copy in {@code destinationScope}. Repeated indices
   * duplicate rows; the source stays valid.
   */
  public KVCache reorder(int[] indices, MLXScope destinationScope) {
    Objects.requireNonNull(indices, "indices");
    Objects.requireNonNull(destinationScope, "destinationScope");
    if (indices.length == 0) {
      throw new IllegalArgumentException("reorder requires at least one row");
    }
    if (keys == null) {
      throw new IllegalStateException("cannot reorder an uninitialized cache");
    }
    if (keys.ndim() != 4) {
      throw new IllegalArgumentException("reorder requires [batch, heads, sequence, headDim]");
    }
    for (int index : indices) {
      if (index < 0 || index >= keys.shape()[0]) {
        throw new IllegalArgumentException("reorder index outside cache batch");
      }
    }
    if (poisoned) {
      throw new IllegalStateException("KVCache is poisoned; reset before reuse");
    }
    return copyAlongAxis(indices, 0, destinationScope);
  }

  private KVCache copyAlongAxis(int[] indices, int axis, MLXScope destinationScope) {
    if (axis == 0 && length() == 0) {
      KVCache empty = new KVCache(destinationScope, policy);
      int[] keyShape = keys.shape();
      int[] valueShape = values.shape();
      keyShape[0] = indices.length;
      valueShape[0] = indices.length;
      empty.keys = MLX.zeros(destinationScope, keyShape, keys.dtype());
      empty.values = MLX.zeros(destinationScope, valueShape, values.dtype());
      empty.ownsKeys = true;
      empty.ownsValues = true;
      empty.rowNextPositions = new int[indices.length];
      empty.rowStartPositions = new int[indices.length];
      for (int i = 0; i < indices.length; i++) {
        empty.rowNextPositions[i] = rowNextPositions[indices[i]];
        empty.rowStartPositions[i] = rowStartPositions[indices[i]];
      }
      empty.offset = Arrays.stream(empty.rowNextPositions).max().orElse(0);
      empty.startPosition = Arrays.stream(empty.rowStartPositions).min().orElse(0);
      MLX.eval(empty.keys, empty.values);
      return empty;
    }
    if (indices.length == 0) {
      KVCache empty = new KVCache(destinationScope, policy);
      empty.keys = MLX.zeros(destinationScope, keys.shape(), keys.dtype());
      empty.values = MLX.zeros(destinationScope, values.shape(), values.dtype());
      empty.ownsKeys = true;
      empty.ownsValues = true;
      empty.offset = offset;
      empty.startPosition = startPosition;
      empty.rowNextPositions = rowNextPositions.clone();
      empty.rowStartPositions = rowStartPositions.clone();
      MLX.eval(empty.keys, empty.values);
      return empty;
    }
    MLXArray indexArray = MLX.array(destinationScope, indices, new int[] {indices.length});
    KVCache result = new KVCache(destinationScope, policy);
    MLXArray sourceKeys = keys;
    MLXArray sourceValues = values;
    if (axis == 0) {
      int longest = 0;
      for (int index : indices) {
        longest = Math.max(longest, rowLength(index));
      }
      if (longest < length()) {
        int[] positions = new int[longest];
        for (int i = 0; i < longest; i++) {
          positions[i] = length() - longest + i;
        }
        if (longest == 0) {
          int[] keyShape = keys.shape();
          int[] valueShape = values.shape();
          keyShape[2] = 0;
          valueShape[2] = 0;
          sourceKeys = MLX.zeros(scope, keyShape, keys.dtype());
          sourceValues = MLX.zeros(scope, valueShape, values.dtype());
        } else {
          MLXArray sequenceIndices = MLX.array(scope, positions, new int[] {longest});
          sourceKeys = MLXShape.takeAxis(keys, sequenceIndices, 2);
          sourceValues = MLXShape.takeAxis(values, sequenceIndices, 2);
        }
      }
    }
    MLXArray copiedKeys = MLXShape.takeAxis(sourceKeys, indexArray, destinationScope, axis);
    MLXArray copiedValues;
    try {
      copiedValues = MLXShape.takeAxis(sourceValues, indexArray, destinationScope, axis);
    } catch (RuntimeException | Error failure) {
      copiedKeys.close();
      throw failure;
    }
    try {
      MLX.eval(copiedKeys, copiedValues);
      result.keys = copiedKeys;
      result.values = copiedValues;
      result.ownsKeys = true;
      result.ownsValues = true;
      result.offset = offset;
      result.startPosition = startPosition;
      result.rowNextPositions = new int[axis == 0 ? indices.length : rowNextPositions.length];
      result.rowStartPositions = new int[result.rowNextPositions.length];
      for (int i = 0; i < result.rowNextPositions.length; i++) {
        int sourceRow = axis == 0 ? indices[i] : i;
        result.rowNextPositions[i] = rowNextPositions[sourceRow];
        result.rowStartPositions[i] = rowStartPositions[sourceRow];
      }
      result.offset = Arrays.stream(result.rowNextPositions).max().orElse(0);
      result.startPosition = Arrays.stream(result.rowStartPositions).min().orElse(0);
      return result;
    } catch (RuntimeException | Error failure) {
      copiedKeys.close();
      copiedValues.close();
      throw failure;
    }
  }

  /** Closes whichever of the superseded {@code keys}/{@code values} this cache itself owns. */
  private void replaceAccumulated(MLXArray newKeys, MLXArray newValues) {
    replaceAccumulated(newKeys, newValues, true, true);
  }

  private void replaceAccumulated(
      MLXArray newKeys, MLXArray newValues, boolean newOwnsKeys, boolean newOwnsValues) {
    if (ownsKeys) {
      keys.close();
    }
    if (ownsValues) {
      values.close();
    }
    keys = newKeys;
    values = newValues;
    ownsKeys = newOwnsKeys;
    ownsValues = newOwnsValues;
  }
}
