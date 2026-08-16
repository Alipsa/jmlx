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
  private MLXArray keys;
  private MLXArray values;
  private int offset;
  private boolean ownsKeys; // false when the first hoist returned k unchanged (same scope as this)
  private boolean ownsValues; // same, for v -- tracked independently since hoist checks each alone

  /** Creates an empty cache whose accumulated tensors live in {@code scope}. */
  public KVCache(MLXScope scope) {
    this.scope = Objects.requireNonNull(scope, "KVCache: scope must not be null");
  }

  /** The number of positions already accumulated -- the position of the next appended row. */
  public int offset() {
    return offset;
  }

  /**
   * The accumulated keys, shape {@code [..., offset(), headDim]}, or {@code null} before the first
   * {@link #append}.
   */
  public MLXArray keys() {
    return keys;
  }

  /**
   * The accumulated values, shape {@code [..., offset(), headDim]}, or {@code null} before the
   * first {@link #append}.
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
    if (vs[seqAxis] != newLength) {
      throw new IllegalArgumentException(
          "KVCache.append: v's sequence length must match k's ("
              + newLength
              + "), got k shape "
              + Arrays.toString(ks)
              + ", v shape "
              + Arrays.toString(vs));
    }
    if (keys == null) {
      // MLX.hoist is a documented no-copy optimization when its argument is already in the target
      // scope -- when that happens here, keys/values alias the caller's own arrays, and the next
      // append must not close() them. Tracked independently for keys/values: k and v may live in
      // different (though each individually valid) scopes.
      keys = MLX.hoist(k, scope);
      values = MLX.hoist(v, scope);
      ownsKeys = keys != k;
      ownsValues = values != v;
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
  }

  /** Closes whichever of the superseded {@code keys}/{@code values} this cache itself owns. */
  private void replaceAccumulated(MLXArray newKeys, MLXArray newValues) {
    if (ownsKeys) {
      keys.close();
    }
    if (ownsValues) {
      values.close();
    }
    keys = newKeys;
    values = newValues;
    ownsKeys = true;
    ownsValues = true;
  }
}
