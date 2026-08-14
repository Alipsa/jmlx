package se.alipsa.jmlx.nn;

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
 */
public final class KVCache {

  private final MLXScope scope;
  private MLXArray keys;
  private MLXArray values;
  private int offset;

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
   * @throws IllegalArgumentException if {@code k}/{@code v}'s scope is neither this cache's own
   *     scope nor a descendant of it (via {@link MLX#hoist}), or (from {@link
   *     MLXShape#concatenate}) if their shape disagrees with the existing accumulated tensor on any
   *     axis but the sequence axis
   */
  public void append(MLXArray k, MLXArray v) {
    Objects.requireNonNull(k, "KVCache.append: k must not be null");
    Objects.requireNonNull(v, "KVCache.append: v must not be null");
    int seqAxis = k.ndim() - 2;
    int newLength = k.shape()[seqAxis];
    if (keys == null) {
      keys = MLX.hoist(k, scope);
      values = MLX.hoist(v, scope);
    } else {
      MLXArray concatenatedKeys = MLXShape.concatenate(new MLXArray[] {keys, k}, seqAxis);
      MLXArray concatenatedValues = MLXShape.concatenate(new MLXArray[] {values, v}, seqAxis);
      MLXArray hoistedKeys = MLX.hoist(concatenatedKeys, scope);
      MLXArray hoistedValues = MLX.hoist(concatenatedValues, scope);
      keys.close();
      values.close();
      keys = hoistedKeys;
      values = hoistedValues;
    }
    offset += newLength;
  }
}
