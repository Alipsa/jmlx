package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** Non-appending projected cross-attention state owned by one request scope. */
public final class StaticKVCache {
  private final MLXScope scope;
  private MLXArray keys;
  private MLXArray values;

  /** Creates an uninitialized cache in an explicit request owner. */
  public StaticKVCache(MLXScope scope) {
    this.scope = Objects.requireNonNull(scope);
    scope.checkAccess();
  }

  /**
   * Adopts already-evaluated [B,H,S,D] projections, moving them into the owner scope via {@link
   * MLX#hoist}; may be called only once.
   */
  public void initialize(MLXArray keys, MLXArray values) {
    scope.checkAccess();
    if (initialized()) {
      throw new IllegalStateException("static K/V cache is already initialized");
    }
    if (keys.ndim() != 4
        || !Arrays.equals(keys.shape(), values.shape())
        || Arrays.stream(keys.shape()).anyMatch(d -> d <= 0)) {
      throw new IllegalArgumentException("static K/V requires equal positive [B,H,S,D] dimensions");
    }
    MLX.eval(keys, values);
    this.keys = MLX.hoist(keys, scope);
    this.values = MLX.hoist(values, scope);
    MLX.eval(this.keys, this.values);
  }

  /** Whether projections have been initialized. */
  public boolean initialized() {
    scope.checkAccess();
    return keys != null;
  }

  /** Validates source dimensions before the first decoder step. */
  public void validate(int batch, int heads, int sourceLength, int headDim) {
    if (!Arrays.equals(keys().shape(), new int[] {batch, heads, sourceLength, headDim})) {
      throw new IllegalArgumentException("static K/V dimensions differ from encoder output");
    }
  }

  /** Returns keys while the request scope remains open. */
  public MLXArray keys() {
    scope.checkAccess();
    if (keys == null) {
      throw new IllegalStateException("static K/V cache is not initialized");
    }
    return keys;
  }

  /** Returns values while the request scope remains open. */
  public MLXArray values() {
    keys();
    return values;
  }
}
