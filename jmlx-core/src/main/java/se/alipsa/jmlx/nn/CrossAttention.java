package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** Query projections attend over independently projected, request-owned encoder keys/values. */
public final class CrossAttention extends Module {
  private final BidirectionalAttention attention;

  /** Creates rectangular no-RoPE attention with explicit dimensions and scale. */
  public CrossAttention(
      MLXScope scope,
      int heads,
      int headDim,
      float scale,
      UnaryLayer query,
      UnaryLayer key,
      UnaryLayer value,
      UnaryLayer output) {
    super(scope);
    attention =
        child(
            "attention",
            new BidirectionalAttention(scope, heads, headDim, scale, query, key, value, output));
  }

  /** Initializes encoder projections once per request. */
  public void initialize(MLXArray encoderOutput, StaticKVCache cache) {
    attention.initialize(encoderOutput, cache);
  }

  /** Attends to static keys/values with a key-only padding mask. */
  public MLXArray forward(MLXArray queries, StaticKVCache cache, MLXArray sourceMask) {
    return attention.forward(queries, cache, sourceMask);
  }
}
