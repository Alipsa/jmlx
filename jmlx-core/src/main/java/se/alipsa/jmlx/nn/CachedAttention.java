package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** Attention layer with optional key/value cache and a shared attention mask. */
public abstract class CachedAttention extends Module {

  /** Creates an attention layer owned by {@code scope}. */
  protected CachedAttention(MLXScope scope) {
    super(scope);
  }

  /** Applies attention, optionally using a cache and an explicit mask. */
  public abstract MLXArray forward(MLXArray x, KVCache cache, MLXArray attentionMask);

  /**
   * Applies attention with rotary frequencies the caller built once for this step, or {@code null}
   * to let the layer derive its own. Layers without dynamic rotary scaling ignore them.
   */
  public MLXArray forward(
      MLXArray x, KVCache cache, MLXArray attentionMask, MLXArray stepFrequencies) {
    return forward(x, cache, attentionMask);
  }

  /** Applies attention without an explicit mask. */
  public MLXArray forward(MLXArray x, KVCache cache) {
    return forward(x, cache, null);
  }
}
