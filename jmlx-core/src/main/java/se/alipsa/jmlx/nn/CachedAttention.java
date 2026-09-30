package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** Attention layer with optional key/value cache and a shared attention mask. */
public abstract class CachedAttention extends Module {

  /** Creates an attention layer owned by {@code scope}. */
  protected CachedAttention(MLXScope scope) {
    super(scope);
  }

  /**
   * Applies attention, optionally using a cache and an explicit mask.
   *
   * <p>A non-null {@code attentionMask} replaces the causal mask rather than adding to it, so it
   * must already encode causality (as {@link AttentionMask#slidingWindow} does). It has no batch
   * dimension and is shaped {@code [sequence, keyLength]}.
   */
  public abstract MLXArray forward(MLXArray x, KVCache cache, MLXArray attentionMask);

  /**
   * Applies attention with rotary frequencies the caller built once for this step, or {@code null}
   * to let the layer derive its own. Layers without dynamic rotary scaling ignore them.
   */
  public MLXArray forward(
      MLXArray x, KVCache cache, MLXArray attentionMask, MLXArray stepFrequencies) {
    return forward(x, cache, attentionMask);
  }

  /**
   * Batch-aware extension. Existing subclasses continue to work for uniform, unpadded input; ragged
   * input requires an override that understands per-row positions and a batched mask.
   */
  public MLXArray forward(
      MLXArray x,
      KVCache cache,
      MLXArray attentionMask,
      MLXArray stepFrequencies,
      int[] validLengths) {
    if (validLengths == null || validLengths.length != x.shape()[0]) {
      throw new IllegalArgumentException("one valid length is required per batch row");
    }
    for (int valid : validLengths) {
      if (valid != x.shape()[1]) {
        throw new UnsupportedOperationException(
            "attention subclass does not support ragged batches");
      }
    }
    if (cache != null && !cache.isUniform()) {
      throw new UnsupportedOperationException(
          "attention subclass does not support unequal positions");
    }
    return forward(x, cache, attentionMask, stepFrequencies);
  }

  /** Applies attention without an explicit mask. */
  public MLXArray forward(MLXArray x, KVCache cache) {
    return forward(x, cache, null);
  }
}
