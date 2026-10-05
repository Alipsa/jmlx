package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Decoder self-attention with configurable head width, rotary frequencies, and attention window.
 *
 * <p>The optional per-head QK-normalization constructors add an {@link RMSNorm} over {@code
 * headDim} applied to the query and key heads after projection and before RoPE, matching
 * transformers' Qwen3 {@code q_norm}/{@code k_norm} ordering.
 */
public final class DecoderAttention extends CachedAttention {

  private final int numHeads;
  private final int numKeyValueHeads;
  private final int headDim;
  private final int rotaryDims;
  private final int queriesPerKeyValueHead;
  private final float scale;
  private final RopeSpec rope;
  private final MLXArray staticFreqs;
  private final Integer slidingWindow;
  private final UnaryLayer queryProj;
  private final UnaryLayer keyProj;
  private final UnaryLayer valueProj;
  private final UnaryLayer outProj;
  private final RMSNorm queryNorm;
  private final RMSNorm keyNorm;

  /** Creates attention from registered projection modules, without per-head QK normalization. */
  public DecoderAttention(
      MLXScope scope,
      int numHeads,
      int numKeyValueHeads,
      int headDim,
      RopeSpec rope,
      int rotaryDims,
      MLXArray staticFreqs,
      Integer slidingWindow,
      UnaryLayer q,
      UnaryLayer k,
      UnaryLayer v,
      UnaryLayer out) {
    this(
        scope,
        numHeads,
        numKeyValueHeads,
        headDim,
        rope,
        rotaryDims,
        staticFreqs,
        slidingWindow,
        q,
        k,
        v,
        out,
        null,
        null);
  }

  /**
   * Creates attention from registered projection modules with optional per-head QK normalization:
   * each non-null norm is an {@link RMSNorm} over {@code headDim} applied to the query or key heads
   * after projection and before RoPE (transformers' Qwen3 ordering).
   */
  public DecoderAttention(
      MLXScope scope,
      int numHeads,
      int numKeyValueHeads,
      int headDim,
      RopeSpec rope,
      int rotaryDims,
      MLXArray staticFreqs,
      Integer slidingWindow,
      UnaryLayer q,
      UnaryLayer k,
      UnaryLayer v,
      UnaryLayer out,
      RMSNorm queryNorm,
      RMSNorm keyNorm) {
    this(
        scope,
        numHeads,
        numKeyValueHeads,
        headDim,
        rope,
        rotaryDims,
        staticFreqs,
        slidingWindow,
        q,
        k,
        v,
        out,
        queryNorm,
        keyNorm,
        (float) (1.0 / Math.sqrt(headDim)));
  }

  /** Creates decoder attention with explicit scale and optional QK norms. */
  public DecoderAttention(
      MLXScope scope,
      int numHeads,
      int numKeyValueHeads,
      int headDim,
      RopeSpec rope,
      int rotaryDims,
      MLXArray staticFreqs,
      Integer slidingWindow,
      UnaryLayer q,
      UnaryLayer k,
      UnaryLayer v,
      UnaryLayer out,
      RMSNorm queryNorm,
      RMSNorm keyNorm,
      float scale) {
    super(scope);
    if (numHeads <= 0 || numKeyValueHeads <= 0 || numHeads % numKeyValueHeads != 0) {
      throw new IllegalArgumentException(
          "numHeads must be a positive multiple of numKeyValueHeads");
    }
    if (headDim <= 0 || rotaryDims <= 0 || rotaryDims > headDim || (rotaryDims & 1) != 0) {
      throw new IllegalArgumentException("headDim must be positive and rotaryDims even within it");
    }
    if (slidingWindow != null && slidingWindow <= 0) {
      throw new IllegalArgumentException("slidingWindow must be positive");
    }
    this.numHeads = numHeads;
    this.numKeyValueHeads = numKeyValueHeads;
    this.headDim = headDim;
    this.rotaryDims = rotaryDims;
    queriesPerKeyValueHead = numHeads / numKeyValueHeads;
    if (!Float.isFinite(scale) || scale <= 0) {
      throw new IllegalArgumentException("attention scale must be finite and positive");
    }
    this.scale = scale;
    this.rope = Objects.requireNonNull(rope, "rope");
    this.staticFreqs =
        staticFreqs != null ? staticFreqs : rope.staticFrequencies(scope, rotaryDims);
    this.slidingWindow = slidingWindow;
    queryProj = child("queryProj", Objects.requireNonNull(q, "q"));
    keyProj = child("keyProj", Objects.requireNonNull(k, "k"));
    valueProj = child("valueProj", Objects.requireNonNull(v, "v"));
    outProj = child("outProj", Objects.requireNonNull(out, "out"));
    if (queryNorm != null) {
      checkNormDimension(queryNorm, "queryNorm");
      queryNorm = child("queryNorm", queryNorm);
    }
    if (keyNorm != null) {
      checkNormDimension(keyNorm, "keyNorm");
      keyNorm = child("keyNorm", keyNorm);
    }
    this.queryNorm = queryNorm;
    this.keyNorm = keyNorm;
  }

  private void checkNormDimension(RMSNorm norm, String name) {
    int[] weight = norm.parameters().get("weight").shape();
    if (weight.length != 1 || weight[0] != headDim) {
      throw new IllegalArgumentException(
          name + " weight must be [headDim] == [" + headDim + "], got " + Arrays.toString(weight));
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>A non-null mask replaces causal masking, so it must already be causal where that is wanted.
   * It must be BOOL, shaped {@code [sequence, keyLength]} with no batch dimension, and is required
   * once the key length exceeds the sliding window.
   */
  @Override
  public MLXArray forward(MLXArray x, KVCache cache, MLXArray attentionMask) {
    return forward(x, cache, attentionMask, null);
  }

  @Override
  public MLXArray forward(
      MLXArray x, KVCache cache, MLXArray attentionMask, MLXArray stepFrequencies) {
    return forwardInternal(x, cache, attentionMask, stepFrequencies, null);
  }

  @Override
  public MLXArray forward(
      MLXArray x,
      KVCache cache,
      MLXArray attentionMask,
      MLXArray stepFrequencies,
      int[] validLengths) {
    return forwardInternal(x, cache, attentionMask, stepFrequencies, validLengths);
  }

  private MLXArray forwardInternal(
      MLXArray x,
      KVCache cache,
      MLXArray attentionMask,
      MLXArray stepFrequencies,
      int[] validLengths) {
    Objects.requireNonNull(x, "x");
    int[] shape = x.shape();
    if (shape.length != 3 || shape[0] <= 0 || shape[1] <= 0) {
      throw new IllegalArgumentException("DecoderAttention: x must be [batch, sequence, hidden]");
    }
    final int batch = shape[0];
    final int sequence = shape[1];
    if (validLengths != null) {
      if (validLengths.length != batch) {
        throw new IllegalArgumentException("one valid length is required per batch row");
      }
      if (Arrays.stream(validLengths).max().orElse(0) != sequence) {
        throw new IllegalArgumentException("left-padded sequence must equal the longest valid row");
      }
      if (cache != null && cache.batchSize() != 0 && cache.batchSize() != batch) {
        throw new IllegalArgumentException("cache batch size differs from x batch size");
      }
    }
    if (cache != null
        && cache.policy().evicts()
        && !Objects.equals(slidingWindow, cache.policy().limit())) {
      throw new IllegalArgumentException("SLIDING_WINDOW cache must match layer window");
    }
    if (cache != null && cache.isPoisoned()) {
      throw new IllegalStateException("DecoderAttention cache is poisoned; reset before reuse");
    }
    if (validLengths == null
        && cache != null
        && (!cache.isUniform()
            || (cache.batchSize() > 0 && cache.rowLength(0) != cache.length()))) {
      throw new IllegalArgumentException("left-padded cache requires batch valid lengths");
    }
    final int offset = validLengths == null && cache != null ? cache.offset() : 0;
    if (validLengths == null && (long) offset + sequence > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("attention position overflow");
    }
    int keyLength = validLengths == null ? (cache == null ? 0 : cache.length()) + sequence : 0;
    if (validLengths != null) {
      for (int row = 0; row < batch; row++) {
        if (validLengths[row] <= 0 || validLengths[row] > sequence) {
          throw new IllegalArgumentException("invalid batch row length");
        }
        if (cache != null
            && (long) cache.nextPosition(row) + validLengths[row] > Integer.MAX_VALUE) {
          throw new IllegalArgumentException("attention batch position overflow");
        }
        keyLength =
            Math.max(keyLength, (cache == null ? 0 : cache.rowLength(row)) + validLengths[row]);
      }
    }
    boolean windowed = slidingWindow != null && slidingWindow < keyLength;
    if (windowed && attentionMask == null) {
      throw new IllegalArgumentException("slidingWindow requires an attentionMask");
    }
    int[] expectedMask =
        validLengths == null
            ? new int[] {sequence, keyLength}
            : new int[] {batch, 1, sequence, keyLength};
    if (attentionMask != null
        && (attentionMask.dtype() != DType.BOOL
            || !Arrays.equals(attentionMask.shape(), expectedMask))) {
      throw new IllegalArgumentException(
          "attentionMask must be BOOL " + Arrays.toString(expectedMask));
    }
    MLXArray freqs = staticFreqs != null ? staticFreqs : stepFrequencies;
    if (freqs == null) {
      // Only dynamic specs reach here; computed once so q and k share one upload.
      int ending = offset + sequence;
      if (validLengths != null) {
        ending = 0;
        for (int row = 0; row < batch; row++) {
          ending =
              Math.max(ending, (cache == null ? 0 : cache.nextPosition(row)) + validLengths[row]);
        }
      }
      freqs = rope.stepFrequencies(x.scope(), rotaryDims, ending);
    }
    MLXArray q = AttentionHeads.toHeads(queryProj.forward(x), batch, sequence, numHeads, headDim);
    MLXArray k =
        AttentionHeads.toHeads(keyProj.forward(x), batch, sequence, numKeyValueHeads, headDim);
    MLXArray v =
        AttentionHeads.toHeads(valueProj.forward(x), batch, sequence, numKeyValueHeads, headDim);
    // QK normalization (Qwen3 q_norm/k_norm) sits between projection and RoPE; normalized keys
    // enter the cache, so cached rows are never re-normalized.
    if (queryNorm != null) {
      q = queryNorm.forward(q);
    }
    if (keyNorm != null) {
      k = keyNorm.forward(k);
    }
    if (validLengths == null) {
      q = rope.apply(q, rotaryDims, offset, freqs);
      k = rope.apply(k, rotaryDims, offset, freqs);
    } else {
      int[] positions = new int[batch];
      for (int row = 0; row < batch; row++) {
        positions[row] = cache == null ? 0 : cache.nextPosition(row);
      }
      q = rope.apply(q, rotaryDims, positions, validLengths, freqs);
      k = rope.apply(k, rotaryDims, positions, validLengths, freqs);
    }
    if (cache != null) {
      if (validLengths == null) {
        cache.append(k, v);
      } else {
        cache.append(k, v, validLengths);
      }
      k = cache.keys();
      v = cache.values();
    }
    MLXArray attended = attendWithMask(q, k, v, attentionMask);
    // The attention graph owns the complete keys by value. Evict only after constructing it.
    if (cache != null && cache.policy().evicts()) {
      cache.trimToLast(Math.min(cache.length(), slidingWindow - 1));
    }
    MLXArray merged = MLXShape.flatten(MLXShape.transpose(attended, new int[] {0, 2, 1, 3}), 2, 3);
    return outProj.forward(merged);
  }

  /** Attention seam for pre-rotated query/key heads and a complete cache. */
  MLXArray attend(MLXArray q, MLXArray k, MLXArray v, int offset) {
    int keyLength = k.shape()[2];
    if (offset != keyLength - q.shape()[2]) {
      throw new IllegalArgumentException("offset must equal keyLength - queryLength");
    }
    MLXArray mask =
        slidingWindow != null && slidingWindow < keyLength
            ? AttentionMask.slidingWindow(q.scope(), q.shape()[2], keyLength, slidingWindow)
            : null;
    return attendWithMask(q, k, v, mask);
  }

  private MLXArray attendWithMask(MLXArray q, MLXArray k, MLXArray v, MLXArray mask) {
    return MLXFast.scaledDotProductAttention(
        q,
        AttentionHeads.repeatKeyValueHeads(k, q.scope(), numKeyValueHeads, queriesPerKeyValueHead),
        AttentionHeads.repeatKeyValueHeads(v, q.scope(), numKeyValueHeads, queriesPerKeyValueHead),
        scale,
        mask == null,
        mask,
        null);
  }
}
