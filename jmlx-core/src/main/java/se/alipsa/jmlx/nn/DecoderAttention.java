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

  /** Creates attention from registered projection modules. */
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
    scale = (float) (1.0 / Math.sqrt(headDim));
    this.rope = Objects.requireNonNull(rope, "rope");
    this.staticFreqs = staticFreqs;
    this.slidingWindow = slidingWindow;
    queryProj = child("queryProj", Objects.requireNonNull(q, "q"));
    keyProj = child("keyProj", Objects.requireNonNull(k, "k"));
    valueProj = child("valueProj", Objects.requireNonNull(v, "v"));
    outProj = child("outProj", Objects.requireNonNull(out, "out"));
  }

  @Override
  public MLXArray forward(MLXArray x, KVCache cache, MLXArray attentionMask) {
    Objects.requireNonNull(x, "x");
    int[] shape = x.shape();
    if (shape.length != 3 || shape[1] <= 0) {
      throw new IllegalArgumentException("DecoderAttention: x must be [batch, sequence, hidden]");
    }
    int batch = shape[0];
    int sequence = shape[1];
    int offset = cache == null ? 0 : cache.offset();
    int keyLength = offset + sequence;
    boolean windowed = slidingWindow != null && slidingWindow < keyLength;
    if (windowed) {
      if (attentionMask == null) {
        throw new IllegalArgumentException("slidingWindow requires an attentionMask");
      }
      if (attentionMask.dtype() != DType.BOOL
          || !Arrays.equals(attentionMask.shape(), new int[] {sequence, keyLength})) {
        throw new IllegalArgumentException(
            "slidingWindow attentionMask must be BOOL [" + sequence + ", " + keyLength + "]");
      }
    }
    MLXArray q = toHeads(queryProj.forward(x), batch, sequence, numHeads, headDim);
    MLXArray k = toHeads(keyProj.forward(x), batch, sequence, numKeyValueHeads, headDim);
    MLXArray v = toHeads(valueProj.forward(x), batch, sequence, numKeyValueHeads, headDim);
    q = rope.apply(q, rotaryDims, offset, staticFreqs);
    k = rope.apply(k, rotaryDims, offset, staticFreqs);
    if (cache != null) {
      cache.append(k, v);
      k = cache.keys();
      v = cache.values();
    }
    MLXArray attended = attendWithMask(q, k, v, windowed ? attentionMask : null);
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
        repeatKeyValueHeads(k, q.scope()),
        repeatKeyValueHeads(v, q.scope()),
        scale,
        mask == null,
        mask,
        null);
  }

  private static MLXArray toHeads(
      MLXArray projected, int batch, int sequence, int heads, int headDim) {
    int[] expected = {batch, sequence, heads * headDim};
    if (!Arrays.equals(projected.shape(), expected)) {
      throw new IllegalArgumentException(
          "projection must produce "
              + Arrays.toString(expected)
              + ", got "
              + Arrays.toString(projected.shape()));
    }
    return MLXShape.transpose(
        MLXShape.reshape(projected, new int[] {batch, sequence, heads, headDim}),
        new int[] {0, 2, 1, 3});
  }

  private MLXArray repeatKeyValueHeads(MLXArray value, MLXScope target) {
    if (queriesPerKeyValueHead == 1) {
      return value;
    }
    int[] shape = value.shape();
    MLXArray expanded = MLXShape.expandDims(value, target, 2);
    MLXArray broadcast =
        MLXShape.broadcastTo(
            expanded,
            target,
            new int[] {shape[0], numKeyValueHeads, queriesPerKeyValueHead, shape[2], shape[3]});
    return MLXShape.flatten(broadcast, target, 1, 2);
  }
}
