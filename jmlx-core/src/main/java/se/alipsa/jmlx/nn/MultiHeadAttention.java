package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Multi-head self-attention with a fused QKV projection and optional KV caching: {@code x -> qkv ->
 * split -> per-head RoPE -> (optional cache append) -> causal-or-not SDPA -> merge heads -> output
 * projection}. See req/phase4-plan.md §7.
 *
 * <p>Not a {@link UnaryModule}: {@link #forward} takes a {@link KVCache} and a causal flag in
 * addition to {@code x} -- {@code UnaryModule}'s own javadoc names this class as the reason that
 * interface is not folded into {@link Module} itself.
 */
public final class MultiHeadAttention extends Module {

  private static final float ROPE_BASE = 10000f;

  private final int numHeads;
  private final int headDim;
  private final float scale;
  private final Linear qkvProj;
  private final Linear outProj;

  /**
   * Creates a layer with {@code numHeads} heads. {@code qkvWeight} is {@code [3*embedDim,
   * embedDim]} (the checkpoint layout for a fused QKV projection); {@code outWeight} is {@code
   * [embedDim, embedDim]}. {@code embedDim} is derived from {@code outWeight} and must be evenly
   * divisible by {@code numHeads}.
   */
  public MultiHeadAttention(
      MLXScope scope,
      int numHeads,
      MLXArray qkvWeight,
      MLXArray qkvBias,
      MLXArray outWeight,
      MLXArray outBias) {
    super(scope);
    if (outWeight.ndim() != 2 || outWeight.shape()[0] != outWeight.shape()[1]) {
      throw new IllegalArgumentException(
          "MultiHeadAttention: outWeight must be square [embedDim, embedDim], got shape "
              + Arrays.toString(outWeight.shape()));
    }
    int embedDim = outWeight.shape()[0];
    if (qkvWeight.ndim() != 2
        || qkvWeight.shape()[0] != 3 * embedDim
        || qkvWeight.shape()[1] != embedDim) {
      throw new IllegalArgumentException(
          "MultiHeadAttention: qkvWeight must be [3*embedDim, embedDim] = ["
              + (3 * embedDim)
              + ", "
              + embedDim
              + "], got shape "
              + Arrays.toString(qkvWeight.shape()));
    }
    if (numHeads <= 0 || embedDim % numHeads != 0) {
      throw new IllegalArgumentException(
          "MultiHeadAttention: numHeads ("
              + numHeads
              + ") must evenly divide embedDim ("
              + embedDim
              + ")");
    }
    this.numHeads = numHeads;
    this.headDim = embedDim / numHeads;
    this.scale = (float) (1.0 / Math.sqrt(headDim));
    qkvProj = child("qkvProj", new Linear(scope, qkvWeight, qkvBias));
    outProj = child("outProj", new Linear(scope, outWeight, outBias));
  }

  /**
   * Computes this layer's output for {@code x} (shape {@code [batch, seq, embedDim]}). If {@code
   * cache} is non-null, its accumulated keys/values (from earlier calls) are attended over too,
   * RoPE positions start at {@code cache.offset()}, and {@code cache} is advanced by this call's
   * {@code seq} positions -- {@code cache} may be {@code null} for a one-shot forward pass with no
   * carried-over state (RoPE positions then start at {@code 0}). {@code causal}, together with
   * {@code cache}, is what makes both a full-sequence prefill and a single-token decode step
   * correct with the same flag: mlx's causal mask bottom-aligns a shorter query against a longer
   * key/value (see {@link MLXFast#scaledDotProductAttention}'s javadoc).
   *
   * <p>{@code cache}'s own scope must be this layer's {@code scope()} or an ancestor of it -- see
   * {@link KVCache}'s javadoc; violating this throws from inside {@link KVCache#append}.
   */
  public MLXArray forward(MLXArray x, KVCache cache, boolean causal) {
    int[] shape = x.shape();
    int batch = shape[0];
    int seq = shape[1];
    int offset = cache != null ? cache.offset() : 0;

    MLXArray qkv = qkvProj.forward(x); // [batch, seq, 3*embedDim]
    MLXArray[] parts = MLXShape.split(qkv, 3, 2);
    MLXArray q = toHeads(parts[0], batch, seq);
    MLXArray k = toHeads(parts[1], batch, seq);
    MLXArray v = toHeads(parts[2], batch, seq);

    q = MLXFast.rope(q, headDim, false, ROPE_BASE, 1.0f, offset, null);
    k = MLXFast.rope(k, headDim, false, ROPE_BASE, 1.0f, offset, null);

    if (cache != null) {
      cache.append(k, v);
      k = cache.keys();
      v = cache.values();
    }

    MLXArray attn = MLXFast.scaledDotProductAttention(q, k, v, scale, causal, null, null);
    MLXArray merged = MLXShape.flatten(MLXShape.transpose(attn, new int[] {0, 2, 1, 3}), 2, 3);
    return outProj.forward(merged);
  }

  /**
   * {@code [batch, seq, embedDim] -> [batch, seq, numHeads, headDim] -> [batch, numHeads, seq,
   * headDim]}.
   */
  private MLXArray toHeads(MLXArray part, int batch, int seq) {
    MLXArray reshaped = MLXShape.reshape(part, new int[] {batch, seq, numHeads, headDim});
    return MLXShape.transpose(reshaped, new int[] {0, 2, 1, 3});
  }
}
