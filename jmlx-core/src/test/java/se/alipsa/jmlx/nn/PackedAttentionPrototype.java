package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Candidate A (packed {@code quantized_matmul} attention) prototype used by the 6.4.1 probe. */
final class PackedAttentionPrototype {
  private PackedAttentionPrototype() {}

  /** Packed tensor plus its affine scales and biases. */
  record Packed(MLXArray w, MLXArray scales, MLXArray biases) {
    MLXArray[] arrays() {
      return new MLXArray[] {w, scales, biases};
    }

    Packed concat(Packed next) {
      return new Packed(
          MLXShape.concatenate(new MLXArray[] {w, next.w}, 2),
          MLXShape.concatenate(new MLXArray[] {scales, next.scales}, 2),
          MLXShape.concatenate(new MLXArray[] {biases, next.biases}, 2));
    }
  }

  static int padded(int headDim, int group) {
    return (headDim + group - 1) / group * group;
  }

  /** Pads the head axis of {@code [B, kvHeads, T, D]} with zeros and quantizes it. */
  static Packed pack(MLXArray x, int group, int bits) {
    int[] shape = x.shape();
    int dpad = padded(shape[3], group);
    MLXArray padded = x;
    if (dpad != shape[3]) {
      MLXArray zeros =
          MLX.zeros(
              x.scope(), new int[] {shape[0], shape[1], shape[2], dpad - shape[3]}, x.dtype());
      padded = MLXShape.concatenate(new MLXArray[] {x, zeros}, 3);
    }
    MLXArray[] parts = MLXQuant.quantize(padded, group, bits, "affine", null);
    return new Packed(parts[0], parts[1], parts[2]);
  }

  /** BOOL {@code [queryLength, keyLength]} causal mask over absolute positions; true = attend. */
  static MLXArray causal(MLXScope scope, int queryStart, int queryLength, int keyLength) {
    MLXArray positions =
        MLXShape.reshape(
            MLX.arange(scope, queryStart, queryStart + queryLength, 1, DType.INT32),
            new int[] {queryLength, 1});
    MLXArray keys =
        MLXShape.reshape(MLX.arange(scope, 0, keyLength, 1, DType.INT32), new int[] {1, keyLength});
    return MLXOps.lessEqual(keys, positions);
  }

  static float lowest(DType dtype) {
    return switch (dtype) {
      case FLOAT16 -> -65504f;
      case BFLOAT16 -> -3.3895314e38f;
      default -> -Float.MAX_VALUE;
    };
  }

  /**
   * Attention with GQA folded into the query axis ({@code [B, kvHeads, queriesPerKv * L, D]}), so
   * the packed operands need no head broadcast. {@code mask} is BOOL, {@code [L, S]} or {@code [B,
   * 1, L, S]}, or null for no masking.
   */
  static MLXArray attend(
      MLXArray q,
      Packed k,
      Packed v,
      MLXArray mask,
      int kvHeads,
      int headDim,
      int group,
      int bits) {
    int[] qs = q.shape();
    int batch = qs[0];
    int heads = qs[1];
    int length = qs[2];
    int perKv = heads / kvHeads;
    int dpad = padded(headDim, group);
    MLXScope scope = q.scope();
    float scale = (float) (1.0 / Math.sqrt(headDim));
    MLXArray scaled = MLXOps.multiply(q, MLX.full(scope, new int[] {1}, scale, q.dtype()));
    if (dpad != headDim) {
      MLXArray zeros =
          MLX.zeros(scope, new int[] {batch, heads, length, dpad - headDim}, q.dtype());
      scaled = MLXShape.concatenate(new MLXArray[] {scaled, zeros}, 3);
    }
    MLXArray folded = MLXShape.reshape(scaled, new int[] {batch, kvHeads, perKv * length, dpad});
    MLXArray scores =
        MLXQuant.quantizedMatmul(folded, k.w, k.scales, k.biases, true, group, bits, "affine");
    int keyLength = scores.shape()[3];
    if (mask != null) {
      scores = MLXShape.reshape(scores, new int[] {batch, kvHeads, perKv, length, keyLength});
      MLXArray shaped =
          mask.shape().length == 4
              ? MLXShape.reshape(mask, new int[] {mask.shape()[0], 1, 1, length, keyLength})
              : mask;
      scores =
          MLXOps.where(
              shaped, scores, MLX.full(scope, new int[] {1}, lowest(q.dtype()), q.dtype()));
      scores = MLXShape.reshape(scores, new int[] {batch, kvHeads, perKv * length, keyLength});
    }
    MLXArray weights = MLXOps.softmaxAxis(scores, -1, true);
    MLXArray out =
        MLXQuant.quantizedMatmul(weights, v.w, v.scales, v.biases, false, group, bits, "affine");
    MLXArray unfolded = MLXShape.reshape(out, new int[] {batch, heads, length, dpad});
    return dpad == headDim
        ? unfolded
        : MLXShape.slice(
            unfolded, new int[] {0, 0, 0, 0}, new int[] {batch, heads, length, headDim});
  }

  /** Float reference: repeat K/V heads then the native fused SDPA. */
  static MLXArray floatAttend(
      MLXArray q, MLXArray k, MLXArray v, MLXArray mask, boolean causal, int kvHeads) {
    int perKv = q.shape()[1] / kvHeads;
    MLXScope scope = q.scope();
    return MLXFast.scaledDotProductAttention(
        q,
        AttentionHeads.repeatKeyValueHeads(k, scope, kvHeads, perKv),
        AttentionHeads.repeatKeyValueHeads(v, scope, kvHeads, perKv),
        (float) (1.0 / Math.sqrt(q.shape()[3])),
        causal,
        mask,
        null);
  }
}
