package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Mask/bias-aware attention without RoPE, supporting rectangular head projections. */
public final class BidirectionalAttention extends Module {
  private final int heads;
  private final int headDim;
  private final float scale;
  private final UnaryLayer query;
  private final UnaryLayer key;
  private final UnaryLayer value;
  private final UnaryLayer output;

  /** Creates registered projections; masks are BOOL and bias is additive in score space. */
  public BidirectionalAttention(
      MLXScope scope,
      int heads,
      int headDim,
      float scale,
      UnaryLayer query,
      UnaryLayer key,
      UnaryLayer value,
      UnaryLayer output) {
    super(scope);
    if (heads <= 0 || headDim <= 0 || !Float.isFinite(scale) || scale <= 0) {
      throw new IllegalArgumentException(
          "attention heads, headDim and finite scale must be positive");
    }
    this.heads = heads;
    this.headDim = headDim;
    this.scale = scale;
    this.query = child("query", Objects.requireNonNull(query));
    this.key = child("key", Objects.requireNonNull(key));
    this.value = child("value", Objects.requireNonNull(value));
    this.output = child("output", Objects.requireNonNull(output));
  }

  /** Self attention; an optional FULL cache appends projected keys and values. */
  public MLXArray forward(MLXArray input, MLXArray mask, MLXArray bias, KVCache cache) {
    MLXArray q = split(query.forward(input));
    MLXArray k = split(key.forward(input));
    MLXArray v = split(value.forward(input));
    if (cache != null) {
      if (cache.policy().mode() != KVCachePolicy.Mode.FULL) {
        throw new IllegalArgumentException("no-RoPE attention requires a FULL cache");
      }
      cache.append(k, v);
      k = cache.keys();
      v = cache.values();
    }
    return attend(q, k, v, mask, bias);
  }

  /** Cross attention using a previously initialized request cache. */
  public MLXArray forward(MLXArray input, StaticKVCache cache, MLXArray mask) {
    return forward(input, cache, mask, null);
  }

  /** Cross attention with an optional additive score bias. */
  public MLXArray forward(MLXArray input, StaticKVCache cache, MLXArray mask, MLXArray bias) {
    return attend(split(query.forward(input)), cache.keys(), cache.values(), mask, bias);
  }

  /** Projects source keys/values exactly once into the request-owned cache. */
  public void initialize(MLXArray source, StaticKVCache cache) {
    if (cache.initialized()) {
      throw new IllegalStateException("static K/V cache is already initialized");
    }
    cache.initialize(split(key.forward(source)), split(value.forward(source)));
  }

  private MLXArray split(MLXArray projected) {
    int[] shape = projected.shape();
    if (shape.length != 3 || shape[2] != Math.multiplyExact(heads, headDim)) {
      throw new IllegalArgumentException("projection must be [batch,length,heads*headDim]");
    }
    return MLXShape.transpose(
        MLXShape.reshape(projected, new int[] {shape[0], shape[1], heads, headDim}),
        new int[] {0, 2, 1, 3});
  }

  /**
   * The largest negative finite value the scores' dtype can hold: {@code -Float.MAX_VALUE}
   * overflows float16 (max finite 65504) and even rounds to -inf in bfloat16, where softmax would
   * then see NaN on an all-masked row instead of the intended uniform 1/n degenerate result.
   */
  private static float maskFill(DType dtype) {
    return switch (dtype) {
      case FLOAT16 -> -65504f;
      case BFLOAT16 -> -3.3895313892515355e38f;
      default -> -Float.MAX_VALUE;
    };
  }

  private MLXArray attend(MLXArray q, MLXArray k, MLXArray v, MLXArray mask, MLXArray bias) {
    if (mask == null || mask.dtype() != DType.BOOL) {
      throw new IllegalArgumentException("attention requires an explicit BOOL mask");
    }
    // Constants take the scores' dtype (float32 only for non-inexact inputs, as before) so
    // half-precision q/k/v stay half precision through scores, softmax and output.
    DType dtype = q.dtype().isInexact() ? q.dtype() : DType.FLOAT32;
    // Explicit composition preserves both padding and relative bias on every native path.
    MLXArray scores = MLXOps.matmul(q, MLXShape.transpose(k, new int[] {0, 1, 3, 2}));
    scores = MLXOps.multiply(scores, MLX.full(q.scope(), new int[] {1}, scale, dtype));
    if (bias != null) {
      scores = MLXOps.add(scores, bias);
    }
    scores = MLXOps.where(mask, scores, MLX.full(q.scope(), new int[] {1}, maskFill(dtype), dtype));
    MLXArray attended = MLXOps.matmul(MLXOps.softmaxAxis(scores, -1, true), v);
    MLXArray merged = MLXShape.flatten(MLXShape.transpose(attended, new int[] {0, 2, 1, 3}), 2, 3);
    return output.forward(merged);
  }
}
