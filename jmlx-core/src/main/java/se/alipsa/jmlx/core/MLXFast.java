package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Home for the {@code fast.h} op family: {@code rmsNorm}, {@code layerNorm}, {@code rope} and
 * {@code scaledDotProductAttention} -- the mlx-c group carrying nullable-array,
 * by-value-optional-struct and {@code const char*} parameters simultaneously (req/phase4-plan.md,
 * Research findings). Empty until M1 ({@code rmsNorm}/{@code layerNorm}) and M3 ({@code rope}/SDPA)
 * land those ops; created now, during M0a's pure-motion facade split, so later merge points add ops
 * to an address that already exists rather than growing {@link MLX} past the point §1 named as its
 * split trigger.
 */
public final class MLXFast {

  private MLXFast() {}

  /**
   * Root Mean Square normalization ({@code mlx_fast_rms_norm}). {@code weight} is a Java {@code
   * null} for an unweighted norm -- a legitimate call, not an error: {@link
   * NativeOps#nullableHandle} turns it into the zero-{@code ctx} struct mlx-c's by-value-optional
   * {@code weight} parameter expects, and {@link NativeOps#scopeOf} resolves the result's scope
   * across both operands (including the nullable one) rather than assuming {@code x}'s scope alone.
   */
  public static MLXArray rmsNorm(MLXArray x, MLXArray weight, float eps) {
    MLXScope scope = NativeOps.scopeOf("rmsNorm", x, weight);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment weightHandle = NativeOps.nullableHandle(weight, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "rmsNorm",
          () ->
              mlx_h.mlx_fast_rms_norm(
                  res, x.handle(), weightHandle, eps, NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  /**
   * Layer normalization ({@code mlx_fast_layer_norm}). {@code weight}/{@code bias} are each a Java
   * {@code null} for an unweighted/unbiased norm -- a legitimate call, not an error: see {@link
   * #rmsNorm} for why {@code null} maps to a zero-{@code ctx} struct rather than throwing, and why
   * the result's scope is resolved via {@link NativeOps#scopeOf} across all three operands.
   */
  public static MLXArray layerNorm(MLXArray x, MLXArray weight, MLXArray bias, float eps) {
    MLXScope scope = NativeOps.scopeOf("layerNorm", x, weight, bias);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment weightHandle = NativeOps.nullableHandle(weight, tmp);
      MemorySegment biasHandle = NativeOps.nullableHandle(bias, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "layerNorm",
          () ->
              mlx_h.mlx_fast_layer_norm(
                  res, x.handle(), weightHandle, biasHandle, eps, NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  /**
   * Rotary position embedding ({@code mlx_fast_rope}). {@code dims} is the (even) width of the head
   * dimension to rotate -- for {@code traditional=false} (the half-split/NeoX form this facade
   * exposes), pairs are {@code (x_i, x_{i+dims/2})} for {@code i} in {@code [0, dims/2)}, each
   * rotated by {@code angle_i = offset * scale * base^(-2i/dims)} (confirmed empirically against
   * this mlx-c version; req/plans/phase4-m3-plan.md's Findings section). {@code x}'s second-to-last
   * axis is the position axis: {@code offset} is the position of {@code x}'s first row along that
   * axis, so a single call over a {@code [..., T, dims]} tensor rotates row {@code t} by position
   * {@code offset+t} -- this is what lets a KV-cache decode step pass just the new token's own row
   * with {@code offset = cache.offset()} rather than needing per-position calls.
   *
   * <p>{@code base} and {@code freqs} are each a Java {@code null} for "absent" -- but mlx-c
   * requires at least one of the two to be present, and throws a clean {@link MLXException} naming
   * that ("Neither base nor freqs has a value") if both are {@code null}; this method does not
   * duplicate that check.
   */
  public static MLXArray rope(
      MLXArray x,
      int dims,
      boolean traditional,
      Float base,
      float scale,
      int offset,
      MLXArray freqs) {
    MLXScope scope = NativeOps.scopeOf("rope", x, freqs);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment baseStruct = NativeOps.optFloat(tmp, base);
      MemorySegment freqsHandle = NativeOps.nullableHandle(freqs, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "rope",
          () ->
              mlx_h.mlx_fast_rope(
                  res,
                  x.handle(),
                  dims,
                  traditional,
                  baseStruct,
                  scale,
                  offset,
                  freqsHandle,
                  NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  private static final MemorySegment MASK_MODE_NONE = NativeOps.cstr("");
  private static final MemorySegment MASK_MODE_CAUSAL = NativeOps.cstr("causal");
  private static final MemorySegment MASK_MODE_ARRAY = NativeOps.cstr("array");

  /**
   * Scaled dot-product attention ({@code mlx_fast_scaled_dot_product_attention}): {@code
   * softmax(scale * queries @ keys^T + mask) @ values}, batched over every leading axis. {@code
   * causal} and {@code maskArr} are mutually exclusive Java-side conveniences over mlx-c's single
   * {@code mask_mode} string -- {@code causal=true} sends {@code "causal"}; a non-null {@code
   * maskArr} sends {@code "array"}; neither sends {@code ""} (unmasked).
   *
   * <p>{@code maskArr}, if given, is either {@code BOOL} ({@code true} = attend, {@code false} =
   * mask out) or an inexact dtype added directly to the scaled scores (a large negative value, e.g.
   * {@code -1e9}, masks a position out) -- both confirmed empirically to reproduce {@code causal}'s
   * own result exactly when built to encode the same constraint (req/plans/phase4-m3-plan.md's
   * Findings section).
   *
   * <p>{@code causal} bottom-aligns when {@code queries} has fewer rows than {@code keys}/{@code
   * values}: row {@code i} of {@code queries} is treated as absolute position {@code
   * keys.shape()[-2] - queries.shape()[-2] + i}, not position {@code i} -- this is what makes a
   * single new token's decode-step attention over a longer KV cache correct without a custom mask
   * (confirmed empirically, same Findings section).
   *
   * <p>{@code sinks} is passed straight through to {@code mlx_fast_scaled_dot_product_attention}'s
   * own {@code sinks} parameter (verified against the mlx-c header's parameter order) but is not
   * yet exercised with a non-null value by any test in this codebase.
   *
   * @throws IllegalArgumentException if both {@code causal} and a non-null {@code maskArr} are
   *     given
   */
  public static MLXArray scaledDotProductAttention(
      MLXArray queries,
      MLXArray keys,
      MLXArray values,
      float scale,
      boolean causal,
      MLXArray maskArr,
      MLXArray sinks) {
    if (causal && maskArr != null) {
      throw new IllegalArgumentException(
          "scaledDotProductAttention: causal and an explicit maskArr are mutually exclusive");
    }
    MLXScope scope =
        NativeOps.scopeOf("scaledDotProductAttention", queries, keys, values, maskArr, sinks);
    MemorySegment maskMode =
        causal ? MASK_MODE_CAUSAL : (maskArr != null ? MASK_MODE_ARRAY : MASK_MODE_NONE);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment maskHandle = NativeOps.nullableHandle(maskArr, tmp);
      MemorySegment sinksHandle = NativeOps.nullableHandle(sinks, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "scaledDotProductAttention",
          () ->
              mlx_h.mlx_fast_scaled_dot_product_attention(
                  res,
                  queries.handle(),
                  keys.handle(),
                  values.handle(),
                  scale,
                  maskMode,
                  maskHandle,
                  sinksHandle,
                  NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }
}
