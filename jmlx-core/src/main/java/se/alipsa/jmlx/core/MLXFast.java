package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Home for the {@code fast.h} op family: {@code rmsNorm}, {@code layerNorm}, {@code rope} and
 * {@code scaledDotProductAttention} -- the mlx-c group carrying nullable-array, by-value-optional-struct and
 * {@code const char*} parameters simultaneously (req/phase4-plan.md, Research findings). Empty until M1 ({@code
 * rmsNorm}/{@code layerNorm}) and M3 ({@code rope}/SDPA) land those ops; created now, during M0a's pure-motion facade
 * split, so later merge points add ops to an address that already exists rather than growing {@link MLX} past the point
 * §1 named as its split trigger.
 */
public final class MLXFast {

  private MLXFast() {}

  /**
   * Root Mean Square normalization ({@code mlx_fast_rms_norm}). {@code weight} is a Java {@code null} for an unweighted
   * norm -- a legitimate call, not an error: {@link NativeOps#nullableHandle} turns it into the zero-{@code
   * ctx} struct mlx-c's by-value-optional {@code weight} parameter expects, and {@link NativeOps#scopeOf} resolves the
   * result's scope across both operands (including the nullable one) rather than assuming {@code x}'s scope alone.
   */
  public static MLXArray rmsNorm(MLXArray x, MLXArray weight, float eps) {
    MLXScope scope = NativeOps.scopeOf("rmsNorm", x, weight);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment weightHandle = NativeOps.nullableHandle(weight, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked("rmsNorm",
          () -> mlx_h.mlx_fast_rms_norm(res, x.handle(), weightHandle, eps, NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  /**
   * Layer normalization ({@code mlx_fast_layer_norm}). {@code weight}/{@code bias} are each a Java {@code null} for an
   * unweighted/unbiased norm -- a legitimate call, not an error: see {@link #rmsNorm} for why {@code null} maps to a
   * zero-{@code ctx} struct rather than throwing, and why the result's scope is resolved via {@link NativeOps#scopeOf}
   * across all three operands.
   */
  public static MLXArray layerNorm(MLXArray x, MLXArray weight, MLXArray bias, float eps) {
    MLXScope scope = NativeOps.scopeOf("layerNorm", x, weight, bias);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment weightHandle = NativeOps.nullableHandle(weight, tmp);
      MemorySegment biasHandle = NativeOps.nullableHandle(bias, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked("layerNorm",
          () -> mlx_h.mlx_fast_layer_norm(res, x.handle(), weightHandle, biasHandle, eps, NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }
}
