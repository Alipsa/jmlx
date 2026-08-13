package se.alipsa.jmlx.core;

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
}
