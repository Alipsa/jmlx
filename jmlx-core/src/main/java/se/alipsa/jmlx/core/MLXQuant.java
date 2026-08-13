package se.alipsa.jmlx.core;

/**
 * Home for {@code quantize}, {@code dequantize} and {@code quantizedMatmul} (req/phase4-plan.md
 * §8). Empty until M4; created now, during M0a's pure-motion facade split, so M4 adds ops to an
 * address that already exists rather than growing {@link MLX} past the point §1 named as its split
 * trigger.
 */
public final class MLXQuant {

  private MLXQuant() {}
}
