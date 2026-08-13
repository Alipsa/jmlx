package se.alipsa.jmlx.core;

/**
 * Home for {@code seed}, {@code normal}, {@code uniform} and (if a caller ever needs explicit keys) {@code key}/
 * {@code split} (req/phase4-plan.md §1, §5). Empty until M1's weight initialization needs it; created now, during M0a's
 * pure-motion facade split, so M1 adds ops to an address that already exists rather than growing {@link MLX} past the
 * point §1 named as its split trigger.
 */
public final class MLXRandom {

  private MLXRandom() {}
}
