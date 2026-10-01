package se.alipsa.jmlx.models;

import se.alipsa.jmlx.nn.KVCachePolicy;

/** Request-level cache policy, resolved against the loaded checkpoint before generation. */
public record GenerationCachePolicy(Mode mode, int limit) {
  /** Request retention modes. */
  public enum Mode {
    FULL,
    SLIDING_WINDOW,
    SLIDING_WINDOW_FROM_MODEL
  }

  /** Validates a request policy. */
  public GenerationCachePolicy {
    if (mode == null
        || limit < 0
        || (mode == Mode.SLIDING_WINDOW && limit == 0)
        || (mode == Mode.SLIDING_WINDOW_FROM_MODEL && limit != 0)) {
      throw new IllegalArgumentException("invalid generation cache policy");
    }
  }

  /** Unbounded full context, the default for every checkpoint. */
  public static GenerationCachePolicy full() {
    return new GenerationCachePolicy(Mode.FULL, 0);
  }

  /** Full context bounded by absolute position. */
  public static GenerationCachePolicy full(int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("FULL capacity must be positive");
    }
    return new GenerationCachePolicy(Mode.FULL, capacity);
  }

  /** Sliding retention with an explicit window matching the checkpoint. */
  public static GenerationCachePolicy slidingWindow(int window) {
    return new GenerationCachePolicy(Mode.SLIDING_WINDOW, window);
  }

  /** Sliding retention using the checkpoint's configured window. */
  public static GenerationCachePolicy slidingWindowFromModel() {
    return new GenerationCachePolicy(Mode.SLIDING_WINDOW_FROM_MODEL, 0);
  }

  /**
   * Named unsupported capability: the pinned runtime cannot attend directly over packed K/V. This
   * fails at request construction so generation never silently falls back to a float cache.
   *
   * @throws UnsupportedOperationException always
   */
  public static GenerationCachePolicy quantized(int bits, int groupSize) {
    throw new UnsupportedOperationException(
        "quantized KV retention is unsupported: pinned MLX SDPA requires float K/V; "
            + "full-cache dequantization raises decode peak memory");
  }

  KVCachePolicy resolve(Integer descriptorWindow) {
    if (mode == Mode.FULL) {
      return limit == 0 ? KVCachePolicy.full() : KVCachePolicy.full(limit);
    }
    if (descriptorWindow == null) {
      throw new IllegalArgumentException(mode + " requires a checkpoint sliding_window");
    }
    if (mode == Mode.SLIDING_WINDOW && limit != descriptorWindow) {
      throw new IllegalArgumentException("SLIDING_WINDOW must match checkpoint sliding_window");
    }
    return KVCachePolicy.slidingWindow(descriptorWindow);
  }
}
