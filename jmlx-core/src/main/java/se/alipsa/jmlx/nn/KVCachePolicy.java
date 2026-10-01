package se.alipsa.jmlx.nn;

/** Resolved retention policy for one decoder layer's key/value cache. */
public record KVCachePolicy(Mode mode, int limit) {
  /** Cache retention modes. */
  public enum Mode {
    /** Retain every key, subject to an optional capacity. */
    FULL,
    /** Retain at most {@code window - 1} keys between attention calls. */
    SLIDING_WINDOW
  }

  /** Validates a resolved policy. Zero means unbounded only for {@link Mode#FULL}. */
  public KVCachePolicy {
    if (mode == null || limit < 0 || (mode == Mode.SLIDING_WINDOW && limit == 0)) {
      throw new IllegalArgumentException("invalid KV cache mode or limit");
    }
  }

  /** Unbounded full-context retention. */
  public static KVCachePolicy full() {
    return new KVCachePolicy(Mode.FULL, 0);
  }

  /** Full-context retention with a maximum absolute position. */
  public static KVCachePolicy full(int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("FULL capacity must be positive");
    }
    return new KVCachePolicy(Mode.FULL, capacity);
  }

  /** Sliding retention with the architecture's resolved numerical window. */
  public static KVCachePolicy slidingWindow(int window) {
    return new KVCachePolicy(Mode.SLIDING_WINDOW, window);
  }

  /**
   * Named unsupported capability. The pinned runtime's SDPA rejects packed K/V; unpacking the whole
   * cache before every attention call raises decode peak memory above the float cache. Consequently
   * no quantized retention policy is constructed.
   *
   * @throws UnsupportedOperationException always
   */
  public static KVCachePolicy quantized(int bits, int groupSize) {
    throw new UnsupportedOperationException(
        "quantized KV retention is unsupported: pinned MLX SDPA requires float K/V; "
            + "full-cache dequantization raises decode peak memory");
  }

  /**
   * Rejects a position or padded width that overflows {@code int} or exceeds a bounded {@link
   * Mode#FULL} capacity. Sliding policies evict instead of failing, so only the overflow applies.
   *
   * @throws IllegalArgumentException naming {@code subject} when {@code count} is not retainable
   */
  public void requireCapacity(long count, String subject) {
    if (count > Integer.MAX_VALUE || (mode == Mode.FULL && limit > 0 && count > limit)) {
      throw new IllegalArgumentException(subject + " exceeds capacity");
    }
  }

  /** Returns whether this policy may discard previously attended keys. */
  public boolean evicts() {
    return mode == Mode.SLIDING_WINDOW;
  }
}
