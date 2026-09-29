package se.alipsa.jmlx.tokenizer;

import java.util.Objects;

/**
 * Single-sequence truncation policy. A maximum length of zero disables truncation.
 *
 * @param maxLength maximum result length including reserved special tokens
 * @param direction side from which excess tokens are removed
 */
public record Truncation(int maxLength, Direction direction) {
  /** Validates a non-negative maximum length. */
  public Truncation {
    if (maxLength < 0) {
      throw new IllegalArgumentException("Truncation.maxLength must not be negative");
    }
    direction = Objects.requireNonNull(direction, "direction");
  }

  /**
   * Returns disabled truncation.
   *
   * @return disabled policy
   */
  public static Truncation disabled() {
    return new Truncation(0, Direction.RIGHT);
  }

  /**
   * Whether truncation is enabled.
   *
   * @return true when a positive maximum is configured
   */
  public boolean enabled() {
    return maxLength > 0;
  }
}
