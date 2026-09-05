package se.alipsa.jmlx.tokenizer;

import java.util.Objects;

/**
 * Fixed-length single-sequence padding. A length of zero disables padding.
 *
 * @param length fixed result length
 * @param direction side on which padding is added
 * @param padId padding token ID
 * @param padToken padding token text
 * @param padTypeId type ID assigned to padding
 */
public record Padding(int length, Direction direction, int padId, String padToken, int padTypeId) {
  /** Validates the fixed-length policy. */
  public Padding {
    if (length < 0) {
      throw new IllegalArgumentException("Padding.length must not be negative");
    }
    direction = Objects.requireNonNull(direction, "direction");
    padToken = Objects.requireNonNull(padToken, "padToken");
    if (length > 0 && padToken.isEmpty()) {
      throw new IllegalArgumentException("Padding.padToken must not be empty when enabled");
    }
    if (padId < 0 || padTypeId < 0) {
      throw new IllegalArgumentException("Padding ids must not be negative");
    }
  }

  /**
   * Returns disabled padding.
   *
   * @return disabled policy
   */
  public static Padding disabled() {
    return new Padding(0, Direction.RIGHT, 0, "", 0);
  }

  /**
   * Whether padding is enabled.
   *
   * @return true when a positive fixed length is configured
   */
  public boolean enabled() {
    return length > 0;
  }
}
