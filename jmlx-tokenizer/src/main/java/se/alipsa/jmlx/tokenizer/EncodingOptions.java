package se.alipsa.jmlx.tokenizer;

import java.util.Objects;

/**
 * Explicit single-sequence encode options.
 *
 * @param addSpecialTokens whether to apply post-processor special tokens
 * @param truncation truncation policy
 * @param padding padding policy
 */
public record EncodingOptions(boolean addSpecialTokens, Truncation truncation, Padding padding) {
  /** Validates non-null policies. */
  public EncodingOptions {
    truncation = Objects.requireNonNull(truncation, "truncation");
    padding = Objects.requireNonNull(padding, "padding");
    if (truncation.enabled() && padding.enabled() && padding.length() < truncation.maxLength()) {
      throw new IllegalArgumentException(
          "Padding length must not be smaller than truncation length");
    }
  }

  /**
   * Returns unbounded, unpadded options.
   *
   * @param addSpecialTokens whether to apply post-processor special tokens
   * @return compatibility options
   */
  public static EncodingOptions unbounded(boolean addSpecialTokens) {
    return new EncodingOptions(addSpecialTokens, Truncation.disabled(), Padding.disabled());
  }
}
