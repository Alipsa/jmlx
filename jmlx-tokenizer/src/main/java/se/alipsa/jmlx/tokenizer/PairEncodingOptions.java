package se.alipsa.jmlx.tokenizer;

import java.util.Objects;

/**
 * Length, padding and special-token options plus a pair truncation strategy.
 *
 * @param options special-token insertion, length and padding policies
 * @param strategy sequence selection when truncation removes tokens
 */
public record PairEncodingOptions(EncodingOptions options, PairTruncationStrategy strategy) {
  /** Requires non-null options and strategy. */
  public PairEncodingOptions {
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(strategy, "strategy");
  }
}
