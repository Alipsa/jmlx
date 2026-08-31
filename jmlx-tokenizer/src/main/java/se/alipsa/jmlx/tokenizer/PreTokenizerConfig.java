package se.alipsa.jmlx.tokenizer;

import java.util.Objects;
import java.util.regex.Pattern;

/**
 * {@code tokenizer.json}'s {@code pre_tokenizer}, scoped to the {@code Split}+{@code ByteLevel}
 * shape both target models use.
 */
public record PreTokenizerConfig(Pattern splitPattern, boolean addPrefixSpace) {
  /** Requires a compiled split pattern. */
  public PreTokenizerConfig {
    Objects.requireNonNull(splitPattern, "PreTokenizerConfig: splitPattern must not be null");
  }
}
