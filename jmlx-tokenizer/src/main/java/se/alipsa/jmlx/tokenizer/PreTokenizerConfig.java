package se.alipsa.jmlx.tokenizer;

import java.util.Objects;
import java.util.regex.Pattern;

/**
 * {@code tokenizer.json}'s {@code pre_tokenizer}, scoped to the {@code Split}+{@code ByteLevel}
 * shape both target models use.
 *
 * @param splitPattern pattern that isolates input pieces
 * @param addPrefixSpace whether each non-empty piece gains a leading space
 */
public record PreTokenizerConfig(Pattern splitPattern, boolean addPrefixSpace) {
  /** Requires a compiled split pattern. */
  public PreTokenizerConfig {
    Objects.requireNonNull(splitPattern, "PreTokenizerConfig: splitPattern must not be null");
  }
}
