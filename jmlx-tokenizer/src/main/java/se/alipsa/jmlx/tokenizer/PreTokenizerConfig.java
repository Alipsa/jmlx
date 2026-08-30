package se.alipsa.jmlx.tokenizer;

import java.util.regex.Pattern;

/**
 * {@code tokenizer.json}'s {@code pre_tokenizer}, scoped to the {@code Split}+{@code ByteLevel}
 * shape both target models use.
 */
public record PreTokenizerConfig(Pattern splitPattern, boolean addPrefixSpace) {}
