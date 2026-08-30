package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;

/** Splits normalized text into byte-level-encoded pre-token chunks via the model's Split regex. */
public final class ByteLevelPreTokenizer {

  private final PreTokenizerConfig config;

  /** Wraps the pre-tokenizer's split-regex and add-prefix-space configuration. */
  public ByteLevelPreTokenizer(PreTokenizerConfig config) {
    this.config = Objects.requireNonNull(config, "ByteLevelPreTokenizer: config must not be null");
  }

  /** Splits {@code text} into byte-level-encoded chunks, one per regex match, in order. */
  public List<String> split(String text) {
    Objects.requireNonNull(text, "ByteLevelPreTokenizer.split: text must not be null");
    String input =
        config.addPrefixSpace() && !text.isEmpty() && text.charAt(0) != ' ' ? " " + text : text;
    List<String> chunks = new ArrayList<>();
    Matcher matcher = config.splitPattern().matcher(input);
    while (matcher.find()) {
      chunks.add(ByteLevelCoding.encode(matcher.group()));
    }
    return chunks;
  }
}
