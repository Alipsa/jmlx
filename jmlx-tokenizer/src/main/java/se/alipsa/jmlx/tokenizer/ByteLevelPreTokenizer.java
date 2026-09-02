package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;

/** Splits normalized text into byte-level-encoded pre-token chunks via the model's Split regex. */
public final class ByteLevelPreTokenizer {

  private final PreTokenizerConfig config;

  /**
   * Wraps the pre-tokenizer's split-regex and add-prefix-space configuration.
   *
   * @param config pre-tokenizer configuration
   */
  public ByteLevelPreTokenizer(PreTokenizerConfig config) {
    this.config = Objects.requireNonNull(config, "ByteLevelPreTokenizer: config must not be null");
  }

  /**
   * Splits {@code text} into byte-level-encoded chunks, in order. {@code add_prefix_space} is
   * applied per resulting piece (HF's {@code ByteLevel} step runs after {@code Split}, over each
   * already-split piece independently), not once to the whole input before matching -- prefixing
   * the whole input first would let a synthetic leading space change how the *first* match is
   * delimited, and would leave every later match's own leading space unadded.
   *
   * <p>A span the regex doesn't match is emitted as its own chunk rather than dropped: HF's {@code
   * find_matches} contract requires the whole string be covered by contiguous, ordered slices, with
   * non-matching spans kept (not discarded) under {@code SplitDelimiterBehavior::Isolated} -- so a
   * file whose regex doesn't cover every character still round-trips through HF, and this port must
   * match that instead of hard-failing where HF wouldn't. An unmatched span that turns out to be a
   * genuinely unrepresentable symbol still surfaces loudly, via {@link BpeMerger#merge}'s existing
   * no-vocabulary-entry check.
   *
   * @param text normalized input text
   * @return byte-level-encoded chunks
   */
  public List<String> split(String text) {
    Objects.requireNonNull(text, "ByteLevelPreTokenizer.split: text must not be null");
    List<String> chunks = new ArrayList<>();
    Matcher matcher = config.splitPattern().matcher(text);
    int lastEnd = 0;
    while (matcher.find()) {
      if (matcher.start() > lastEnd) {
        chunks.add(encodePiece(text.substring(lastEnd, matcher.start())));
      }
      // A zero-width match (only reachable with a regex containing a zero-width alternative --
      // neither shipped Split pattern has one) would otherwise emit an empty chunk here (PR #14
      // review round 3, finding 9); BpeMerger.merge("") already tolerates one harmlessly, but
      // skipping it is simpler than relying on that.
      if (!matcher.group().isEmpty()) {
        chunks.add(encodePiece(matcher.group()));
      }
      lastEnd = matcher.end();
    }
    if (lastEnd < text.length()) {
      chunks.add(encodePiece(text.substring(lastEnd)));
    }
    return chunks;
  }

  private String encodePiece(String piece) {
    if (config.addPrefixSpace() && !piece.isEmpty() && piece.charAt(0) != ' ') {
      piece = " " + piece;
    }
    return ByteLevelCoding.encode(piece);
  }
}
