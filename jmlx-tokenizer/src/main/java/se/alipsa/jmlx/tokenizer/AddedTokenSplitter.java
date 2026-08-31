package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Splits input text around literal added-token strings (longest-first, no lstrip/rstrip — see
 * Findings).
 */
public final class AddedTokenSplitter {

  /**
   * One segment of split input: either literal added-token text, or plain text needing full
   * tokenization.
   */
  public record Segment(String text, boolean isAddedToken) {}

  private final Pattern addedTokenPattern;

  /** Builds a longest-first alternation regex from the added tokens' literal content. */
  public AddedTokenSplitter(List<AddedToken> addedTokens) {
    Objects.requireNonNull(addedTokens, "AddedTokenSplitter: addedTokens must not be null");
    if (addedTokens.isEmpty()) {
      this.addedTokenPattern = null;
      return;
    }
    String alternation =
        addedTokens.stream()
            .map(AddedToken::content)
            .sorted(Comparator.comparingInt(String::length).reversed())
            .map(Pattern::quote)
            .collect(Collectors.joining("|"));
    this.addedTokenPattern = Pattern.compile(alternation);
  }

  /**
   * Splits {@code text} into ordered segments, tagging which ones are literal added-token strings.
   */
  public List<Segment> split(String text) {
    Objects.requireNonNull(text, "AddedTokenSplitter.split: text must not be null");
    List<Segment> segments = new ArrayList<>();
    if (addedTokenPattern == null) {
      if (!text.isEmpty()) {
        segments.add(new Segment(text, false));
      }
      return segments;
    }
    Matcher matcher = addedTokenPattern.matcher(text);
    int last = 0;
    while (matcher.find()) {
      if (matcher.start() > last) {
        segments.add(new Segment(text.substring(last, matcher.start()), false));
      }
      segments.add(new Segment(matcher.group(), true));
      last = matcher.end();
    }
    if (last < text.length()) {
      segments.add(new Segment(text.substring(last), false));
    }
    return segments;
  }
}
