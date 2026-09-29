package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;

/** Longest-first added-token matching with stripping and word-boundary behavior. */
final class AddedTokenMatcher {

  private static final Pattern WORD =
      Pattern.compile("[" + PreTokenizerPipeline.WORD_CHARACTERS + "]");

  private final Map<Integer, List<Candidate>> byFirstCodePoint = new HashMap<>();
  private final boolean empty;

  /**
   * Prepares the candidates whose {@code normalized} flag equals {@code normalized}, bucketed by
   * first code point and ordered longest-first so a lookup only scans tokens that can match there.
   */
  AddedTokenMatcher(List<AddedToken> tokens, boolean normalized, JsonNode normalizer) {
    for (AddedToken token : tokens) {
      if (token.normalized() != normalized) {
        continue;
      }
      String content =
          token.normalized()
              ? NormalizerPipeline.apply(normalizer, AlignedText.original(token.content())).text()
              : token.content();
      if (!content.isEmpty()) {
        byFirstCodePoint
            .computeIfAbsent(content.codePointAt(0), key -> new ArrayList<>())
            .add(new Candidate(token, content));
      }
    }
    byFirstCodePoint
        .values()
        .forEach(list -> list.sort(Comparator.comparingInt(c -> -c.content().length())));
    empty = byFirstCodePoint.isEmpty();
  }

  List<Segment> split(AlignedText input) {
    if (empty || input.units().isEmpty()) {
      return List.of(new Segment(input, null));
    }
    String text = input.text();
    int[] unitAtChar = unitAtChar(input);
    List<Segment> result = new ArrayList<>();
    int last = 0;
    int index = 0;
    while (index < text.length()) {
      Candidate match = longestMatch(text, index);
      if (match == null) {
        index += Character.charCount(text.codePointAt(index));
        continue;
      }
      int matchEnd = index + match.content().length();
      if (match.token().singleWord() && !isWholeWord(text, index, matchEnd)) {
        index = matchEnd;
        continue;
      }
      int consumeStart =
          match.token().leftStrip() ? Math.max(last, precedingWhitespace(text, index)) : index;
      int contentEnd = index + match.content().length();
      int consumeEnd =
          match.token().rightStrip() ? followingWhitespace(text, contentEnd) : contentEnd;
      if (consumeStart > last) {
        result.add(new Segment(slice(input, unitAtChar, last, consumeStart), null));
      }
      result.add(new Segment(slice(input, unitAtChar, consumeStart, consumeEnd), match.token()));
      last = consumeEnd;
      index = consumeEnd;
    }
    if (last < text.length()) {
      result.add(new Segment(slice(input, unitAtChar, last, text.length()), null));
    }
    return result;
  }

  // Like HF's leftmost-longest find_iter: the longest match at a position wins outright. When
  // single_word then rejects it, the search resumes after that match's end, so neither a shorter
  // token at the same position nor one inside the rejected span can match.
  private Candidate longestMatch(String text, int index) {
    List<Candidate> bucket = byFirstCodePoint.get(text.codePointAt(index));
    if (bucket == null) {
      return null;
    }
    for (Candidate candidate : bucket) {
      if (text.startsWith(candidate.content(), index)) {
        return candidate;
      }
    }
    return null;
  }

  private static boolean isWholeWord(String text, int start, int end) {
    boolean left = start == 0 || !isWord(text.codePointBefore(start));
    boolean right = end == text.length() || !isWord(text.codePointAt(end));
    return left && right;
  }

  private static boolean isWord(int codePoint) {
    return WORD.matcher(Character.toString(codePoint)).matches();
  }

  private static int precedingWhitespace(String text, int index) {
    int result = index;
    while (result > 0) {
      int cp = text.codePointBefore(result);
      if (!UnicodeWhitespace.isWhitespace(cp)) {
        break;
      }
      result -= Character.charCount(cp);
    }
    return result;
  }

  private static int followingWhitespace(String text, int index) {
    int result = index;
    while (result < text.length()) {
      int cp = text.codePointAt(result);
      if (!UnicodeWhitespace.isWhitespace(cp)) {
        break;
      }
      result += Character.charCount(cp);
    }
    return result;
  }

  /**
   * Maps every char index in {@code input.text()} to the index of the unit it belongs to, so {@link
   * #slice} can locate a match's boundary units in O(1) instead of rescanning every unit of the
   * whole input per match -- {@code split} calls it once per segment and reuses it across every
   * match found within that segment.
   */
  private static int[] unitAtChar(AlignedText input) {
    String text = input.text();
    int[] result = new int[text.length() + 1];
    int charIndex = 0;
    List<AlignedText.Unit> units = input.units();
    for (int unit = 0; unit < units.size(); unit++) {
      String value = units.get(unit).value();
      for (int i = 0; i < value.length(); i++) {
        result[charIndex++] = unit;
      }
    }
    result[text.length()] = units.size();
    return result;
  }

  private static AlignedText slice(
      AlignedText input, int[] unitAtChar, int startChar, int endChar) {
    if (startChar >= endChar) {
      return new AlignedText(List.of());
    }
    int first = unitAtChar[startChar];
    int last = unitAtChar[endChar - 1];
    return new AlignedText(input.units().subList(first, last + 1));
  }

  record Segment(AlignedText text, AddedToken token) {}

  private record Candidate(AddedToken token, String content) {}
}
