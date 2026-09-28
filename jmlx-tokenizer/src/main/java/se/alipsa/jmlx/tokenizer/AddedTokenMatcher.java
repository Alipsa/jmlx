package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import tools.jackson.databind.JsonNode;

/** Longest-first added-token matching with stripping and word-boundary behavior. */
final class AddedTokenMatcher {

  private AddedTokenMatcher() {}

  static List<Segment> split(
      AlignedText input, List<AddedToken> tokens, boolean normalized, JsonNode normalizer) {
    List<AddedToken> candidates =
        tokens.stream().filter(token -> token.normalized() == normalized).toList();
    if (candidates.isEmpty() || input.units().isEmpty()) {
      return List.of(new Segment(input, null));
    }
    String text = input.text();
    int[] unitAtChar = unitAtChar(input);
    // Normalizing every candidate's content once per call, rather than once per (position,
    // candidate) pair inside bestMatch's loop, avoids re-running the full normalizer pipeline at
    // every character position -- it does not depend on index (PR #24 review, finding 6).
    List<String> candidateContents = normalizedContents(candidates, normalizer);
    List<Segment> result = new ArrayList<>();
    int last = 0;
    int index = 0;
    while (index < text.length()) {
      Match match = bestMatch(text, index, candidates, candidateContents);
      if (match == null) {
        index += Character.charCount(text.codePointAt(index));
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

  private static List<String> normalizedContents(List<AddedToken> candidates, JsonNode normalizer) {
    List<String> result = new ArrayList<>(candidates.size());
    for (AddedToken token : candidates) {
      result.add(
          token.normalized()
              ? NormalizerPipeline.apply(normalizer, AlignedText.original(token.content())).text()
              : token.content());
    }
    return result;
  }

  private static Match bestMatch(
      String text, int index, List<AddedToken> candidates, List<String> candidateContents) {
    Match best = null;
    for (int i = 0; i < candidates.size(); i++) {
      AddedToken token = candidates.get(i);
      String content = candidateContents.get(i);
      if (!content.isEmpty()
          && text.startsWith(content, index)
          && (!token.singleWord() || isWholeWord(text, index, index + content.length()))
          && (best == null || content.length() > best.content().length())) {
        best = new Match(token, content);
      }
    }
    return best;
  }

  private static boolean isWholeWord(String text, int start, int end) {
    boolean left = start == 0 || !isWord(text.codePointBefore(start));
    boolean right = end == text.length() || !isWord(text.codePointAt(end));
    return left && right;
  }

  private static boolean isWord(int codePoint) {
    return Character.isLetterOrDigit(codePoint) || codePoint == '_';
  }

  private static int precedingWhitespace(String text, int index) {
    int result = index;
    while (result > 0) {
      int cp = text.codePointBefore(result);
      if (!Character.isWhitespace(cp)) {
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
      if (!Character.isWhitespace(cp)) {
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
   * match found within that segment (PR #24 review, finding 5; mirrors the same fix in {@link
   * PreTokenizerPipeline}).
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

  private record Match(AddedToken token, String content) {}
}
