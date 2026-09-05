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
    List<Segment> result = new ArrayList<>();
    int last = 0;
    int index = 0;
    while (index < text.length()) {
      Match match = bestMatch(text, index, candidates, normalizer);
      if (match == null) {
        index += Character.charCount(text.codePointAt(index));
        continue;
      }
      int consumeStart = match.token().leftStrip() ? precedingWhitespace(text, index) : index;
      int contentEnd = index + match.content().length();
      int consumeEnd =
          match.token().rightStrip() ? followingWhitespace(text, contentEnd) : contentEnd;
      if (consumeStart > last) {
        result.add(new Segment(slice(input, last, consumeStart), null));
      }
      result.add(new Segment(slice(input, consumeStart, consumeEnd), match.token()));
      last = consumeEnd;
      index = consumeEnd;
    }
    if (last < text.length()) {
      result.add(new Segment(slice(input, last, text.length()), null));
    }
    return result;
  }

  private static Match bestMatch(
      String text, int index, List<AddedToken> candidates, JsonNode normalizer) {
    Match best = null;
    for (AddedToken token : candidates) {
      String content =
          token.normalized()
              ? NormalizerPipeline.apply(normalizer, AlignedText.original(token.content())).text()
              : token.content();
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

  private static AlignedText slice(AlignedText input, int startChar, int endChar) {
    List<AlignedText.Unit> result = new ArrayList<>();
    int index = 0;
    for (AlignedText.Unit unit : input.units()) {
      int next = index + unit.value().length();
      if (next > startChar && index < endChar) {
        result.add(unit);
      }
      index = next;
    }
    return new AlignedText(result);
  }

  record Segment(AlignedText text, AddedToken token) {}

  private record Match(AddedToken token, String content) {}
}
