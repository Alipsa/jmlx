package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import tools.jackson.databind.JsonNode;

/** Applies supported pre-tokenizer components to aligned text spans. */
final class PreTokenizerPipeline {

  // onig's \s is Unicode White_Space (matches U+00A0 NBSP, U+2007, etc.), not ASCII-only --
  // verified against the pinned oracle on NBSP runs of length 1-4, which showed the same
  // "merge in groups, treating a maximal *ASCII*-space-adjoining run specially" behavior as
  // \p{IsWhite_Space} and not plain \s (PR #24 review round 2, finding 3). Only \s/\S here needs
  // the explicit Unicode class: \p{N} is already Unicode-scoped by definition, and this regex has
  // no \d/\w/\b for onig's separate ASCII-only-\w semantics to matter.
  private static final Pattern BYTE_LEVEL_PATTERN =
      Pattern.compile(
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+"
              + "| ?[^\\p{IsWhite_Space}\\p{L}\\p{N}]+"
              + "|\\p{IsWhite_Space}+(?!\\P{IsWhite_Space})"
              + "|\\p{IsWhite_Space}+");
  // Unicode \w as Rust's regex defines it; shared with AddedTokenMatcher's single_word boundaries.
  static final String WORD_CHARACTERS = "\\p{IsAlphabetic}\\p{M}\\p{Nd}\\p{Pc}\\u200c\\u200d";
  private static final Pattern WHITESPACE_PATTERN =
      Pattern.compile("[" + WORD_CHARACTERS + "]+|[^" + WORD_CHARACTERS + "\\p{IsWhite_Space}]+");
  private static final Pattern WHITESPACE_SPLIT_PATTERN = Pattern.compile("[^\\p{IsWhite_Space}]+");

  private PreTokenizerPipeline() {}

  static List<AlignedText> apply(JsonNode config, AlignedText input) {
    if (config == null || config.isNull() || config.isMissingNode()) {
      return input.units().isEmpty() ? List.of() : List.of(input);
    }
    return applyOne(config, List.of(input));
  }

  private static List<AlignedText> applyOne(JsonNode config, List<AlignedText> inputs) {
    String type = config.path("type").asString();
    if ("Sequence".equals(type)) {
      List<AlignedText> result = inputs;
      for (JsonNode step : config.path("pretokenizers")) {
        result = applyOne(step, result);
      }
      return result;
    }
    List<AlignedText> result = new ArrayList<>();
    for (AlignedText input : inputs) {
      result.addAll(
          switch (type) {
            case "ByteLevel" -> byteLevel(config, input);
            case "Metaspace" -> metaspace(config, input);
            case "Whitespace" -> matches(input, WHITESPACE_PATTERN);
            case "WhitespaceSplit" -> whitespaceSplit(input);
            case "BertPreTokenizer" -> bert(input);
            case "Split" -> split(config, input);
            case "Digits" -> digits(config, input);
            default ->
                throw new TokenizerException(
                    "PreTokenizerPipeline: unsupported type '" + type + "'");
          });
    }
    return result;
  }

  private static List<AlignedText> byteLevel(JsonNode config, AlignedText input) {
    boolean prefix = config.path("add_prefix_space").asBoolean(false);
    boolean regex = config.path("use_regex").asBoolean(true);
    AlignedText value = input;
    if (prefix && !value.units().isEmpty() && !value.text().startsWith(" ")) {
      List<AlignedText.Unit> units = new ArrayList<>();
      AlignedText.Unit first = value.units().getFirst();
      units.add(new AlignedText.Unit(" ", first.startByte(), first.endByte()));
      units.addAll(value.units());
      value = new AlignedText(units);
    }
    List<AlignedText> chunks = regex ? matches(value, BYTE_LEVEL_PATTERN) : List.of(value);
    return chunks.stream().map(PreTokenizerPipeline::byteEncode).toList();
  }

  private static AlignedText byteEncode(AlignedText input) {
    List<AlignedText.Unit> result = new ArrayList<>();
    for (AlignedText.Unit unit : input.units()) {
      String encoded = ByteLevelCoding.encode(unit.value());
      encoded
          .codePoints()
          .forEach(
              cp ->
                  result.add(
                      new AlignedText.Unit(
                          new String(Character.toChars(cp)), unit.startByte(), unit.endByte())));
    }
    return new AlignedText(result);
  }

  private static List<AlignedText> metaspace(JsonNode config, AlignedText input) {
    String replacement = config.path("replacement").asString("▁");
    String scheme = config.path("prepend_scheme").asString("always").toLowerCase();
    boolean split = config.path("split").asBoolean(true);
    List<AlignedText.Unit> units = new ArrayList<>();
    boolean shouldPrepend =
        !input.units().isEmpty()
            && !input.text().startsWith(replacement)
            && !input.text().startsWith(" ")
            && ("always".equals(scheme)
                || ("first".equals(scheme) && input.units().getFirst().startByte() == 0));
    if (shouldPrepend) {
      AlignedText.Unit first = input.units().getFirst();
      replacement
          .codePoints()
          .forEach(
              cp ->
                  units.add(
                      new AlignedText.Unit(
                          new String(Character.toChars(cp)), first.startByte(), first.endByte())));
    }
    for (AlignedText.Unit unit : input.units()) {
      if (" ".equals(unit.value())) {
        replacement
            .codePoints()
            .forEach(
                cp ->
                    units.add(
                        new AlignedText.Unit(
                            new String(Character.toChars(cp)), unit.startByte(), unit.endByte())));
      } else {
        units.add(unit);
      }
    }
    AlignedText transformed = new AlignedText(units);
    if (!split) {
      return transformed.units().isEmpty() ? List.of() : List.of(transformed);
    }
    return splitMetaspace(transformed, replacement);
  }

  private static List<AlignedText> splitMetaspace(AlignedText input, String replacement) {
    List<AlignedText> result = new ArrayList<>();
    List<AlignedText.Unit> current = new ArrayList<>();
    for (AlignedText.Unit unit : input.units()) {
      if (unit.value().equals(replacement) && !current.isEmpty()) {
        result.add(new AlignedText(current));
        current = new ArrayList<>();
      }
      current.add(unit);
    }
    if (!current.isEmpty()) {
      result.add(new AlignedText(current));
    }
    return result;
  }

  private static List<AlignedText> whitespaceSplit(AlignedText input) {
    return matches(input, WHITESPACE_SPLIT_PATTERN);
  }

  private static List<AlignedText> digits(JsonNode config, AlignedText input) {
    boolean individual = config.path("individual_digits").asBoolean(false);
    List<AlignedText> result = new ArrayList<>();
    List<AlignedText.Unit> current = new ArrayList<>();
    boolean currentNumeric = false;
    for (AlignedText.Unit unit : input.units()) {
      int codePoint = unit.value().codePointAt(0);
      int category = Character.getType(codePoint);
      boolean numeric =
          category == Character.DECIMAL_DIGIT_NUMBER
              || category == Character.LETTER_NUMBER
              || category == Character.OTHER_NUMBER;
      if (!current.isEmpty() && (numeric != currentNumeric || (numeric && individual))) {
        flush(current, result);
        current = new ArrayList<>();
      }
      current.add(unit);
      currentNumeric = numeric;
    }
    flush(current, result);
    return result;
  }

  private static List<AlignedText> bert(AlignedText input) {
    List<AlignedText> result = new ArrayList<>();
    List<AlignedText.Unit> current = new ArrayList<>();
    for (AlignedText.Unit unit : input.units()) {
      int cp = unit.value().codePointAt(0);
      if (UnicodeWhitespace.isWhitespace(cp)) {
        flush(current, result);
        current = new ArrayList<>();
      } else if (isPunctuation(cp)) {
        flush(current, result);
        current = new ArrayList<>();
        result.add(new AlignedText(List.of(unit)));
      } else {
        current.add(unit);
      }
    }
    flush(current, result);
    return result;
  }

  private static boolean isPunctuation(int cp) {
    int type = Character.getType(cp);
    return (cp >= 33 && cp <= 47)
        || (cp >= 58 && cp <= 64)
        || (cp >= 91 && cp <= 96)
        || (cp >= 123 && cp <= 126)
        || type == Character.CONNECTOR_PUNCTUATION
        || type == Character.DASH_PUNCTUATION
        || type == Character.START_PUNCTUATION
        || type == Character.END_PUNCTUATION
        || type == Character.INITIAL_QUOTE_PUNCTUATION
        || type == Character.FINAL_QUOTE_PUNCTUATION
        || type == Character.OTHER_PUNCTUATION;
  }

  private static void flush(List<AlignedText.Unit> current, List<AlignedText> result) {
    if (!current.isEmpty()) {
      result.add(new AlignedText(current));
    }
  }

  private static List<AlignedText> split(JsonNode config, AlignedText input) {
    JsonNode patternNode = config.path("pattern");
    String expression;
    if (patternNode.has("Regex")) {
      expression = OnigRegex.translate(patternNode.path("Regex").asString());
    } else if (patternNode.has("String")) {
      expression = Pattern.quote(patternNode.path("String").asString());
    } else {
      throw new TokenizerException("PreTokenizerPipeline: Split.pattern is unsupported");
    }
    if (config.path("invert").asBoolean(false)) {
      throw new TokenizerException("PreTokenizerPipeline: Split.invert=true is unsupported");
    }
    String behavior = config.path("behavior").asString("Removed");
    try {
      return splitByBehavior(input, Pattern.compile(expression), behavior);
    } catch (PatternSyntaxException e) {
      throw new TokenizerException("PreTokenizerPipeline: invalid Split pattern", e);
    }
  }

  private static List<AlignedText> splitByBehavior(
      AlignedText input, Pattern pattern, String behavior) {
    String text = input.text();
    int[] unitAtChar = unitAtChar(input);
    Matcher matcher = pattern.matcher(text);
    List<AlignedText> result = new ArrayList<>();
    int last = 0;
    int previousMatchEnd = -1;
    while (matcher.find()) {
      if (matcher.start() > last) {
        result.add(slice(input, unitAtChar, last, matcher.start()));
      }
      if (!matcher.group().isEmpty() && !"Removed".equals(behavior)) {
        if ("Contiguous".equals(behavior)
            && matcher.start() == previousMatchEnd
            && !result.isEmpty()) {
          AlignedText previous = result.removeLast();
          List<AlignedText.Unit> merged = new ArrayList<>(previous.units());
          merged.addAll(slice(input, unitAtChar, matcher.start(), matcher.end()).units());
          result.add(new AlignedText(merged));
        } else if ("Isolated".equals(behavior) || "Contiguous".equals(behavior)) {
          result.add(slice(input, unitAtChar, matcher.start(), matcher.end()));
        } else if ("MergedWithPrevious".equals(behavior)
            && matcher.start() != previousMatchEnd
            && !result.isEmpty()) {
          AlignedText previous = result.removeLast();
          List<AlignedText.Unit> merged = new ArrayList<>(previous.units());
          merged.addAll(slice(input, unitAtChar, matcher.start(), matcher.end()).units());
          result.add(new AlignedText(merged));
        } else if ("MergedWithPrevious".equals(behavior)) {
          result.add(slice(input, unitAtChar, matcher.start(), matcher.end()));
        } else {
          throw new TokenizerException(
              "PreTokenizerPipeline: unsupported Split.behavior '" + behavior + "'");
        }
      }
      last = matcher.end();
      previousMatchEnd = matcher.end();
    }
    if (last < text.length()) {
      result.add(slice(input, unitAtChar, last, text.length()));
    }
    return result;
  }

  private static List<AlignedText> matches(AlignedText input, Pattern pattern) {
    int[] unitAtChar = unitAtChar(input);
    Matcher matcher = pattern.matcher(input.text());
    List<AlignedText> result = new ArrayList<>();
    while (matcher.find()) {
      if (!matcher.group().isEmpty()) {
        result.add(slice(input, unitAtChar, matcher.start(), matcher.end()));
      }
    }
    return result;
  }

  /**
   * Maps every char index in {@code input.text()} to the index of the unit it belongs to, so {@link
   * #slice} can locate a match's boundary units in O(1) instead of rescanning every unit of the
   * whole input per match -- {@code matches}/{@code splitByBehavior} each call it once per
   * pre-tokenized span and reuse it across every match found within that span, since match starts
   * are visited in increasing order but a single forward-only cursor would not survive the
   * lookbehind/backtracking a caller-supplied regex can still perform (PR #24 review, finding 5).
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
}
