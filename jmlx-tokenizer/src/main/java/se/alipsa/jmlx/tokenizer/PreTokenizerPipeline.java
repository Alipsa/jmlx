package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import tools.jackson.databind.JsonNode;

/** Applies supported pre-tokenizer components to aligned text spans. */
final class PreTokenizerPipeline {

  private static final Pattern BYTE_LEVEL_PATTERN =
      Pattern.compile(
          "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|"
              + "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+");
  private static final Pattern WHITESPACE_PATTERN =
      Pattern.compile("[\\p{L}\\p{N}_]+|[^\\p{L}\\p{N}_\\s]+");

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
      int boundary = value.units().getFirst().startByte();
      units.add(new AlignedText.Unit(" ", boundary, boundary));
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
            && !input.text().startsWith(" ")
            && ("always".equals(scheme) || "first".equals(scheme));
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
    return matches(input, Pattern.compile("\\S+"));
  }

  private static List<AlignedText> bert(AlignedText input) {
    List<AlignedText> result = new ArrayList<>();
    List<AlignedText.Unit> current = new ArrayList<>();
    for (AlignedText.Unit unit : input.units()) {
      int cp = unit.value().codePointAt(0);
      if (Character.isWhitespace(cp)) {
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
      expression = patternNode.path("Regex").asString();
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
    Matcher matcher = pattern.matcher(text);
    List<AlignedText> result = new ArrayList<>();
    int last = 0;
    int previousMatchEnd = -1;
    while (matcher.find()) {
      if (matcher.start() > last) {
        result.add(slice(input, last, matcher.start()));
      }
      if (!matcher.group().isEmpty() && !"Removed".equals(behavior)) {
        if ("Contiguous".equals(behavior)
            && matcher.start() == previousMatchEnd
            && !result.isEmpty()) {
          AlignedText previous = result.removeLast();
          List<AlignedText.Unit> merged = new ArrayList<>(previous.units());
          merged.addAll(slice(input, matcher.start(), matcher.end()).units());
          result.add(new AlignedText(merged));
        } else if ("Isolated".equals(behavior) || "Contiguous".equals(behavior)) {
          result.add(slice(input, matcher.start(), matcher.end()));
        } else if ("MergedWithPrevious".equals(behavior) && !result.isEmpty()) {
          AlignedText previous = result.removeLast();
          List<AlignedText.Unit> merged = new ArrayList<>(previous.units());
          merged.addAll(slice(input, matcher.start(), matcher.end()).units());
          result.add(new AlignedText(merged));
        } else {
          throw new TokenizerException(
              "PreTokenizerPipeline: unsupported Split.behavior '" + behavior + "'");
        }
      }
      last = matcher.end();
      previousMatchEnd = matcher.end();
    }
    if (last < text.length()) {
      result.add(slice(input, last, text.length()));
    }
    return result;
  }

  private static List<AlignedText> matches(AlignedText input, Pattern pattern) {
    Matcher matcher = pattern.matcher(input.text());
    List<AlignedText> result = new ArrayList<>();
    while (matcher.find()) {
      if (!matcher.group().isEmpty()) {
        result.add(slice(input, matcher.start(), matcher.end()));
      }
    }
    return result;
  }

  private static AlignedText slice(AlignedText input, int startChar, int endChar) {
    List<AlignedText.Unit> result = new ArrayList<>();
    int charIndex = 0;
    for (AlignedText.Unit unit : input.units()) {
      int next = charIndex + unit.value().length();
      if (next > startChar && charIndex < endChar) {
        result.add(unit);
      }
      charIndex = next;
    }
    return new AlignedText(result);
  }
}
