package se.alipsa.jmlx.tokenizer;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;

/** Applies supported Hugging Face normalizer components while preserving source ranges. */
final class NormalizerPipeline {

  private NormalizerPipeline() {}

  static AlignedText apply(JsonNode config, AlignedText input) {
    if (config == null || config.isNull() || config.isMissingNode()) {
      return input;
    }
    String type = config.path("type").asString();
    return switch (type) {
      case "Sequence" -> applySequence(config.path("normalizers"), input);
      case "NFC" -> unicode(input, Normalizer.Form.NFC);
      case "NFD" -> unicode(input, Normalizer.Form.NFD);
      case "NFKC" -> unicode(input, Normalizer.Form.NFKC);
      case "NFKD" -> unicode(input, Normalizer.Form.NFKD);
      case "Lowercase" -> map(input, value -> value.toLowerCase(Locale.ROOT));
      case "StripAccents" -> stripAccents(input);
      case "Replace" -> replace(config, input);
      case "Strip" -> strip(config, input);
      case "Prepend" -> prepend(config, input);
      case "BertNormalizer" -> bert(config, input);
      default ->
          throw new TokenizerException("NormalizerPipeline: unsupported type '" + type + "'");
    };
  }

  private static AlignedText applySequence(JsonNode steps, AlignedText input) {
    AlignedText result = input;
    for (JsonNode step : steps) {
      result = apply(step, result);
    }
    return result;
  }

  private static AlignedText unicode(AlignedText input, Normalizer.Form form) {
    List<AlignedText.Unit> output = new ArrayList<>();
    List<AlignedText.Unit> units = input.units();
    for (int index = 0; index < units.size(); ) {
      int end = index + 1;
      while (end < units.size() && isMark(units.get(end).value().codePointAt(0))) {
        end++;
      }
      StringBuilder cluster = new StringBuilder();
      for (int i = index; i < end; i++) {
        cluster.append(units.get(i).value());
      }
      addMapped(
          output,
          Normalizer.normalize(cluster, form),
          units.get(index).startByte(),
          units.get(end - 1).endByte());
      index = end;
    }
    return new AlignedText(output);
  }

  private static AlignedText map(
      AlignedText input, java.util.function.Function<String, String> fn) {
    List<AlignedText.Unit> output = new ArrayList<>();
    for (AlignedText.Unit unit : input.units()) {
      addMapped(output, fn.apply(unit.value()), unit.startByte(), unit.endByte());
    }
    return new AlignedText(output);
  }

  private static AlignedText stripAccents(AlignedText input) {
    AlignedText decomposed = unicode(input, Normalizer.Form.NFD);
    return new AlignedText(
        decomposed.units().stream().filter(unit -> !isMark(unit.value().codePointAt(0))).toList());
  }

  private static boolean isMark(int codePoint) {
    int type = Character.getType(codePoint);
    return type == Character.NON_SPACING_MARK
        || type == Character.COMBINING_SPACING_MARK
        || type == Character.ENCLOSING_MARK;
  }

  private static AlignedText prepend(JsonNode config, AlignedText input) {
    String prefix = config.path("prepend").asString();
    List<AlignedText.Unit> output = new ArrayList<>();
    int boundary = input.units().isEmpty() ? 0 : input.units().getFirst().startByte();
    addMapped(output, prefix, boundary, boundary);
    output.addAll(input.units());
    return new AlignedText(output);
  }

  private static AlignedText strip(JsonNode config, AlignedText input) {
    boolean left = config.path("strip_left").asBoolean(true);
    boolean right = config.path("strip_right").asBoolean(true);
    int start = 0;
    int end = input.units().size();
    while (left && start < end && isWhitespace(input.units().get(start))) {
      start++;
    }
    while (right && end > start && isWhitespace(input.units().get(end - 1))) {
      end--;
    }
    return new AlignedText(input.units().subList(start, end));
  }

  private static boolean isWhitespace(AlignedText.Unit unit) {
    return Character.isWhitespace(unit.value().codePointAt(0));
  }

  private static AlignedText replace(JsonNode config, AlignedText input) {
    JsonNode patternNode = config.path("pattern");
    String expression;
    boolean literal;
    if (patternNode.has("String")) {
      expression = Pattern.quote(patternNode.path("String").asString());
      literal = true;
    } else if (patternNode.has("Regex")) {
      expression = patternNode.path("Regex").asString();
      literal = false;
    } else {
      throw new TokenizerException("NormalizerPipeline: Replace.pattern is unsupported");
    }
    String replacement = config.path("content").asString();
    try {
      return replaceMatches(input, Pattern.compile(expression), replacement, literal);
    } catch (RuntimeException e) {
      if (e instanceof TokenizerException) {
        throw e;
      }
      throw new TokenizerException("NormalizerPipeline: invalid Replace pattern", e);
    }
  }

  private static AlignedText replaceMatches(
      AlignedText input, Pattern pattern, String replacement, boolean literalReplacement) {
    String text = input.text();
    int[] unitAtChar = unitAtChar(input);
    Matcher matcher = pattern.matcher(text);
    List<AlignedText.Unit> output = new ArrayList<>();
    int last = 0;
    while (matcher.find()) {
      appendRange(input, unitAtChar, last, matcher.start(), output);
      TokenOffset range = range(input, unitAtChar, matcher.start(), matcher.end());
      String value =
          literalReplacement
              ? replacement
              : pattern.matcher(matcher.group()).replaceFirst(replacement);
      addMapped(output, value, range.startByte(), range.endByte());
      last = matcher.end();
    }
    appendRange(input, unitAtChar, last, text.length(), output);
    return new AlignedText(output);
  }

  private static AlignedText bert(JsonNode config, AlignedText input) {
    boolean clean = config.path("clean_text").asBoolean(true);
    boolean chinese = config.path("handle_chinese_chars").asBoolean(true);
    boolean lowercase = config.path("lowercase").asBoolean(true);
    JsonNode stripNode = config.path("strip_accents");
    boolean accents =
        stripNode.isNull() || stripNode.isMissingNode() ? lowercase : stripNode.asBoolean();
    List<AlignedText.Unit> units = new ArrayList<>();
    for (AlignedText.Unit unit : input.units()) {
      int cp = unit.value().codePointAt(0);
      if (clean && (cp == 0 || cp == 0xfffd || Character.isISOControl(cp))) {
        if (Character.isWhitespace(cp)) {
          units.add(new AlignedText.Unit(" ", unit.startByte(), unit.endByte()));
        }
        continue;
      }
      if (Character.isWhitespace(cp)) {
        units.add(new AlignedText.Unit(" ", unit.startByte(), unit.endByte()));
      } else if (chinese && isChinese(cp)) {
        units.add(new AlignedText.Unit(" ", unit.startByte(), unit.startByte()));
        units.add(unit);
        units.add(new AlignedText.Unit(" ", unit.endByte(), unit.endByte()));
      } else {
        units.add(unit);
      }
    }
    AlignedText result = new AlignedText(units);
    if (lowercase) {
      result = map(result, value -> value.toLowerCase(Locale.ROOT));
    }
    return accents ? stripAccents(result) : result;
  }

  private static boolean isChinese(int cp) {
    return (cp >= 0x4e00 && cp <= 0x9fff)
        || (cp >= 0x3400 && cp <= 0x4dbf)
        || (cp >= 0x20000 && cp <= 0x2fa1f);
  }

  private static int[] unitAtChar(AlignedText input) {
    String text = input.text();
    int[] result = new int[text.length() + 1];
    int charIndex = 0;
    for (int unit = 0; unit < input.units().size(); unit++) {
      String value = input.units().get(unit).value();
      for (int i = 0; i < value.length(); i++) {
        result[charIndex++] = unit;
      }
    }
    result[text.length()] = input.units().size();
    return result;
  }

  private static void appendRange(
      AlignedText input, int[] unitAtChar, int start, int end, List<AlignedText.Unit> output) {
    if (start >= end) {
      return;
    }
    int first = unitAtChar[start];
    int last = unitAtChar[end - 1];
    output.addAll(input.units().subList(first, last + 1));
  }

  private static TokenOffset range(AlignedText input, int[] unitAtChar, int start, int end) {
    if (start == end || input.units().isEmpty()) {
      return TokenOffset.NONE;
    }
    AlignedText.Unit first = input.units().get(unitAtChar[start]);
    AlignedText.Unit last = input.units().get(unitAtChar[end - 1]);
    return new TokenOffset(first.startByte(), last.endByte());
  }

  private static void addMapped(
      List<AlignedText.Unit> output, String value, int startByte, int endByte) {
    value
        .codePoints()
        .forEach(
            cp ->
                output.add(
                    new AlignedText.Unit(new String(Character.toChars(cp)), startByte, endByte)));
  }
}
