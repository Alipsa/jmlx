package se.alipsa.jmlx.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;

/** Applies supported decoder components to resolved vocabulary token strings. */
final class DecoderPipeline {

  private static final Pattern BYTE_TOKEN = Pattern.compile("<0x[0-9A-Fa-f]{2}>");

  private DecoderPipeline() {}

  static String decode(JsonNode config, List<String> tokens) {
    // A tokenizer.json with no decoder configured (`decoder: null`, valid HF JSON) falls back to
    // HF's own `Tokenizer::decode` default: join the resolved token strings with a plain space.
    if (config == null) {
      return String.join(" ", tokens);
    }
    List<String> result = apply(config, List.copyOf(tokens));
    return String.join("", result);
  }

  private static List<String> apply(JsonNode config, List<String> tokens) {
    String type = config.path("type").asString();
    if ("Sequence".equals(type)) {
      List<String> result = tokens;
      for (JsonNode decoder : config.path("decoders")) {
        result = apply(decoder, result);
      }
      return result;
    }
    return switch (type) {
      case "ByteLevel" -> byteLevel(tokens);
      case "Metaspace" -> metaspace(config, tokens);
      case "WordPiece" -> wordPiece(config, tokens);
      case "Replace" -> replace(config, tokens);
      case "Strip" -> tokens.stream().map(token -> strip(config, token)).toList();
      case "ByteFallback" -> byteFallback(tokens);
      case "Fuse" -> List.of(String.join("", tokens));
      default ->
          throw new TokenizerException("DecoderPipeline: unsupported decoder type '" + type + "'");
    };
  }

  private static List<String> metaspace(JsonNode config, List<String> tokens) {
    String replacement = config.path("replacement").asString("▁");
    String scheme = config.path("prepend_scheme").asString("always");
    List<String> result = new ArrayList<>(tokens.size());
    for (int index = 0; index < tokens.size(); index++) {
      String token = tokens.get(index);
      result.add(
          index == 0 && !"never".equalsIgnoreCase(scheme)
              ? token.replace(replacement, "")
              : token.replace(replacement, " "));
    }
    return result;
  }

  private static List<String> wordPiece(JsonNode config, List<String> tokens) {
    String prefix = config.path("prefix").asString("##");
    boolean cleanup = config.path("cleanup").asBoolean(true);
    List<String> result = new ArrayList<>(tokens.size());
    for (int index = 0; index < tokens.size(); index++) {
      String token = tokens.get(index);
      String value;
      if (index > 0 && token.startsWith(prefix)) {
        value = token.substring(prefix.length());
      } else {
        value = index == 0 ? token : " " + token;
      }
      result.add(cleanup ? cleanupWordPiece(value) : value);
    }
    return result;
  }

  private static String cleanupWordPiece(String value) {
    return value
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re");
  }

  private static List<String> replace(JsonNode config, List<String> tokens) {
    JsonNode pattern = config.path("pattern");
    String target =
        pattern.has("String")
            ? Pattern.quote(pattern.path("String").asString())
            : OnigRegex.whitespace(pattern.path("Regex").asString());
    // content is a literal replacement string, not a $1/backreference template (PR #24 review
    // round 2, finding 7) -- quoteReplacement keeps a literal `$` or `\` from being misread as one.
    String replacement = Matcher.quoteReplacement(config.path("content").asString());
    Pattern compiled = Pattern.compile(target);
    return tokens.stream().map(token -> compiled.matcher(token).replaceAll(replacement)).toList();
  }

  private static String strip(JsonNode config, String value) {
    char content = config.path("content").asString(" ").charAt(0);
    int start = config.path("start").asInt(0);
    int stop = config.path("stop").asInt(0);
    int left = 0;
    while (left < value.length() && left < start && value.charAt(left) == content) {
      left++;
    }
    int right = value.length();
    int removed = 0;
    while (right > left && removed < stop && value.charAt(right - 1) == content) {
      right--;
      removed++;
    }
    return value.substring(left, right);
  }

  private static List<String> byteFallback(List<String> tokens) {
    List<String> result = new ArrayList<>();
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    for (String token : tokens) {
      if (BYTE_TOKEN.matcher(token).matches()) {
        bytes.write(Integer.parseInt(token.substring(3, 5), 16));
      } else {
        flushBytes(bytes, result);
        result.add(token);
      }
    }
    flushBytes(bytes, result);
    return result;
  }

  private static void flushBytes(ByteArrayOutputStream bytes, List<String> result) {
    if (bytes.size() > 0) {
      result.add(decodeFallbackBytes(bytes.toByteArray()));
      bytes.reset();
    }
  }

  static String decodeFallbackBytes(byte[] bytes) {
    try {
      return StandardCharsets.UTF_8
          .newDecoder()
          .onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT)
          .decode(ByteBuffer.wrap(bytes))
          .toString();
    } catch (CharacterCodingException e) {
      return "\ufffd".repeat(bytes.length);
    }
  }

  private static List<String> byteLevel(List<String> tokens) {
    var decoder =
        StandardCharsets.UTF_8
            .newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE)
            .onUnmappableCharacter(CodingErrorAction.REPLACE);
    byte[] pending = new byte[0];
    List<String> result = new ArrayList<>(tokens.size());
    for (int index = 0; index < tokens.size(); index++) {
      byte[] next = ByteLevelCoding.decodeToBytes(tokens.get(index));
      byte[] combined = new byte[pending.length + next.length];
      System.arraycopy(pending, 0, combined, 0, pending.length);
      System.arraycopy(next, 0, combined, pending.length, next.length);
      ByteBuffer input = ByteBuffer.wrap(combined);
      CharBuffer output = CharBuffer.allocate(Math.max(8, combined.length * 2 + 2));
      boolean last = index + 1 == tokens.size();
      decoder.decode(input, output, last);
      if (last) {
        decoder.flush(output);
      }
      pending = new byte[input.remaining()];
      input.get(pending);
      output.flip();
      result.add(output.toString());
    }
    return List.of(String.join("", result));
  }
}
