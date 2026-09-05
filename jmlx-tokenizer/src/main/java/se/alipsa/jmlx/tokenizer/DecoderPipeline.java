package se.alipsa.jmlx.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;

/** Applies supported decoder components to resolved vocabulary token strings. */
final class DecoderPipeline {

  private static final Pattern BYTE_TOKEN = Pattern.compile("<0x[0-9A-Fa-f]{2}>");

  private DecoderPipeline() {}

  static String decode(JsonNode config, List<String> tokens) {
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
      case "ByteLevel" -> List.of(ByteLevelDecoder.decode(tokens));
      case "Metaspace" -> List.of(metaspace(config, tokens));
      case "WordPiece" -> List.of(wordPiece(config, tokens));
      case "Replace" -> replace(config, tokens);
      case "Strip" -> List.of(strip(config, String.join("", tokens)));
      case "ByteFallback" -> byteFallback(tokens);
      case "Fuse" -> List.of(String.join("", tokens));
      default ->
          throw new TokenizerException("DecoderPipeline: unsupported decoder type '" + type + "'");
    };
  }

  private static String metaspace(JsonNode config, List<String> tokens) {
    String replacement = config.path("replacement").asString("▁");
    String result = String.join("", tokens).replace(replacement, " ");
    String scheme = config.path("prepend_scheme").asString("always");
    if (("always".equalsIgnoreCase(scheme) || "first".equalsIgnoreCase(scheme))
        && result.startsWith(" ")) {
      return result.substring(1);
    }
    return result;
  }

  private static String wordPiece(JsonNode config, List<String> tokens) {
    String prefix = config.path("prefix").asString("##");
    boolean cleanup = config.path("cleanup").asBoolean(true);
    StringBuilder result = new StringBuilder();
    for (String token : tokens) {
      if (token.startsWith(prefix)) {
        result.append(token.substring(prefix.length()));
      } else {
        if (!result.isEmpty()) {
          result.append(' ');
        }
        result.append(token);
      }
    }
    String value = result.toString();
    if (!cleanup) {
      return value;
    }
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
            : pattern.path("Regex").asString();
    String replacement = config.path("content").asString();
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
      result.add(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
      bytes.reset();
    }
  }
}
