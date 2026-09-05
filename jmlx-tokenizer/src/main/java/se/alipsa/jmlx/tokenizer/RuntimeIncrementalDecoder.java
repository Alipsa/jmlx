package se.alipsa.jmlx.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;

/** Incremental decoder implementation owned by one generation request. */
final class RuntimeIncrementalDecoder implements IncrementalTokenDecoder {

  private static final Pattern BYTE_TOKEN = Pattern.compile("<0x[0-9A-Fa-f]{2}>");
  private static final int WORDPIECE_PENDING_CHARS = 5;

  private final TokenizerRuntime runtime;
  private final boolean skipSpecialTokens;
  private final JsonNode decoder;
  private final Mode mode;
  private final java.nio.charset.CharsetDecoder utf8 = newUtf8Decoder();
  private final List<Integer> bufferedIds = new ArrayList<>();
  private final ByteArrayOutputStream fallbackBytes = new ByteArrayOutputStream();
  private byte[] pendingUtf8 = new byte[0];
  private boolean firstText = true;
  private String wordPiecePending = "";
  private boolean finished;

  RuntimeIncrementalDecoder(TokenizerRuntime runtime, boolean skipSpecialTokens, JsonNode decoder) {
    this.runtime = runtime;
    this.skipSpecialTokens = skipSpecialTokens;
    this.decoder = decoder.deepCopy();
    this.mode = mode(decoder);
  }

  @Override
  public String append(int tokenId) {
    requireOpen();
    TokenizerRuntime.DecodableToken token = runtime.decodableToken(tokenId, skipSpecialTokens);
    if (token == null) {
      return "";
    }
    String text = token.text();
    return switch (mode) {
      case BYTE_LEVEL -> decodeUtf8(ByteLevelCoding.decodeToBytes(text), false);
      case METASPACE -> metaspace(text);
      case BYTE_FALLBACK_METASPACE -> byteFallbackMetaspace(text, false);
      case WORDPIECE -> wordPiece(text, false);
      case BUFFERED -> {
        bufferedIds.add(tokenId);
        yield "";
      }
    };
  }

  @Override
  public String finish() {
    requireOpen();
    finished = true;
    return switch (mode) {
      case BYTE_LEVEL -> decodeUtf8(new byte[0], true);
      case METASPACE -> "";
      case BYTE_FALLBACK_METASPACE -> byteFallbackMetaspace("", true);
      case WORDPIECE -> wordPiece("", true);
      case BUFFERED -> runtime.decode(bufferedIds, skipSpecialTokens);
    };
  }

  private String byteFallbackMetaspace(String token, boolean end) {
    if (!end && BYTE_TOKEN.matcher(token).matches()) {
      fallbackBytes.write(Integer.parseInt(token.substring(3, 5), 16));
      return "";
    }
    StringBuilder result = new StringBuilder();
    if (fallbackBytes.size() > 0) {
      result.append(new String(fallbackBytes.toByteArray(), StandardCharsets.UTF_8));
      fallbackBytes.reset();
    }
    if (!end) {
      result.append(token);
    }
    return metaspace(result.toString());
  }

  private String metaspace(String token) {
    JsonNode component = singleComponent(decoder, "Metaspace");
    String replacement = component.path("replacement").asString("▁");
    String value = token.replace(replacement, " ");
    String scheme = component.path("prepend_scheme").asString("always");
    if (firstText) {
      firstText = false;
      if (("always".equalsIgnoreCase(scheme) || "first".equalsIgnoreCase(scheme))
          && value.startsWith(" ")) {
        return value.substring(1);
      }
    }
    return value;
  }

  private String wordPiece(String token, boolean end) {
    JsonNode component = singleComponent(decoder, "WordPiece");
    String prefix = component.path("prefix").asString("##");
    boolean cleanup = component.path("cleanup").asBoolean(true);
    if (!end) {
      String raw;
      if (token.startsWith(prefix)) {
        raw = token.substring(prefix.length());
      } else {
        raw = firstText ? token : " " + token;
      }
      firstText = false;
      wordPiecePending += raw;
    }
    if (!cleanup) {
      String result = wordPiecePending;
      wordPiecePending = "";
      return result;
    }
    String cleaned = cleanupWordPiece(wordPiecePending);
    if (!end && cleaned.length() <= WORDPIECE_PENDING_CHARS) {
      wordPiecePending = cleaned;
      return "";
    }
    if (end) {
      wordPiecePending = "";
      return cleaned;
    }
    int emit = cleaned.length() - WORDPIECE_PENDING_CHARS;
    if (emit > 0 && Character.isHighSurrogate(cleaned.charAt(emit - 1))) {
      emit--;
    }
    String result = cleaned.substring(0, emit);
    wordPiecePending = cleaned.substring(emit);
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

  private String decodeUtf8(byte[] bytes, boolean end) {
    byte[] combined = new byte[pendingUtf8.length + bytes.length];
    System.arraycopy(pendingUtf8, 0, combined, 0, pendingUtf8.length);
    System.arraycopy(bytes, 0, combined, pendingUtf8.length, bytes.length);
    ByteBuffer input = ByteBuffer.wrap(combined);
    CharBuffer output = CharBuffer.allocate(Math.max(8, combined.length * 2 + 2));
    try {
      var result = utf8.decode(input, output, end);
      if (result.isError()) {
        result.throwException();
      }
      if (end) {
        result = utf8.flush(output);
        if (result.isError()) {
          result.throwException();
        }
      }
    } catch (CharacterCodingException e) {
      throw new TokenizerException("IncrementalTokenDecoder: UTF-8 decoding failed", e);
    }
    pendingUtf8 = new byte[input.remaining()];
    input.get(pendingUtf8);
    output.flip();
    return output.toString();
  }

  private static Mode mode(JsonNode config) {
    if (isOnly(config, "ByteLevel")) {
      return Mode.BYTE_LEVEL;
    }
    if (isOnly(config, "Metaspace")) {
      return Mode.METASPACE;
    }
    if (isOnly(config, "WordPiece")) {
      return Mode.WORDPIECE;
    }
    if (isSequence(config, "ByteFallback", "Metaspace")) {
      return Mode.BYTE_FALLBACK_METASPACE;
    }
    return Mode.BUFFERED;
  }

  private static boolean isOnly(JsonNode config, String type) {
    if (type.equals(config.path("type").asString())) {
      return true;
    }
    return "Sequence".equals(config.path("type").asString())
        && config.path("decoders").size() == 1
        && type.equals(config.path("decoders").get(0).path("type").asString());
  }

  private static boolean isSequence(JsonNode config, String first, String second) {
    JsonNode steps = config.path("decoders");
    return "Sequence".equals(config.path("type").asString())
        && steps.size() == 2
        && first.equals(steps.get(0).path("type").asString())
        && second.equals(steps.get(1).path("type").asString());
  }

  private static JsonNode singleComponent(JsonNode config, String type) {
    if (type.equals(config.path("type").asString())) {
      return config;
    }
    for (JsonNode child : config.path("decoders")) {
      if (type.equals(child.path("type").asString())) {
        return child;
      }
    }
    throw new IllegalStateException("incremental decoder component is missing: " + type);
  }

  private static java.nio.charset.CharsetDecoder newUtf8Decoder() {
    return StandardCharsets.UTF_8
        .newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE);
  }

  private void requireOpen() {
    if (finished) {
      throw new IllegalStateException("IncrementalTokenDecoder is already finished");
    }
  }

  private enum Mode {
    BYTE_LEVEL,
    METASPACE,
    BYTE_FALLBACK_METASPACE,
    WORDPIECE,
    BUFFERED
  }
}
