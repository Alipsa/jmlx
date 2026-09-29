package se.alipsa.jmlx.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;

/** Incremental decoder implementation owned by one generation request. */
final class RuntimeIncrementalDecoder implements IncrementalTokenDecoder {

  private static final Pattern BYTE_TOKEN = Pattern.compile("<0x[0-9A-Fa-f]{2}>");

  private final TokenizerRuntime runtime;
  private final boolean skipSpecialTokens;
  private final JsonNode decoder;
  private final Mode mode;
  private final java.nio.charset.CharsetDecoder utf8 = newUtf8Decoder();
  private final List<Integer> bufferedIds = new ArrayList<>();
  private final ByteArrayOutputStream fallbackBytes = new ByteArrayOutputStream();
  private final UnaryOperator<String> replacer;
  private final char stripContent;
  private int stripRemaining;
  private byte[] pendingUtf8 = new byte[0];
  private boolean firstText = true;
  private boolean finished;

  RuntimeIncrementalDecoder(TokenizerRuntime runtime, boolean skipSpecialTokens, JsonNode decoder) {
    this.runtime = runtime;
    this.skipSpecialTokens = skipSpecialTokens;
    this.decoder = decoder == null ? null : decoder.deepCopy();
    this.mode = decoder == null ? Mode.BUFFERED : mode(decoder);
    if (mode == Mode.REPLACE_FALLBACK_STRIP) {
      this.replacer = DecoderPipeline.replacer(decoder.path("decoders").get(0));
      JsonNode strip = decoder.path("decoders").get(3);
      this.stripContent = strip.path("content").asString(" ").charAt(0);
      this.stripRemaining = strip.path("start").asInt(0);
    } else {
      this.replacer = null;
      this.stripContent = ' ';
    }
  }

  @Override
  public boolean streams() {
    return mode != Mode.BUFFERED;
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
      case REPLACE_FALLBACK_STRIP -> replaceFallbackStrip(text, false);
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
      case REPLACE_FALLBACK_STRIP -> replaceFallbackStrip("", true);
      case BUFFERED -> runtime.decode(bufferedIds, skipSpecialTokens);
    };
  }

  // Streams Sequence[Replace, ByteFallback, Fuse, Strip(stop=0)] (Llama 2): Replace applies to
  // ordinary tokens only, byte-fallback runs are decoded as a group, and Strip removes at most
  // `start` leading characters from the fused output, so only the leading emitted text is stripped.
  private String replaceFallbackStrip(String token, boolean end) {
    if (!end && BYTE_TOKEN.matcher(token).matches()) {
      fallbackBytes.write(Integer.parseInt(token.substring(3, 5), 16));
      return "";
    }
    StringBuilder chunk = new StringBuilder();
    if (fallbackBytes.size() > 0) {
      chunk.append(DecoderPipeline.decodeFallbackBytes(fallbackBytes.toByteArray()));
      fallbackBytes.reset();
    }
    if (!end) {
      chunk.append(replacer.apply(token));
    }
    int skip = 0;
    while (stripRemaining > 0 && skip < chunk.length() && chunk.charAt(skip) == stripContent) {
      skip++;
      stripRemaining--;
    }
    if (skip < chunk.length()) {
      stripRemaining = 0;
    }
    return chunk.substring(skip);
  }

  private String byteFallbackMetaspace(String token, boolean end) {
    if (!end && BYTE_TOKEN.matcher(token).matches()) {
      fallbackBytes.write(Integer.parseInt(token.substring(3, 5), 16));
      return "";
    }
    StringBuilder result = new StringBuilder();
    if (fallbackBytes.size() > 0) {
      result.append(metaspace(DecoderPipeline.decodeFallbackBytes(fallbackBytes.toByteArray())));
      fallbackBytes.reset();
    }
    if (!end) {
      result.append(metaspace(token));
    }
    return result.toString();
  }

  private String metaspace(String token) {
    JsonNode component = singleComponent(decoder, "Metaspace");
    String replacement = component.path("replacement").asString("▁");
    String scheme = component.path("prepend_scheme").asString("always");
    if (firstText) {
      firstText = false;
      return "never".equalsIgnoreCase(scheme)
          ? token.replace(replacement, " ")
          : token.replace(replacement, "");
    }
    return token.replace(replacement, " ");
  }

  private String wordPiece(String token, boolean end) {
    if (end) {
      return "";
    }
    JsonNode component = singleComponent(decoder, "WordPiece");
    String prefix = component.path("prefix").asString("##");
    boolean cleanup = component.path("cleanup").asBoolean(true);
    String raw =
        !firstText && token.startsWith(prefix)
            ? token.substring(prefix.length())
            : firstText ? token : " " + token;
    firstText = false;
    return cleanup ? DecoderPipeline.cleanupWordPiece(raw) : raw;
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
    if (isReplaceFallbackStrip(config)) {
      return Mode.REPLACE_FALLBACK_STRIP;
    }
    return Mode.BUFFERED;
  }

  private static boolean isReplaceFallbackStrip(JsonNode config) {
    JsonNode steps = config.path("decoders");
    if (!"Sequence".equals(config.path("type").asString()) || steps.size() != 4) {
      return false;
    }
    String[] expected = {"Replace", "ByteFallback", "Fuse", "Strip"};
    for (int i = 0; i < expected.length; i++) {
      if (!expected[i].equals(steps.get(i).path("type").asString())) {
        return false;
      }
    }
    JsonNode strip = steps.get(3);
    return strip.path("stop").asInt(0) == 0 && strip.path("content").asString(" ").length() == 1;
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
    REPLACE_FALLBACK_STRIP,
    BUFFERED
  }
}
