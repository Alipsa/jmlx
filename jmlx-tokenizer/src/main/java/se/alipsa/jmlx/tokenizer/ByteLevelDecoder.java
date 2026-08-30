package se.alipsa.jmlx.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Decodes a list of token strings back to text, byte-decoding non-added tokens and passing added
 * tokens through.
 */
public final class ByteLevelDecoder {

  private ByteLevelDecoder() {}

  /**
   * Decodes {@code tokens} to text. Consecutive non-added-token strings have their raw bytes
   * concatenated before UTF-8 decoding (a multi-byte character can be split across token
   * boundaries); added-token strings are literal text and pass through unchanged as their own
   * segment, exactly as {@code Decoder.swift}'s {@code ByteLevelDecoder} does.
   */
  public static String decode(List<String> tokens, Set<String> addedTokenContents) {
    Objects.requireNonNull(tokens, "ByteLevelDecoder.decode: tokens must not be null");
    Objects.requireNonNull(
        addedTokenContents, "ByteLevelDecoder.decode: addedTokenContents must not be null");
    StringBuilder out = new StringBuilder();
    ByteArrayOutputStream pending = new ByteArrayOutputStream();
    for (String token : tokens) {
      if (addedTokenContents.contains(token)) {
        out.append(new String(pending.toByteArray(), StandardCharsets.UTF_8));
        pending.reset();
        out.append(token);
      } else {
        pending.writeBytes(ByteLevelCoding.decodeToBytes(token));
      }
    }
    out.append(new String(pending.toByteArray(), StandardCharsets.UTF_8));
    return out.toString();
  }
}
