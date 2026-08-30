package se.alipsa.jmlx.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The GPT-2 byte-level encoding: maps each of the 256 possible byte values to a printable Unicode
 * character, so that BPE vocab/merges (which operate on printable text) can represent arbitrary
 * bytes, including whitespace and control characters, without ambiguity.
 */
public final class ByteLevelCoding {

  private static final int[] BYTE_TO_CODE_POINT = new int[256];
  private static final Map<Integer, Integer> CODE_POINT_TO_BYTE = new HashMap<>();

  static {
    boolean[] isPrintable = new boolean[256];
    for (int b = '!'; b <= '~'; b++) {
      isPrintable[b] = true;
    }
    for (int b = 0xA1; b <= 0xAC; b++) {
      isPrintable[b] = true;
    }
    for (int b = 0xAE; b <= 0xFF; b++) {
      isPrintable[b] = true;
    }
    int nextExtraCodePoint = 256;
    for (int b = 0; b < 256; b++) {
      int codePoint = isPrintable[b] ? b : nextExtraCodePoint++;
      BYTE_TO_CODE_POINT[b] = codePoint;
      CODE_POINT_TO_BYTE.put(codePoint, b);
    }
  }

  private ByteLevelCoding() {}

  /** Encodes raw UTF-8 bytes as a byte-level string: one Unicode character per input byte. */
  public static String encode(byte[] utf8Bytes) {
    Objects.requireNonNull(utf8Bytes, "ByteLevelCoding.encode: utf8Bytes must not be null");
    StringBuilder sb = new StringBuilder(utf8Bytes.length);
    for (byte b : utf8Bytes) {
      sb.appendCodePoint(BYTE_TO_CODE_POINT[b & 0xFF]);
    }
    return sb.toString();
  }

  /** Encodes a plain-text string (its UTF-8 bytes) as a byte-level string. */
  public static String encode(String text) {
    Objects.requireNonNull(text, "ByteLevelCoding.encode: text must not be null");
    return encode(text.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Decodes a byte-level string back to raw bytes. Does not decode as UTF-8 itself: a multi-byte
   * UTF-8 character can be split across separate BPE tokens, so callers must concatenate the raw
   * bytes of every consecutive byte-level token before UTF-8-decoding the combined buffer (see
   * {@link ByteLevelDecoder}).
   */
  public static byte[] decodeToBytes(String byteLevelText) {
    Objects.requireNonNull(
        byteLevelText, "ByteLevelCoding.decodeToBytes: byteLevelText must not be null");
    byte[] out = new byte[byteLevelText.codePointCount(0, byteLevelText.length())];
    int i = 0;
    int index = 0;
    while (index < byteLevelText.length()) {
      int codePoint = byteLevelText.codePointAt(index);
      Integer b = CODE_POINT_TO_BYTE.get(codePoint);
      if (b == null) {
        throw new TokenizerException(
            "ByteLevelCoding.decodeToBytes: code point "
                + codePoint
                + " is not a valid byte-level character");
      }
      out[i++] = b.byteValue();
      index += Character.charCount(codePoint);
    }
    return out;
  }
}
