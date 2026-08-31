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
   *
   * <p>If any character of {@code byteLevelText} is not one of the 256 byte-level symbols, the
   * whole string falls back to its own raw UTF-8 bytes instead of throwing -- matching HF's actual
   * {@code ByteLevel::decode_chain} ({@code huggingface/tokenizers}' Rust source), which tries a
   * per-character mapping and falls back to {@code t.as_bytes()} for the *entire* token as soon as
   * any character fails, rather than erroring. This is a real, reachable case for this port, not
   * just a defensive check: a plain {@code model.vocab} token string -- which, after round 5
   * finding 1's fix, can also be a template token matching an existing {@code model.vocab} entry --
   * can contain a character like a plain space (code point 32, itself not byte-level-printable --
   * see the static initializer above) that no legitimate byte-level BPE symbol ever would. Since
   * {@link ByteLevelDecoder} stopped special-casing added-token strings with a literal pass-through
   * (PR #14 review round 6, finding 1), an {@code added_tokens} entry's own content is reachable
   * here too, for the same reason: neither case is unreachable the way an id with no vocabulary
   * entry at all is (that case is filtered out in {@code HfTokenizer#decode} before any token
   * string is ever produced, so it never reaches this method). The eventual {@code new
   * String(bytes, UTF_8)} in {@link ByteLevelDecoder} already matches {@code from_utf8_lossy}'s
   * replacement-character behavior for whatever these raw bytes decode to, so no further change is
   * needed there (PR #14 review round 5, finding 3; reachability description corrected in round 6,
   * finding 6, to account for round 6 finding 1's own change to {@link ByteLevelDecoder}).
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
        return byteLevelText.getBytes(StandardCharsets.UTF_8);
      }
      out[i++] = b.byteValue();
      index += Character.charCount(codePoint);
    }
    return out;
  }
}
