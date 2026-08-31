package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class ByteLevelCodingTest {

  @Test
  void spaceEncodesToGSubscriptCharacter() {
    // The well-known GPT-2 encoding: byte 0x20 (space) maps to U+0120 ('Ġ').
    assertEquals("Ġ", ByteLevelCoding.encode(" "));
  }

  @Test
  void asciiLettersEncodeUnchanged() {
    assertEquals("low", ByteLevelCoding.encode("low"));
  }

  @Test
  void everyByteValueRoundTrips() {
    byte[] allBytes = new byte[256];
    for (int b = 0; b < 256; b++) {
      allBytes[b] = (byte) b;
    }
    String encoded = ByteLevelCoding.encode(allBytes);
    assertArrayEquals(allBytes, ByteLevelCoding.decodeToBytes(encoded));
  }

  @Test
  void multiByteUtf8RoundTrips() {
    byte[] utf8 = "héllo 中文".getBytes(StandardCharsets.UTF_8);
    String encoded = ByteLevelCoding.encode(utf8);
    assertArrayEquals(utf8, ByteLevelCoding.decodeToBytes(encoded));
  }

  @Test
  void stringWithNonByteLevelCharacterFallsBackToItsOwnUtf8BytesInsteadOfThrowing() {
    // A plain space (code point 32) is not itself byte-level-printable -- see the class's static
    // initializer -- so it can never occur in a legitimately byte-level-encoded token, but a
    // malformed/unexpected token string can still contain one. HF's actual ByteLevel::decode_chain
    // falls back to the whole token's own raw UTF-8 bytes as soon as any one character fails to
    // map, rather than erroring (PR #14 review round 5, finding 3).
    assertArrayEquals("a b".getBytes(StandardCharsets.UTF_8), ByteLevelCoding.decodeToBytes("a b"));
  }

  @Test
  void stringWithNonAsciiUnmappedCharactersFallsBackToItsOwnUtf8Bytes() {
    assertArrayEquals("你好".getBytes(StandardCharsets.UTF_8), ByteLevelCoding.decodeToBytes("你好"));
  }
}
