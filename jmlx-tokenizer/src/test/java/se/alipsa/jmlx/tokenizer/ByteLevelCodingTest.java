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
}
