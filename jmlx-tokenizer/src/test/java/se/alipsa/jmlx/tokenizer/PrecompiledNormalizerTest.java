package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;
import org.junit.jupiter.api.Test;

class PrecompiledNormalizerTest {
  @Test
  void malformedBinaryTablesAreRejectedBeforeLookup() {
    assertThrows(TokenizerException.class, () -> new PrecompiledNormalizer("!not-base64"));
    assertThrows(TokenizerException.class, () -> new PrecompiledNormalizer("AA=="));
    for (int size : new int[] {0, 3, 12, -1}) {
      byte[] bytes =
          ByteBuffer.allocate(9)
              .order(ByteOrder.LITTLE_ENDIAN)
              .putInt(size)
              .putInt(0)
              .put((byte) 0)
              .array();
      assertThrows(
          TokenizerException.class,
          () -> new PrecompiledNormalizer(Base64.getEncoder().encodeToString(bytes)));
    }
    for (byte replacement : new byte[] {(byte) 0xff, 1}) {
      byte[] bytes =
          ByteBuffer.allocate(9)
              .order(ByteOrder.LITTLE_ENDIAN)
              .putInt(4)
              .putInt(0)
              .put(replacement)
              .array();
      assertThrows(
          TokenizerException.class,
          () -> new PrecompiledNormalizer(Base64.getEncoder().encodeToString(bytes)));
    }
    byte[] invalidRoot =
        ByteBuffer.allocate(9)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(4)
            .putInt(1024)
            .put((byte) 0)
            .array();
    assertThrows(
        TokenizerException.class,
        () -> new PrecompiledNormalizer(Base64.getEncoder().encodeToString(invalidRoot)));
  }
}
