package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ByteLevelDecoderTest {

  @Test
  void decodesPlainByteLevelTokensBackToText() {
    assertEquals("low the", ByteLevelDecoder.decode(List.of("low", "Ġthe"), Set.of()));
  }

  @Test
  void addedTokensPassThroughLiterallyBetweenDecodedText() {
    String result =
        ByteLevelDecoder.decode(
            List.of("<|im_start|>", "Ġthe"), Set.of("<|im_start|>", "<|im_end|>"));
    assertEquals("<|im_start|> the", result);
  }

  @Test
  void multiByteCharacterSplitAcrossTwoTokensStillDecodesCorrectly() {
    // "é" is 2 UTF-8 bytes (0xC3 0xA9); simulate them arriving as two separate BPE token pieces.
    String piece1 = ByteLevelCoding.encode(new byte[] {(byte) 0xC3});
    String piece2 = ByteLevelCoding.encode(new byte[] {(byte) 0xA9});
    assertEquals("é", ByteLevelDecoder.decode(List.of(piece1, piece2), Set.of()));
  }
}
