package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class ByteLevelDecoderTest {

  @Test
  void decodesPlainByteLevelTokensBackToText() {
    assertEquals("low the", ByteLevelDecoder.decode(List.of("low", "Ġthe")));
  }

  @Test
  void specialTokenShapedStringDecodesToItselfSinceItIsAllPrintableAscii() {
    // No added-token special-casing here at all (PR #14 review round 6, finding 1): every token
    // is byte-decoded uniformly, matching HF's own decode_chain, which never distinguishes added
    // tokens from model-vocab tokens either. This still round-trips a real added token's content
    // (e.g. "<|im_start|>") correctly, because every character in it falls in the printable-ASCII
    // range ByteLevelCoding maps identically to its own byte value -- true of every added token in
    // both Qwen2.5's and Llama-3's real tokenizer.json.
    String result = ByteLevelDecoder.decode(List.of("<|im_start|>", "Ġthe"));
    assertEquals("<|im_start|> the", result);
  }

  @Test
  void multiByteCharacterSplitAcrossTwoTokensStillDecodesCorrectly() {
    // "é" is 2 UTF-8 bytes (0xC3 0xA9); simulate them arriving as two separate BPE token pieces.
    String piece1 = ByteLevelCoding.encode(new byte[] {(byte) 0xC3});
    String piece2 = ByteLevelCoding.encode(new byte[] {(byte) 0xA9});
    assertEquals("é", ByteLevelDecoder.decode(List.of(piece1, piece2)));
  }
}
