package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TokenizerJsonTest {

  private Path fixture(String name) {
    return Path.of("src/test/resources/se/alipsa/jmlx/tokenizer/" + name);
  }

  @Test
  void qwenStyleFixtureParsesNfcNormalizerAndByteLevelPostProcessor() {
    TokenizerJson json = TokenizerJsonLoader.load(fixture("qwen-style.tokenizer.json"));
    assertEquals(NormalizerKind.NFC, json.normalizer());
    assertFalse(json.model().ignoreMerges());
    assertEquals(1, json.postProcessor().size());
    assertTrue(json.postProcessor().get(0) instanceof ByteLevelStep);
  }

  @Test
  void llama3StyleFixtureParsesIgnoreMergesAndTemplateProcessing() {
    TokenizerJson json = TokenizerJsonLoader.load(fixture("llama3-style.tokenizer.json"));
    assertEquals(NormalizerKind.NONE, json.normalizer());
    assertTrue(json.model().ignoreMerges());
    assertEquals(2, json.postProcessor().size());
    assertTrue(json.postProcessor().get(1) instanceof TemplateProcessingStep);
  }
}
