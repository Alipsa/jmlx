package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;

/** Hugging Face Digits splits numeric characters before the following ByteLevel step. */
class DigitsPreTokenizerTest {
  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void isolatesIndividualNumericCharacters() {
    var config = mapper.readTree("{\"type\":\"Digits\",\"individual_digits\":true}");
    assertEquals(
        List.of("Call ", "1", "2", "3", " please", "①", "Ⅷ", "!"),
        texts(PreTokenizerPipeline.apply(config, AlignedText.original("Call 123 please①Ⅷ!"))));
  }

  @Test
  void groupsNumericRunsByDefault() {
    var config = mapper.readTree("{\"type\":\"Digits\"}");
    assertEquals(
        List.of("Call ", "123", " please", "①Ⅷ", "!"),
        texts(PreTokenizerPipeline.apply(config, AlignedText.original("Call 123 please①Ⅷ!"))));
  }

  private static List<String> texts(List<AlignedText> spans) {
    return spans.stream().map(AlignedText::text).toList();
  }
}
