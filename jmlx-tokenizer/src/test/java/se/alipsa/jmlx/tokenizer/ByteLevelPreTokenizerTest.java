package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class ByteLevelPreTokenizerTest {

  // The real Qwen2.5/Llama-3 regex (Qwen2.5's \p{N} variant), verified against each model's
  // actual tokenizer.json — see this plan's Findings section.
  private static final Pattern QWEN_REGEX =
      Pattern.compile(
          "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
              + "\\n"
              + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
              + "\\n"
              + "]*|\\s*[\\r"
              + "\\n"
              + "]+|\\s+(?!\\S)|\\s+",
          Pattern.UNICODE_CHARACTER_CLASS);

  @Test
  void splitsWordAndLeadingSpaceIntoSeparateChunks() {
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(QWEN_REGEX, false));
    // "low the": "low" has no leading space; " the" is captured as one chunk with its leading
    // space.
    assertEquals(List.of("low", "Ġthe"), pretokenizer.split("low the"));
  }

  @Test
  void contractionIsItsOwnChunk() {
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(QWEN_REGEX, false));
    assertEquals(List.of("it", "'s"), pretokenizer.split("it's"));
  }
}
