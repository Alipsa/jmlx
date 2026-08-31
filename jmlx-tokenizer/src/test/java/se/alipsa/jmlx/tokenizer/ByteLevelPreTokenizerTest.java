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

  @Test
  void addPrefixSpaceAppliesToEachSplitPieceIndependentlyNotOnlyTheWholeInput() {
    // Every match that doesn't already start with a space gets its own prefix space -- not just
    // the first character of the whole input (PR #14 review, finding 4).
    Pattern letterOrNonLetter = Pattern.compile("\\p{L}+|[^\\p{L}]+");
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(letterOrNonLetter, true));
    assertEquals(List.of("Ġ!", "Ġhi"), pretokenizer.split("!hi"));
  }

  @Test
  void anUnmatchedInteriorSpanIsEmittedAsItsOwnChunkNotDropped() {
    // This pattern only matches letters, leaving "!" uncovered -- HF's find_matches contract
    // keeps such spans (SplitDelimiterBehavior::Isolated) rather than discarding them (PR #14
    // review, finding 3: the prior "throw on any gap" fix was stricter than HF itself).
    Pattern lettersOnly = Pattern.compile("\\p{L}+");
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(lettersOnly, false));
    assertEquals(List.of("hi", "!", "there"), pretokenizer.split("hi!there"));
  }

  @Test
  void anUnmatchedLeadingSpanIsEmittedAsItsOwnChunk() {
    Pattern lettersOnly = Pattern.compile("\\p{L}+");
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(lettersOnly, false));
    assertEquals(List.of("!", "hi"), pretokenizer.split("!hi"));
  }

  @Test
  void anUnmatchedTrailingSpanIsEmittedAsItsOwnChunk() {
    Pattern lettersOnly = Pattern.compile("\\p{L}+");
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(lettersOnly, false));
    assertEquals(List.of("hi", "!"), pretokenizer.split("hi!"));
  }

  @Test
  void zeroWidthMatchDoesNotEmitAnEmptyChunk() {
    // A regex with a zero-width alternative can match an empty string between two real matches;
    // neither shipped Split pattern has one, but the empty chunk it would otherwise produce is
    // harmless downstream (BpeMerger.merge("") yields nothing) -- skipping it is simpler than
    // relying on that (PR #14 review round 3, finding 9).
    Pattern lettersOrZeroWidthBeforeComma = Pattern.compile("\\p{L}+|(?=,)");
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(lettersOrZeroWidthBeforeComma, false));
    assertEquals(List.of("a", ",", "b"), pretokenizer.split("a,b"));
  }
}
