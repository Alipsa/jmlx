package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class ByteLevelPreTokenizerTest {

  @Test
  void rejectsANullSplitPatternAtConfigurationConstruction() {
    assertThrows(NullPointerException.class, () -> new PreTokenizerConfig(null, false));
  }

  // The real Qwen2.5/Llama-3 regex (Qwen2.5's \p{N} variant), verified against each model's
  // actual tokenizer.json — see this plan's Findings section. \s/\S substituted with
  // \p{IsWhite_Space}/\P{IsWhite_Space}: HF's onig backend's \s is Unicode White_Space (e.g.
  // matches U+00A0 NBSP), not ASCII-only, matching what TokenizerJsonLoader.parsePreTokenizer
  // itself substitutes before compiling a file's regex (PR #24 review round 2, finding 3,
  // correcting PR #14 review round 4, finding 1 and PR #14 review round 5, finding 10 -- see
  // unicodeWhitespaceClassMatchesOnigNotJavasAsciiOnlyDefault below).
  private static final Pattern QWEN_REGEX =
      Pattern.compile(
          "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
              + "\\n"
              + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\p{IsWhite_Space}\\p{L}\\p{N}]+[\\r"
              + "\\n"
              + "]*|\\p{IsWhite_Space}*[\\r"
              + "\\n"
              + "]+|\\p{IsWhite_Space}+(?!\\P{IsWhite_Space})|\\p{IsWhite_Space}+");

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
  void twoWhitespaceCharsSplitAsOneChunkThenALeadingSpacePlusWord() {
    // "\s+(?!\S)" only refuses to consume the very last whitespace character of a run (so the
    // next word's own " ?<word>" alternative can claim a single leading space) -- verified
    // against the pinned oracle: a run of two whitespace characters (ASCII space or NBSP alike)
    // splits as [word, one-whitespace-char, one-whitespace-char + next-word], never as
    // [word, both-whitespace-chars-fused, next-word] (PR #24 review round 2, finding 3, correcting
    // this test's own prior, oracle-contradicted expectation from PR #14 review round 4/5).
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(QWEN_REGEX, false));
    assertEquals(
        List.of("hi", ByteLevelCoding.encode(" "), ByteLevelCoding.encode(" there")),
        pretokenizer.split("hi  there"));
  }

  @Test
  void nbspSplitsIdenticallyToAsciiSpaceUnderOnigsUnicodeWhitespace() {
    // HF compiles this regex with onig (the default "onig" cargo feature), whose \s is Unicode
    // White_Space, not ASCII-only -- verified against the pinned oracle: NBSP (U+00A0) splits a
    // two-character whitespace run exactly like two ASCII spaces do (see
    // twoWhitespaceCharsSplitAsOneChunkThenALeadingSpacePlusWord above), not as one merged
    // "any other run of chars" chunk (PR #24 review round 2, finding 3, correcting PR #14 review
    // round 4, finding 1).
    ByteLevelPreTokenizer pretokenizer =
        new ByteLevelPreTokenizer(new PreTokenizerConfig(QWEN_REGEX, false));
    String nbsp = " ";
    assertEquals(
        List.of("hi", ByteLevelCoding.encode(nbsp), ByteLevelCoding.encode(nbsp + "there")),
        pretokenizer.split("hi" + nbsp + nbsp + "there"));
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
