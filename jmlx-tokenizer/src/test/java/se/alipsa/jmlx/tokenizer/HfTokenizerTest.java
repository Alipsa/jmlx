package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class HfTokenizerTest {

  private Path fixture(String name) {
    try {
      return Path.of(getClass().getResource(name).toURI());
    } catch (URISyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  @Test
  void qwenStyleEncodeDecodeRoundTrips() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("qwen-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", false);
    assertEquals(List.of(13, 16), ids);
    assertEquals("low the", tokenizer.decode(ids, false));
  }

  @Test
  void arrayOfPairsMergesFormatEncodeDecodeRoundTripsIdenticallyToSpaceSeparatedFormat() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("qwen-style-array-merges.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", false);
    assertEquals(List.of(13, 16), ids);
    assertEquals("low the", tokenizer.decode(ids, false));
  }

  @Test
  void llama3StylePrependsBosTokenWhenAddSpecialTokensIsTrue() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals(List.of(128000, 13, 16), ids);
  }

  @Test
  void llama3StyleOmitsBosTokenWhenAddSpecialTokensIsFalse() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    assertEquals(List.of(13, 16), tokenizer.encode("low the", false));
  }

  @Test
  void decodeSkipsSpecialTokensWhenRequested() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals("low the", tokenizer.decode(ids, true));
  }

  @Test
  void decodeSkipsAnAboveVocabIdInsteadOfAbortingTheWholeSequence() {
    // e.g. Qwen2.5's config.json vocab_size (152064) exceeds its tokenizer vocab (151665) -- a
    // sampled logit can legitimately land in that gap. qwen-style.tokenizer.json's ids top out at
    // 19, so 99999 is above the known vocabulary range and must be skipped, not thrown on.
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("qwen-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", false);
    List<Integer> withAboveVocabId = new ArrayList<>(ids);
    withAboveVocabId.add(1, 99999);
    assertEquals("low the", tokenizer.decode(withAboveVocabId, false));
  }

  @Test
  void decodeThrowsOnAnInRangeIdWithNoVocabularyEntryInsteadOfSilentlyDroppingIt() {
    // qwen-style-with-id-gap.tokenizer.json adds an id-100 token on top of qwen-style's usual
    // 0..19, leaving 20..99 as a genuine in-range hole -- indistinguishable from the above-vocab
    // case by id value alone, so it must throw instead of being swept up by the same skip (PR #14
    // review, finding 2: the fix must not make an in-range hole -- wrong tokenizer for the
    // checkpoint, a mis-parsed added_tokens, a Vocabulary bug -- as silent as a legitimate gap).
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("qwen-style-with-id-gap.tokenizer.json"));
    assertThrows(TokenizerException.class, () -> tokenizer.decode(List.of(50), false));
  }

  @Test
  void encodeTrustsATemplateSpecialTokenIdThatHasNoVocabularyEntryAtAll() {
    // The template's special_tokens.ids value (999999) has no vocabulary entry at all -- HF's own
    // TemplateProcessing performs no vocabulary lookup on special_tokens ids, and ResolvedToken's
    // own javadoc documents exactly this case, so encode must trust it as-is rather than throw
    // (PR #14 review round 3, finding 2/6, correcting round 2's finding 7, which rejected every
    // unknown id and so also rejected this legitimate case).
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(fixture("llama3-style-template-id-absent-from-vocab.tokenizer.json"));
    assertEquals(List.of(999999, 13, 16), tokenizer.encode("low the", true));
  }

  @Test
  void encodeThrowsWhenATemplateSpecialTokenIdConflictsWithADifferentVocabularyToken() {
    // The template's special_tokens.ids value (15) is a real, existing vocabulary id -- but it
    // belongs to "the", not "<|begin_of_text|>". Unlike an id with no vocabulary entry at all
    // (see the test above), this is a genuine internal contradiction within the same file, and
    // baking it in would silently swap in the wrong token wherever this id is later decoded (PR
    // #14 review, finding 2).
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(
            fixture("llama3-style-template-id-conflicts-with-vocab.tokenizer.json"));
    assertThrows(TokenizerException.class, () -> tokenizer.encode("low the", true));
  }
}
