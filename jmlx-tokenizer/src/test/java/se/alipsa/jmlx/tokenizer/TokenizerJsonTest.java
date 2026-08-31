package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TokenizerJsonTest {

  @TempDir private Path tempDir;

  private Path fixture(String name) {
    try {
      return Path.of(getClass().getResource(name).toURI());
    } catch (URISyntaxException e) {
      throw new IllegalStateException(e);
    }
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

  @Test
  void mutatingThePostProcessorOrAddedTokensListAfterConstructionDoesNotAffectTheStoredJson() {
    // TokenizerJson's compact constructor defensively copies postProcessor/addedTokens (PR #14
    // review round 7, finding 3); without it, a caller mutating the list it passed in after
    // construction would corrupt HfTokenizer's post-processing / added-token splitting mid-flight.
    // This test (round 8, finding 4) is the coverage that fix itself lacked -- constructed
    // directly rather than through TokenizerJsonLoader, since the loader never retains a
    // reference to the lists it builds.
    List<PostProcessorStep> mutablePostProcessor = new ArrayList<>(List.of(new ByteLevelStep()));
    List<AddedToken> mutableAddedTokens = new ArrayList<>();
    TokenizerJson json =
        new TokenizerJson(
            NormalizerKind.NONE,
            new PreTokenizerConfig(Pattern.compile(".+"), false),
            mutablePostProcessor,
            new BpeModelConfig(Map.of("l", 0), Map.of("l o", 0), false),
            mutableAddedTokens);
    mutablePostProcessor.clear();
    mutableAddedTokens.add(new AddedToken(99, "<x>", false));
    assertEquals(1, json.postProcessor().size());
    assertTrue(json.addedTokens().isEmpty());
  }

  @Test
  void arrayOfPairsMergesFormatParsesToTheSameMergeRankAsTheSpaceSeparatedFormat() {
    // tokenizers >= 0.20.0 (HF tokenizers PR #909) emits merges as ["l", "o"] pairs instead of
    // "l o" strings (PR #14 review, finding 1) -- both fixtures otherwise share the same
    // vocab/merge content.
    TokenizerJson spaceSeparated = TokenizerJsonLoader.load(fixture("qwen-style.tokenizer.json"));
    TokenizerJson arrayOfPairs =
        TokenizerJsonLoader.load(fixture("qwen-style-array-merges.tokenizer.json"));
    assertEquals(spaceSeparated.model().mergeRank(), arrayOfPairs.model().mergeRank());
  }

  @Test
  void emptyMergesTableThrowsInsteadOfSilentlyDegradingToPerByteTokens() throws IOException {
    Path path = tempDir.resolve("empty-merges.tokenizer.json");
    Files.writeString(path, tokenizerJson(VALID_PRETOKENIZERS, "[]"));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void malformedArrayOfPairsMergeThrowsInsteadOfNpe() throws IOException {
    // merge.get(1) is null for a 1-element array -- must throw TokenizerException, not NPE (PR
    // #14 review round 2, finding 1).
    Path path = tempDir.resolve("malformed-merge.tokenizer.json");
    Files.writeString(path, tokenizerJson(VALID_PRETOKENIZERS, "[[\"l\"]]"));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void duplicateMergePairKeepsTheFirstOccurrencesRank() throws IOException {
    // "l o" appears at index 0 (rank 0) and again (as an array pair) at index 2 -- HF keeps the
    // first occurrence's rank (PR #14 review round 2, finding 8).
    Path path = tempDir.resolve("duplicate-merge.tokenizer.json");
    Files.writeString(
        path, tokenizerJson(VALID_PRETOKENIZERS, "[\"l o\", \"t h\", [\"l\", \"o\"]]"));
    TokenizerJson json = TokenizerJsonLoader.load(path);
    assertEquals(0, json.model().mergeRank().get("l o"));
  }

  @Test
  void preTokenizerWithOnlyOneStepThrows() throws IOException {
    Path path = tempDir.resolve("one-step.tokenizer.json");
    Files.writeString(
        path,
        tokenizerJson(
            "[{\"type\": \"Split\", \"pattern\": {\"Regex\": \".+\"}, \"behavior\":"
                + " \"Isolated\", \"invert\": false}]",
            "[\"l o\"]"));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void preTokenizerWithReversedStepOrderThrows() throws IOException {
    Path path = tempDir.resolve("reversed-steps.tokenizer.json");
    Files.writeString(
        path,
        tokenizerJson(
            "[{\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"use_regex\": false},"
                + " {\"type\": \"Split\", \"pattern\": {\"Regex\": \".+\"}, \"behavior\":"
                + " \"Isolated\", \"invert\": false}]",
            "[\"l o\"]"));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void preTokenizerSplitWithInvertTrueThrows() throws IOException {
    Path path = tempDir.resolve("invert.tokenizer.json");
    Files.writeString(
        path,
        tokenizerJson(
            "[{\"type\": \"Split\", \"pattern\": {\"Regex\": \".+\"}, \"behavior\":"
                + " \"Isolated\", \"invert\": true}, {\"type\": \"ByteLevel\","
                + " \"add_prefix_space\": false, \"use_regex\": false}]",
            "[\"l o\"]"));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void preTokenizerSplitWithNonIsolatedBehaviorThrows() throws IOException {
    Path path = tempDir.resolve("removed-behavior.tokenizer.json");
    Files.writeString(
        path,
        tokenizerJson(
            "[{\"type\": \"Split\", \"pattern\": {\"Regex\": \".+\"}, \"behavior\": \"Removed\","
                + " \"invert\": false}, {\"type\": \"ByteLevel\", \"add_prefix_space\": false,"
                + " \"use_regex\": false}]",
            "[\"l o\"]"));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void preTokenizerByteLevelWithUseRegexTrueThrows() throws IOException {
    Path path = tempDir.resolve("use-regex.tokenizer.json");
    Files.writeString(
        path,
        tokenizerJson(
            "[{\"type\": \"Split\", \"pattern\": {\"Regex\": \".+\"}, \"behavior\":"
                + " \"Isolated\", \"invert\": false}, {\"type\": \"ByteLevel\","
                + " \"add_prefix_space\": false, \"use_regex\": true}]",
            "[\"l o\"]"));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void templateProcessingSpecialTokenIdsAndTokensLengthMismatchThrows() throws IOException {
    Path path = tempDir.resolve("mismatched-special-tokens.tokenizer.json");
    Files.writeString(
        path,
        """
        {
          "normalizer": null,
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": ".+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
          ]},
          "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [1, 2], "tokens": ["<s>"]}}},
          "decoder": {"type": "ByteLevel"},
          "model": {"type": "BPE", "vocab": {"l": 0}, "merges": ["l o"]},
          "added_tokens": []
        }
        """);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void templateProcessingSpecialTokenWithBothEmptyIdsAndTokensThrows() throws IOException {
    // Both-empty passes an ids.size() != tokens.size() check but would still silently drop the
    // special token (see SpecialTokenInfoTest, PR #14 review round 3, finding 1).
    Path path = tempDir.resolve("empty-special-token.tokenizer.json");
    Files.writeString(
        path,
        """
        {
          "normalizer": null,
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": ".+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
          ]},
          "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [], "tokens": []}}},
          "decoder": {"type": "ByteLevel"},
          "model": {"type": "BPE", "vocab": {"l": 0}, "merges": ["l o"]},
          "added_tokens": []
        }
        """);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void decoderWithUnsupportedTypeThrows() throws IOException {
    // decoder was previously never read at all -- a non-ByteLevel decoder loaded silently and
    // HfTokenizer#decode would produce wrong text with no diagnostic (PR #14 review round 3,
    // finding 3).
    Path path = tempDir.resolve("bad-decoder.tokenizer.json");
    Files.writeString(
        path,
        """
        {
          "normalizer": null,
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": ".+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
          ]},
          "post_processor": null,
          "decoder": {"type": "Metaspace"},
          "model": {"type": "BPE", "vocab": {"l": 0}, "merges": ["l o"]},
          "added_tokens": []
        }
        """);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void emptyVocabThrowsInsteadOfSilentlyProducingAnAllAboveVocabDecode() throws IOException {
    // An empty vocab makes Vocabulary#maxKnownId -1, so every id lands in HfTokenizer#decode's
    // above-vocab branch and decode returns "" with no error (PR #14 review round 3, finding 4).
    Path path = tempDir.resolve("empty-vocab.tokenizer.json");
    Files.writeString(
        path,
        """
        {
          "normalizer": null,
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": ".+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
          ]},
          "post_processor": null,
          "decoder": {"type": "ByteLevel"},
          "model": {"type": "BPE", "vocab": {}, "merges": ["l o"]},
          "added_tokens": []
        }
        """);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  private static final String VALID_PRETOKENIZERS =
      "[{\"type\": \"Split\", \"pattern\": {\"Regex\": \".+\"}, \"behavior\": \"Isolated\","
          + " \"invert\": false}, {\"type\": \"ByteLevel\", \"add_prefix_space\": false,"
          + " \"use_regex\": false}]";

  private static String tokenizerJson(String pretokenizersJson, String mergesJson) {
    String template =
        """
        {
          "normalizer": null,
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": %s},
          "post_processor": null,
          "decoder": {"type": "ByteLevel"},
          "model": {"type": "BPE", "vocab": {"l": 0}, "merges": %s},
          "added_tokens": []
        }
        """;
    return template.formatted(pretokenizersJson, mergesJson);
  }
}
