package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Unit coverage for {@link TokenizerJsonLoader}'s own field-level validations -- {@code
 * decoder}/{@code pre_tokenizer}/{@code model}/{@code added_tokens} shapes this port either
 * requires or rejects as unsupported -- most of which previously had no test at all (PR #14 review
 * round 5, finding 4).
 */
class TokenizerJsonLoaderTest {

  @TempDir Path tempDir;

  private static final String VALID_DECODER =
      "{\"type\": \"ByteLevel\", \"add_prefix_space\": true, \"trim_offsets\": true, \"use_regex\":"
          + " true}";
  private static final String VALID_PRE_TOKENIZER =
      "{\"type\": \"Sequence\", \"pretokenizers\": [{\"type\": \"Split\", \"pattern\": {\"Regex\":"
          + " \"\\\\w+\"}, \"behavior\": \"Isolated\", \"invert\": false},{\"type\": \"ByteLevel\","
          + " \"add_prefix_space\": false, \"trim_offsets\": true, \"use_regex\": false}]}";
  private static final String VALID_MODEL =
      "{\"type\": \"BPE\", \"vocab\": {\"a\": 0, \"b\": 1}, \"merges\": [\"a b\"]}";

  private Path writeTokenizerJson(
      String decoder, String preTokenizer, String model, String addedTokens) {
    String json =
        "{"
            + "\"normalizer\": null,"
            + "\"pre_tokenizer\": "
            + preTokenizer
            + ","
            + "\"post_processor\": null,"
            + "\"decoder\": "
            + decoder
            + ","
            + "\"model\": "
            + model
            + ","
            + "\"added_tokens\": "
            + addedTokens
            + "}";
    try {
      Path path = tempDir.resolve("tokenizer.json");
      Files.writeString(path, json);
      return path;
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private Path validTokenizerJsonWith(String field, String rawValue) {
    String model =
        "{\"type\": \"BPE\", \"vocab\": {\"a\": 0, \"b\": 1}, \"merges\": [\"a b\"], \""
            + field
            + "\": "
            + rawValue
            + "}";
    return writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, model, "[]");
  }

  @Test
  void baselineValidFileLoadsSuccessfully() {
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, "[]");
    assertDoesNotThrow(() -> TokenizerJsonLoader.load(path));
  }

  @Test
  void missingDecoderFieldThrows() {
    String json =
        "{\"normalizer\": null, \"pre_tokenizer\": "
            + VALID_PRE_TOKENIZER
            + ", \"post_processor\": null, \"model\": "
            + VALID_MODEL
            + ", \"added_tokens\": []}";
    Path path = tempDir.resolve("missing-decoder.tokenizer.json");
    assertDoesNotThrow(() -> Files.writeString(path, json));
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void nullDecoderThrows() {
    Path path = writeTokenizerJson("null", VALID_PRE_TOKENIZER, VALID_MODEL, "[]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void unsupportedDecoderTypeThrows() {
    Path path =
        writeTokenizerJson("{\"type\": \"Metaspace\"}", VALID_PRE_TOKENIZER, VALID_MODEL, "[]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void missingSplitBehaviorThrowsDistinctlyFromAWrongBehavior() {
    // behavior is required by HF's own serde, not defaulted to "Isolated" like invert/use_regex
    // are to false (PR #14 review round 4, finding 8) -- an absent behavior must still throw, not
    // silently pass by coincidentally comparing null via asString(null) to a non-null literal.
    String preTokenizer =
        "{\"type\": \"Sequence\", \"pretokenizers\": ["
            + "{\"type\": \"Split\", \"pattern\": {\"Regex\": \"\\\\w+\"}, \"invert\": false},"
            + "{\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true,"
            + " \"use_regex\": false}]}";
    Path path = writeTokenizerJson(VALID_DECODER, preTokenizer, VALID_MODEL, "[]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void wrongSplitBehaviorThrows() {
    String preTokenizer =
        "{\"type\": \"Sequence\", \"pretokenizers\": [{\"type\": \"Split\", \"pattern\":"
            + " {\"Regex\": \"\\\\w+\"}, \"behavior\": \"Removed\", \"invert\": false},{\"type\":"
            + " \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true, \"use_regex\":"
            + " false}]}";
    Path path = writeTokenizerJson(VALID_DECODER, preTokenizer, VALID_MODEL, "[]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void byteFallbackTrueThrows() {
    Path path = validTokenizerJsonWith("byte_fallback", "true");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void positiveDropoutThrows() {
    Path path = validTokenizerJsonWith("dropout", "0.1");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void zeroDropoutDoesNotThrow() {
    // 0.0 is semantically identical to null/absent (no dropout applied) and must load cleanly,
    // unlike a genuine positive value (PR #14 review round 5, finding 6).
    Path path = validTokenizerJsonWith("dropout", "0.0");
    assertDoesNotThrow(() -> TokenizerJsonLoader.load(path));
  }

  @Test
  void nullDropoutDoesNotThrow() {
    Path path = validTokenizerJsonWith("dropout", "null");
    assertDoesNotThrow(() -> TokenizerJsonLoader.load(path));
  }

  @Test
  void negativeDropoutThrows() {
    Path path = validTokenizerJsonWith("dropout", "-1.0");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void nonNumericObjectDropoutThrows() {
    // dropout.asDouble(0.0) silently returns its default 0.0 for a node it can't coerce to a
    // number at all, which used to make an object value pass the same check as a genuine "0.0"
    // (PR #14 review round 6, finding 3, correcting round 5's own dropout fix).
    Path path = validTokenizerJsonWith("dropout", "{\"p\": 0.5}");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void nonNumericStringDropoutThrows() {
    Path path = validTokenizerJsonWith("dropout", "\"half\"");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void nonNumericArrayDropoutThrows() {
    Path path = validTokenizerJsonWith("dropout", "[0.5]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void unkTokenThrows() {
    Path path = validTokenizerJsonWith("unk_token", "\"<unk>\"");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void continuingSubwordPrefixThrows() {
    Path path = validTokenizerJsonWith("continuing_subword_prefix", "\"##\"");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void nullContinuingSubwordPrefixDoesNotThrow() {
    // Jackson's NullNode.asString("") returns "" for an explicit JSON null, matching the
    // absent-field case -- must not be falsely rejected (PR #14 review round 4 -- verified by
    // hand at the time; now covered here per round 5, finding 4).
    Path path = validTokenizerJsonWith("continuing_subword_prefix", "null");
    assertDoesNotThrow(() -> TokenizerJsonLoader.load(path));
  }

  @Test
  void endOfWordSuffixThrows() {
    Path path = validTokenizerJsonWith("end_of_word_suffix", "\"</w>\"");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  private Path validTokenizerJsonWithAddedToken(String flagsJson) {
    String addedTokens =
        "[{\"id\": 2, \"content\": \"<x>\", \"special\": false, " + flagsJson + "}]";
    return writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
  }

  @Test
  void addedTokenLstripThrows() {
    Path path = validTokenizerJsonWithAddedToken("\"lstrip\": true");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenRstripThrows() {
    Path path = validTokenizerJsonWithAddedToken("\"rstrip\": true");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenSingleWordThrows() {
    Path path = validTokenizerJsonWithAddedToken("\"single_word\": true");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNormalizedTrueThrowsWhenANormalizerExists() {
    String addedTokens =
        "[{\"id\": 2, \"content\": \"<x>\", \"special\": false, \"normalized\": true}]";
    Path path =
        writeTokenizerJsonWithNormalizer(
            "{\"type\": \"NFC\"}", VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNormalizedTrueDoesNotThrowWhenThereIsNoNormalizer() {
    // normalized only matters when a normalizer is actually configured -- with normalizer:
    // null/NONE (true of the entire Llama-3 family), HF never routes an added token through
    // normalization regardless of this flag, so it is not a real divergence from
    // AddedTokenSplitter's un-implemented normalization (PR #14 review round 5, finding 5).
    Path path = validTokenizerJsonWithAddedToken("\"normalized\": true");
    assertDoesNotThrow(() -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNormalizedAbsentDefaultsToTrueForNonSpecialTokenAndStillThrows() {
    // HF's own serde defaults an absent `normalized` to !special, not uniformly to false -- so a
    // non-special token that omits normalized (special: false, no explicit normalized field) must
    // still be rejected when a normalizer exists, matching what HF would actually do (PR #14
    // review round 5, finding 5).
    String addedTokens = "[{\"id\": 2, \"content\": \"<x>\", \"special\": false}]";
    Path path =
        writeTokenizerJsonWithNormalizer(
            "{\"type\": \"NFC\"}", VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNormalizedAbsentDefaultsToFalseForASpecialTokenEvenWithANormalizer() {
    String addedTokens = "[{\"id\": 2, \"content\": \"<x>\", \"special\": true}]";
    Path path =
        writeTokenizerJsonWithNormalizer(
            "{\"type\": \"NFC\"}", VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertDoesNotThrow(() -> TokenizerJsonLoader.load(path));
  }

  private Path writeTokenizerJsonWithNormalizer(
      String normalizer, String decoder, String preTokenizer, String model, String addedTokens) {
    String json =
        "{"
            + "\"normalizer\": "
            + normalizer
            + ","
            + "\"pre_tokenizer\": "
            + preTokenizer
            + ","
            + "\"post_processor\": null,"
            + "\"decoder\": "
            + decoder
            + ","
            + "\"model\": "
            + model
            + ","
            + "\"added_tokens\": "
            + addedTokens
            + "}";
    try {
      Path path = tempDir.resolve("tokenizer.json");
      Files.writeString(path, json);
      return path;
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }
}
