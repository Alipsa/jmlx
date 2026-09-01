package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
    return writeTokenizerJson(decoder, preTokenizer, "null", model, addedTokens);
  }

  private Path writeTokenizerJson(
      String decoder, String preTokenizer, String postProcessor, String model, String addedTokens) {
    String json =
        "{"
            + "\"normalizer\": null,"
            + "\"pre_tokenizer\": "
            + preTokenizer
            + ","
            + "\"post_processor\": "
            + postProcessor
            + ","
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
  void preTokenizerAddPrefixSpaceMustBeBoolean() {
    String preTokenizer =
        VALID_PRE_TOKENIZER.replace(
            "\"add_prefix_space\": false", "\"add_prefix_space\": \"true\"");
    Path path = writeTokenizerJson(VALID_DECODER, preTokenizer, VALID_MODEL, "[]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
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
  void preTokenizerSplitInvertNonBooleanThrows() {
    // Same non-boolean-fails-open exposure as byte_fallback, for pre_tokenizer Split's own invert
    // guard (PR #14 review round 9, finding 3).
    String preTokenizer =
        "{\"type\": \"Sequence\", \"pretokenizers\": [{\"type\": \"Split\", \"pattern\":"
            + " {\"Regex\": \"\\\\w+\"}, \"behavior\": \"Isolated\", \"invert\": \"yes\"},"
            + "{\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true,"
            + " \"use_regex\": false}]}";
    Path path = writeTokenizerJson(VALID_DECODER, preTokenizer, VALID_MODEL, "[]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void preTokenizerByteLevelUseRegexNonBooleanThrows() {
    String preTokenizer =
        "{\"type\": \"Sequence\", \"pretokenizers\": [{\"type\": \"Split\", \"pattern\":"
            + " {\"Regex\": \"\\\\w+\"}, \"behavior\": \"Isolated\", \"invert\": false},"
            + "{\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true,"
            + " \"use_regex\": \"yes\"}]}";
    Path path = writeTokenizerJson(VALID_DECODER, preTokenizer, VALID_MODEL, "[]");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void byteFallbackTrueThrows() {
    Path path = validTokenizerJsonWith("byte_fallback", "true");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void byteFallbackNonBooleanThrows() {
    // "yes" doesn't parse as the literal boolean true, so asBoolean(false) silently returned its
    // false default and defeated this exact guard -- verified directly (PR #14 review round 9,
    // finding 3, generalizing round 8 finding 3's non-boolean rejection beyond normalized).
    Path path = validTokenizerJsonWith("byte_fallback", "\"yes\"");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void ignoreMergesNonBooleanThrows() {
    // Unlike byte_fallback, ignore_merges has no "must be false" guard to defeat -- it's a real
    // config value, so a non-boolean here would have silently flipped tokenization to whichever
    // meaning the default happens to be, rather than merely bypassing a validation (PR #14 review
    // round 9, finding 3).
    Path path = validTokenizerJsonWith("ignore_merges", "\"yes\"");
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
  void addedTokenSpecialMustBeBooleanEvenWithoutANormalizer() {
    String addedTokens = "[{\"id\": 2, \"content\": \"<x>\", \"special\": \"true\"}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
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
  void addedTokenLstripNonBooleanThrows() {
    // Same non-boolean-fails-open exposure as byte_fallback, for added_tokens' own
    // lstrip/rstrip/single_word guards (PR #14 review round 9, finding 3).
    Path path = validTokenizerJsonWithAddedToken("\"lstrip\": \"yes\"");
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNormalizedTrueThrowsWhenANormalizerExists() {
    String addedTokens =
        "[{\"id\": 2, \"content\": \"<x>\", \"special\": false, \"normalized\": true}]";
    Path path =
        writeTokenizerJsonWithNormalizer(
            "{\"type\": \"NFC\"}", VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    // The thrown message is the whole point of round 6 finding 5's fix -- pin its wording, not
    // just that some TokenizerException is thrown, or a revert to the pre-round-6 message would
    // pass this suite unnoticed (PR #14 review round 7, finding 5).
    TokenizerException e =
        assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
    assertTrue(e.getMessage().contains("normalized=true"), e.getMessage());
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
    // Pins the "absent" wording specifically, distinguishing it from the "normalized=true" case
    // above -- both throw, but naming a field the file doesn't contain would be its own defect
    // (PR #14 review round 6, finding 5; round 7, finding 5).
    TokenizerException e =
        assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
    assertTrue(e.getMessage().contains("normalized absent"), e.getMessage());
  }

  @Test
  void addedTokenNormalizedNonBooleanValueThrowsNamingTheActualValueNotTrueOrAbsent() {
    // asBoolean(default) silently returns its default for ANY node it can't coerce to a boolean,
    // not just an absent/null one -- a string like "yes" reaches the same "would throw" branch as
    // a literal "normalized": true. Since round 8 finding 3, a non-boolean value is now rejected
    // outright before the true/false coercion is even considered, so this assertion also pins that
    // the message names the actual offending value rather than a boolean literal the file never
    // wrote (PR #14 review round 7, finding 4; round 8, finding 3).
    String addedTokens =
        "[{\"id\": 2, \"content\": \"<x>\", \"special\": false, \"normalized\": \"yes\"}]";
    Path path =
        writeTokenizerJsonWithNormalizer(
            "{\"type\": \"NFC\"}", VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    TokenizerException e =
        assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
    assertTrue(e.getMessage().contains("yes"), e.getMessage());
    assertFalse(e.getMessage().contains("normalized=true"), e.getMessage());
    assertFalse(e.getMessage().contains("normalized absent"), e.getMessage());
  }

  @Test
  void addedTokenNormalizedStringFalseStillThrowsInsteadOfSilentlyBypassingValidation() {
    // "false" (a JSON string, not a boolean) coerces via asBoolean to false regardless of the
    // default, so the pre-round-8 check (!asBoolean(!special)) never fired here -- silently
    // bypassing this validation entirely for a file HF's own strictly-typed serde would itself
    // reject (PR #14 review round 8, finding 3).
    String addedTokens =
        "[{\"id\": 2, \"content\": \"<x>\", \"special\": false, \"normalized\": \"false\"}]";
    Path path =
        writeTokenizerJsonWithNormalizer(
            "{\"type\": \"NFC\"}", VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNormalizedNumericOnASpecialTokenDoesNotClaimANonSpecialDefault() {
    // A numeric 1 coerces to boolean true regardless of the special-token default, so the
    // pre-round-8 three-branch message would have wrongly said "defaults to true for a
    // non-special added token" even though special=true here -- it isn't a default, and the token
    // isn't non-special (PR #14 review round 8, finding 3).
    String addedTokens =
        "[{\"id\": 2, \"content\": \"<x>\", \"special\": true, \"normalized\": 1}]";
    Path path =
        writeTokenizerJsonWithNormalizer(
            "{\"type\": \"NFC\"}", VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    TokenizerException e =
        assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
    assertFalse(e.getMessage().contains("non-special"), e.getMessage());
  }

  @Test
  void duplicateIdInModelVocabThrows() {
    // Two different token strings sharing one model.vocab id is a genuine internal contradiction
    // Vocabulary can't represent correctly: idToToken keeps only whichever token HashMap iterates
    // last, while tokenToId keeps both, so encode(theOtherToken) and decode(thatId) would silently
    // disagree on which string the id means (PR #14 review round 7, finding 2).
    String model = "{\"type\": \"BPE\", \"vocab\": {\"a\": 0, \"b\": 0}, \"merges\": [\"a b\"]}";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, model, "[]");
    // Message-content assertion, matching the two sibling normalized-message tests above rather
    // than only asserting the exception type (PR #14 review round 8, finding 6).
    TokenizerException e =
        assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
    assertTrue(e.getMessage().contains("id 0 for both"), e.getMessage());
  }

  @Test
  void modelVocabRequiresIntegralIds() {
    for (String malformedId : new String[] {"null", "\"0\"", "{}", "100.9"}) {
      String model =
          "{\"type\": \"BPE\", \"vocab\": {\"a\": "
              + malformedId
              + ", \"b\": 1}, \"merges\": [\"a b\"]}";
      Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, model, "[]");
      TokenizerException failure =
          assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
      assertTrue(failure.getMessage().contains("model.vocab['a'] has no integral id"));
    }
  }

  @Test
  void templateProcessingRequiresIntegralIdsAndStringTokens() {
    String templateProcessing =
        "{\"type\": \"TemplateProcessing\", \"single\": [{\"Sequence\": {}}], "
            + "\"special_tokens\": {\"<S>\": {\"id\": \"<S>\", \"ids\": [%s], "
            + "\"tokens\": [%s]}}}";
    Path nonIntegralId =
        writeTokenizerJson(
            VALID_DECODER,
            VALID_PRE_TOKENIZER,
            templateProcessing.formatted("null", "\"<S>\""),
            VALID_MODEL,
            "[]");
    TokenizerException idFailure =
        assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(nonIntegralId));
    assertTrue(idFailure.getMessage().contains("special token '<S>' has a non-integral id"));

    Path nonStringToken =
        writeTokenizerJson(
            VALID_DECODER,
            VALID_PRE_TOKENIZER,
            templateProcessing.formatted("2", "null"),
            VALID_MODEL,
            "[]");
    TokenizerException tokenFailure =
        assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(nonStringToken));
    assertTrue(tokenFailure.getMessage().contains("special token '<S>' has a non-string token"));
  }

  @Test
  void addedTokenMissingContentThrows() {
    // AddedTokenSplitter compiles Pattern.quote(content) into its alternation regex; an empty or
    // absent content compiles to a pattern that matches at every position, interleaving the
    // token's id between every character of the input while decode still round-trips the
    // original text -- a passing round-trip assertion hiding garbage encode ids (PR #14 review
    // round 8, finding 1).
    String addedTokens = "[{\"id\": 2, \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenEmptyContentThrows() {
    String addedTokens = "[{\"id\": 2, \"content\": \"\", \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenMissingIdThrows() {
    // An absent id defaults via asInt() to 0, and Vocabulary's added-token collision cleanup then
    // vacates whatever real model.vocab token currently owns id 0, making it permanently
    // un-encodable with no diagnostic at either site (PR #14 review round 8, finding 1).
    String addedTokens = "[{\"content\": \"<x>\", \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNonIntegralIdThrows() {
    // 100.9 loads via asInt() as a silently truncated 100 rather than being rejected as malformed
    // (PR #14 review round 9, finding 4).
    String addedTokens = "[{\"id\": 100.9, \"content\": \"<x>\", \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenStringIdThrows() {
    // "3" (a JSON string) loads via asInt() as the coerced integer 3 rather than being rejected
    // (PR #14 review round 9, finding 4).
    String addedTokens = "[{\"id\": \"3\", \"content\": \"<x>\", \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokenNonStringContentThrows() {
    // A numeric 123 loads via asString() as the stringified "123" rather than being rejected (PR
    // #14 review round 9, finding 4).
    String addedTokens = "[{\"id\": 2, \"content\": 123, \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokensSameIdDifferentContentThrows() {
    // added_tokens is the one declaration source with no id<->content bijection check: model.vocab
    // has one (round 7, finding 2) and TemplateProcessing special tokens have one
    // (requireInternallyConsistentTemplateTokens). Verified directly: this loaded cleanly, then
    // AddedTokenSplitter (built from every entry's content) still split input text at "<a>" while
    // Vocabulary's added-token collision cleanup had already vacated it in favor of "<b>", so
    // encoding perfectly valid input threw "no vocabulary entry" for a token the file itself
    // declared (PR #14 review round 9, finding 1).
    String addedTokens =
        "[{\"id\": 100, \"content\": \"<a>\", \"special\": false},"
            + " {\"id\": 100, \"content\": \"<b>\", \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
    assertThrows(TokenizerException.class, () -> TokenizerJsonLoader.load(path));
  }

  @Test
  void addedTokensSameContentDifferentIdThrows() {
    // The mirror case: this loaded cleanly and then made decode(100) throw for an id the file
    // itself declared, once the second entry's collision vacated the first (PR #14 review round
    // 9, finding 1).
    String addedTokens =
        "[{\"id\": 100, \"content\": \"<x>\", \"special\": false},"
            + " {\"id\": 101, \"content\": \"<x>\", \"special\": false}]";
    Path path = writeTokenizerJson(VALID_DECODER, VALID_PRE_TOKENIZER, VALID_MODEL, addedTokens);
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
