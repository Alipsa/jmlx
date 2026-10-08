package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HfTokenizerTest {

  @TempDir Path temporaryDirectory;

  @Test
  void qwen25GoldenVectorsMatchHuggingFaceTokenizers() {
    // tokenizer.json is the unmodified Qwen/Qwen2.5-0.5B-Instruct artifact from Hugging Face.
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("qwen2.5-0.5b-instruct.tokenizer.json"));
    assertEquals(List.of(9707, 11, 1879, 0), tokenizer.encode("Hello, world!", false));
    assertEquals(List.of(785, 3974, 13876, 38835), tokenizer.encode("The quick brown fox", false));
    assertEquals(List.of(220, 12621, 198), tokenizer.encode("  spaces\n", false));
    assertEquals(OptionalInt.empty(), tokenizer.bosTokenId());
    assertEquals(OptionalInt.of(151645), tokenizer.eosTokenId());
    assertEquals(OptionalInt.of(151643), tokenizer.eosTokenId("<|endoftext|>"));
    assertEquals(
        List.of(151645, 151643), tokenizer.eosTokenIds(List.of("<|im_end|>", "<|endoftext|>")));
    assertThrows(TokenizerException.class, () -> tokenizer.bosTokenId("not-a-token"));
    assertThrows(TokenizerException.class, () -> tokenizer.eosTokenId("not-a-token"));
    assertThrows(TokenizerException.class, () -> tokenizer.eosTokenIds(List.of("not-a-token")));
    assertEquals(151665, tokenizer.vocabSize());
  }

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
  void exposesModelServingTokenIdsAndVocabularySize() {
    HfTokenizer qwen = HfTokenizer.fromFile(fixture("qwen-style.tokenizer.json"));
    HfTokenizer llama = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    assertEquals(OptionalInt.of(19), qwen.eosTokenId());
    assertEquals(20, qwen.vocabSize());
    assertEquals(OptionalInt.of(128000), llama.bosTokenId());
    assertEquals(128001, llama.vocabSize());
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
  void templateSpecialTokenAbsentFromBaseVocabIsRegisteredSoEncodeAndDecodeAgree() {
    // The template's special_tokens.ids value (999999) has no vocabulary entry anywhere in
    // model.vocab/added_tokens (this fixture, unlike its round-3 version, genuinely omits
    // "<|begin_of_text|>" from both) -- HF's own TemplateProcessing performs no vocabulary lookup
    // on special_tokens ids, and ResolvedToken's own javadoc documents exactly this case, so
    // encode must trust it. Unlike round 3's fix, which trusted the id WITHOUT registering it,
    // decode must also resolve it back correctly instead of silently dropping it as an
    // above-vocab id (PR #14 review round 4, finding 2).
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(fixture("llama3-style-template-id-absent-from-vocab.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals(List.of(999999, 13, 16), ids);
    assertEquals("<|begin_of_text|>low the", tokenizer.decode(ids, false));
  }

  @Test
  void decodeSkipsAnIdBetweenTheRealVocabTopAndASparseTemplateIdInsteadOfThrowingAsAHole() {
    // llama3-style-template-id-absent-from-vocab.tokenizer.json registers a single, arbitrarily
    // sparse template id (999999) far above the real model.vocab's own top (16). Before this fix,
    // Vocabulary#maxKnownId (999999, since the template id is merged into the same Vocabulary)
    // was used directly as decode's above-vocab skip threshold, so every id between 17 and 999998
    // -- a legitimate "sampled outside the real vocab" case, the entire reason the skip exists --
    // was wrongly treated as an in-range hole and thrown on. HfTokenizer now tracks
    // baseVocabularyMaxKnownId (model.vocab + added_tokens only, excluding template
    // registrations) separately for this check, so an id in that gap is correctly skipped instead
    // (PR #14 review round 6, finding 2).
    //
    // 999998, not 1000000, pins the fix: 1000000 exceeds even the pre-fix threshold (999999, the
    // template id itself), so it skipped under the old, buggy code too and would not have caught a
    // regression back to it. 999998 is below that old threshold, so it only skips under the fixed
    // baseVocabularyMaxKnownId (16) (PR #14 review round 7, finding 6, correcting round 6's own
    // regression test, whose second assertion was vacuous for exactly this reason).
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(fixture("llama3-style-template-id-absent-from-vocab.tokenizer.json"));
    assertEquals("", tokenizer.decode(List.of(500), false));
    assertEquals("", tokenizer.decode(List.of(999998), false));
  }

  @Test
  void decodeStillThrowsOnAGenuineInRangeHoleBelowTheThresholdOnATemplateBearingFixture() {
    // The complementary case the test above doesn't cover (PR #14 review round 7, finding 6):
    // llama3-style-template-id-below-max-known-id.tokenizer.json has both a template registration
    // (id 500) and a real added_tokens entry (id 128000), so baseVocabularyMaxKnownId is 128000,
    // not the tiny real model.vocab's own top (16). An id like 50 -- no vocabulary entry, but well
    // below that 128000 boundary -- must still throw as a genuine hole, not be swept up by the
    // above-vocab skip, even on a fixture that also happens to carry a template token.
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(fixture("llama3-style-template-id-below-max-known-id.tokenizer.json"));
    assertThrows(TokenizerException.class, () -> tokenizer.decode(List.of(50), false));
  }

  @Test
  void templateSpecialTokenBelowMaxKnownIdIsRegisteredInsteadOfThrowingAsAnInRangeHole() {
    // The mirror failure mode of the test above: here the template's id (500) is unclaimed, but
    // is numerically below maxKnownId (128000, from an unrelated added token) -- so before this
    // id gets registered into the vocabulary, decode would treat it as a genuine in-range hole
    // and throw, even though encode just accepted it (PR #14 review round 4, finding 2).
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(fixture("llama3-style-template-id-below-max-known-id.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals(List.of(500, 13, 16), ids);
    assertEquals("<|begin_of_text|>low the", tokenizer.decode(ids, false));
  }

  @Test
  void loadingThrowsWhenATemplateSpecialTokenIdConflictsWithADifferentVocabularyToken() {
    // The template's special_tokens.ids value (15) is a real, existing vocabulary id -- but it
    // belongs to "the", not "<|begin_of_text|>". This is a genuine internal contradiction within
    // the same file, and baking it in would silently swap in the wrong token wherever this id is
    // later decoded. Caught at load time (HfTokenizer's constructor validates every template
    // token up front), not deferred to the first encode() call (PR #14 review, finding 2).
    Path path = fixture("llama3-style-template-id-conflicts-with-vocab.tokenizer.json");
    assertThrows(TokenizerException.class, () -> HfTokenizer.fromFile(path));
  }

  @Test
  void templateSpecialTokenMatchingAnExistingNonSpecialAddedTokenKeepsItsRealSpecialFlag() {
    // The template's special_tokens entry names an (id, text) pair that already exists as a
    // *non-special* added_tokens entry (special: false) -- not merely absent from the vocabulary
    // like the two tests above. Unconditionally re-registering it as a brand-new AddedToken with
    // special hardcoded to true would make skipSpecialTokens=true wrongly drop it from decode,
    // even though added_tokens itself says it is not special (PR #14 review round 5, finding 1).
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(
            fixture("llama3-style-template-token-matches-existing-added-token.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals(List.of(999999, 13, 16), ids);
    assertEquals("<|begin_of_text|>low the", tokenizer.decode(ids, false));
    assertEquals("<|begin_of_text|>low the", tokenizer.decode(ids, true));
  }

  @Test
  void templateSpecialTokenMatchingAnExistingPlainModelVocabEntryKeepsItsRealSpecialFlag() {
    // The mirror of the added_tokens variant above, for a template token matching a *plain*
    // model.vocab entry instead -- id 15/"the" is an ordinary vocab entry, never touched by the
    // added_tokens collision machinery at all, so it must never become special. Regressing this
    // (unconditionally promoting every template-named token to special=true) would make
    // skipSpecialTokens=true wrongly drop "the" from decode, distinguishably: "thelow the" (fixed)
    // vs "low the" (regressed) (PR #14 review round 6, finding 4, covering the plain-model.vocab
    // half of round 5 finding 1 that the added_tokens-only fixture above didn't exercise).
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(
            fixture(
                "llama3-style-template-token-matches-existing-model-vocab-entry.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals(List.of(15, 13, 16), ids);
    assertEquals("thelow the", tokenizer.decode(ids, false));
    assertEquals("thelow the", tokenizer.decode(ids, true));
  }

  @Test
  void loadingThrowsWhenTwoTemplateSpecialTokensShareTextWithDifferentIds() {
    // Neither entry conflicts with baseVocabulary on its own (id 50/60 and text "<s>" are all
    // absent from model.vocab/added_tokens) -- the contradiction is only visible by comparing the
    // two TemplateProcessing special_tokens entries against each other, which
    // requireNoTemplateVocabularyConflicts alone (round 4, finding 2) does not do (PR #14 review
    // round 5, finding 2).
    Path path =
        fixture("llama3-style-two-template-tokens-share-text-with-different-ids.tokenizer.json");
    assertThrows(TokenizerException.class, () -> HfTokenizer.fromFile(path));
  }

  @Test
  void loadingThrowsWhenTwoTemplateSpecialTokensShareIdWithDifferentText() {
    // The mirror of the test above: two entries both claim id 50, for two different texts ("<a>"
    // and "<b>"), neither of which conflicts with baseVocabulary alone (PR #14 review round 5,
    // finding 2).
    Path path =
        fixture("llama3-style-two-template-tokens-share-id-with-different-text.tokenizer.json");
    assertThrows(TokenizerException.class, () -> HfTokenizer.fromFile(path));
  }

  @Test
  void loadingThrowsWhenATemplateSpecialTokenTextIsAlreadyKnownUnderADifferentId() {
    // The mirror of the test above: here the id (999999) has no vocabulary entry at all, but the
    // TEXT it's declared for ("<|begin_of_text|>") is already known under a different id (128000,
    // from both model.vocab and added_tokens). Round 3's check was one-directional -- it only
    // tested "does this id belong to a different token," not "does this text belong to a
    // different id" -- and missed exactly this case (PR #14 review round 4, finding 2).
    Path path = fixture("llama3-style-template-id-mirrors-a-different-id.tokenizer.json");
    assertThrows(TokenizerException.class, () -> HfTokenizer.fromFile(path));
  }

  @Test
  void additionalTemplatesMayBeSymlinksIntoTheHubBlobCache() throws Exception {
    Path snapshot = Files.createDirectory(temporaryDirectory.resolve("snapshot"));
    Path blobs = Files.createDirectory(temporaryDirectory.resolve("blobs"));
    Files.copy(wordPieceFixture(), snapshot.resolve("tokenizer.json"));
    Path blob = Files.writeString(blobs.resolve("abc123"), "TOOLS {{ messages[0].content }}");
    Path additional = Files.createDirectory(snapshot.resolve("additional_chat_templates"));
    Files.createSymbolicLink(additional.resolve("tool_use.jinja"), blob);
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(snapshot);
    assertEquals(
        "TOOLS hi",
        tokenizer.renderChat(
            List.of(Map.of("role", "user", "content", "hi")),
            new ChatTemplateOptions("tool_use", false, Map.of())));
  }

  @Test
  void legacyProcessorChatTemplateFileFillsTheDefaultWhenNoOtherSourceExists() throws Exception {
    Path directory = Files.createDirectory(temporaryDirectory.resolve("legacy-template"));
    Files.copy(wordPieceFixture(), directory.resolve("tokenizer.json"));
    writeLegacyTemplate(directory, "LEGACY {{ messages[0].content }}");
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    assertEquals(List.of("default"), tokenizer.metadata().chatTemplateNames());
    assertEquals(
        "LEGACY hi",
        tokenizer.renderChat(
            List.of(Map.of("role", "user", "content", "hi")),
            new ChatTemplateOptions("", false, Map.of())));
  }

  @Test
  void configAndStandaloneTemplatesBeatsTheLegacyProcessorChatTemplateFile() throws Exception {
    Path configWins = Files.createDirectory(temporaryDirectory.resolve("config-beats-legacy"));
    Files.copy(wordPieceFixture(), configWins.resolve("tokenizer.json"));
    Files.writeString(
        configWins.resolve("tokenizer_config.json"),
        "{\"chat_template\": \"CONFIG {{ messages[0].content }}\"}");
    writeLegacyTemplate(configWins, "LEGACY {{ messages[0].content }}");
    assertEquals(
        "CONFIG hi",
        HfTokenizer.fromDirectory(configWins)
            .renderChat(
                List.of(Map.of("role", "user", "content", "hi")),
                new ChatTemplateOptions("", false, Map.of())));

    Path jinjaWins = Files.createDirectory(temporaryDirectory.resolve("jinja-beats-legacy"));
    Files.copy(wordPieceFixture(), jinjaWins.resolve("tokenizer.json"));
    Files.writeString(jinjaWins.resolve("chat_template.jinja"), "JINJA {{ messages[0].content }}");
    writeLegacyTemplate(jinjaWins, "LEGACY {{ messages[0].content }}");
    assertEquals(
        "JINJA hi",
        HfTokenizer.fromDirectory(jinjaWins)
            .renderChat(
                List.of(Map.of("role", "user", "content", "hi")),
                new ChatTemplateOptions("", false, Map.of())));
  }

  @Test
  void aBundleWithoutAnyTemplateSourceStillHasNoTemplates() throws Exception {
    Path directory = Files.createDirectory(temporaryDirectory.resolve("no-templates"));
    Files.copy(wordPieceFixture(), directory.resolve("tokenizer.json"));
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    assertThrows(
        TokenizerException.class,
        () ->
            tokenizer.renderChat(
                List.of(Map.of("role", "user", "content", "hi")),
                new ChatTemplateOptions("", false, Map.of())));
  }

  @Test
  void aMalformedLegacyChatTemplateFileIsRejected() throws Exception {
    Path missingKey = Files.createDirectory(temporaryDirectory.resolve("legacy-missing-key"));
    Files.copy(wordPieceFixture(), missingKey.resolve("tokenizer.json"));
    Files.writeString(missingKey.resolve("chat_template.json"), "{\"other\": \"nope\"}");
    assertEquals(
        "TokenizerDirectoryLoader: chat_template.json must contain a 'chat_template' string",
        assertThrows(TokenizerException.class, () -> HfTokenizer.fromDirectory(missingKey))
            .getMessage());

    Path notAString = Files.createDirectory(temporaryDirectory.resolve("legacy-not-a-string"));
    Files.copy(wordPieceFixture(), notAString.resolve("tokenizer.json"));
    Files.writeString(notAString.resolve("chat_template.json"), "{\"chat_template\": 7}");
    assertThrows(TokenizerException.class, () -> HfTokenizer.fromDirectory(notAString));
  }

  private static void writeLegacyTemplate(Path directory, String template) throws Exception {
    Files.writeString(
        directory.resolve("chat_template.json"), "{\"chat_template\": \"" + template + "\"}");
  }

  @Test
  void nullValuesInMessagesAndContextDoNotThrowRawNullPointerException() throws Exception {
    Path directory = Files.createDirectory(temporaryDirectory.resolve("nulls"));
    Files.copy(wordPieceFixture(), directory.resolve("tokenizer.json"));
    Files.writeString(directory.resolve("chat_template.jinja"), "{{ messages[0].content }}");
    Map<String, Object> message = new LinkedHashMap<>();
    message.put("role", "assistant");
    message.put("content", "hello");
    message.put("tool_calls", null);
    Map<String, Object> context = new HashMap<>();
    context.put("tools", null);
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    assertEquals(
        "hello",
        tokenizer.renderChat(List.of(message), new ChatTemplateOptions("", false, context)));
  }

  @Test
  void inputAddedTokensAreNotMarkedAsPostProcessorSpecialTokens() {
    TokenizerDefinition definition =
        new TokenizerDefinition(
            null,
            null,
            List.of(),
            new TokenizerDefinition.WordPiece(Map.of("x", 0, "[UNK]", 1), "[UNK]", "##", 100),
            null,
            List.of(new AddedToken(2, "<A>", true)),
            EncodingOptions.unbounded(false),
            false,
            false);
    TokenizerEncoding encoding =
        new TokenizerRuntime(definition).encode("x<A>x", EncodingOptions.unbounded(false));
    assertEquals(List.of(0, 2, 0), encoding.ids());
    assertEquals(List.of(0, 0, 0), encoding.specialTokensMask());
  }

  @Test
  void fixedPaddingCanBeShorterThanTruncationLimit() {
    EncodingOptions options =
        new EncodingOptions(
            true,
            new Truncation(8, Direction.RIGHT),
            new Padding(4, Direction.RIGHT, 0, "[PAD]", 0));
    assertEquals(8, options.truncation().maxLength());
    assertEquals(4, options.padding().length());
  }

  private static Path wordPieceFixture() {
    return Path.of(System.getProperty("jmlx.repository.root"))
        .resolve("tools/tokenizer-oracle/fixtures/wordpiece.tokenizer.json");
  }
}
