package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeout;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

class Phase62FeedbackTest {
  private static final String UNASSIGNED = "\u0378"; // unassigned code point
  private static final String COMBINING_ENCLOSING = "\u0488"; // combining mark
  private static final String COMPAT_IDEOGRAPH = "\uf900"; // compatibility ideograph

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void adjacentSplitMatchesStaySeparateWhenMergedWithPrevious() throws Exception {
    assertEquals(
        List.of("ba", "a", "b"),
        texts(
            PreTokenizerPipeline.apply(
                json(
                    "{\"type\":\"Split\",\"pattern\":{\"String\":\"a\"},"
                        + "\"behavior\":\"MergedWithPrevious\"}"),
                AlignedText.original("baab"))));
  }

  @Test
  void metaspaceRecognizesExistingLeadingReplacement() throws Exception {
    assertEquals(
        List.of("▁abc"),
        texts(
            PreTokenizerPipeline.apply(
                json(
                    "{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                        + "\"prepend_scheme\":\"always\",\"split\":false}"),
                AlignedText.original("▁abc"))));
    JsonNode metaspace =
        json(
            "{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                + "\"prepend_scheme\":\"always\",\"split\":false}");
    assertEquals(
        List.of("▁a"), texts(PreTokenizerPipeline.apply(metaspace, AlignedText.original(" a"))));
    assertEquals(
        List.of("▁▁a"), texts(PreTokenizerPipeline.apply(metaspace, AlignedText.original("  a"))));
    Path fixtures =
        Path.of(System.getProperty("jmlx.repository.root"), "tools/tokenizer-oracle/fixtures");
    HfTokenizer bpe = HfTokenizer.fromFile(fixtures.resolve("metaspace-bpe.tokenizer.json"));
    assertEquals(List.of(13, 1, 17), bpe.encode(" hello world", false));
    assertEquals(List.of(1, 13), bpe.encode("  hello", false));
    HfTokenizer unigram = HfTokenizer.fromFile(fixtures.resolve("unigram.tokenizer.json"));
    assertEquals(List.of(9, 10), unigram.encode(" hello world", false));
    assertEquals(List.of(9), unigram.encode("\u00a0hello", false));
  }

  @Test
  void unicodeNormalizationOfLongDecomposedTextStaysBounded() throws Exception {
    JsonNode nfc = json("{\"type\":\"NFC\"}");
    String input = "Cafe\u0301 ".repeat(10_000); // e + combining acute
    assertTimeout(
        Duration.ofSeconds(5),
        () ->
            assertEquals(
                "Café ".repeat(10_000),
                NormalizerPipeline.apply(nfc, AlignedText.original(input)).text()));
  }

  @Test
  void splitAndReplaceTreatOnigWhitespaceAsUnicodeWhiteSpace() throws Exception {
    JsonNode qwen =
        MAPPER.readTree(
            Path.of(System.getProperty("jmlx.repository.root"))
                .resolve(
                    "jmlx-tokenizer/src/test/resources/se/alipsa/jmlx/tokenizer/"
                        + "qwen2.5-0.5b-instruct.tokenizer.json")
                .toFile());
    JsonNode split = qwen.path("pre_tokenizer").path("pretokenizers").get(0);
    assertEquals(
        List.of("hi", "\u00a0", "\u00a0there"),
        texts(PreTokenizerPipeline.apply(split, AlignedText.original("hi\u00a0\u00a0there"))));
    assertEquals(
        "a_b",
        NormalizerPipeline.apply(
                json("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"\\\\s\"},\"content\":\"_\"}"),
                AlignedText.original("a\u00a0b"))
            .text());
  }

  @Test
  void whitespaceComponentsUseUnicodeAndOnigWordClasses() throws Exception {
    assertEquals(
        List.of("a", "b"),
        texts(
            PreTokenizerPipeline.apply(
                json("{\"type\":\"WhitespaceSplit\"}"), AlignedText.original("a\u00a0b"))));
    JsonNode whitespace = json("{\"type\":\"Whitespace\"}");
    assertEquals(
        List.of("e\u0301cole"), // NFD e-acute
        texts(PreTokenizerPipeline.apply(whitespace, AlignedText.original("e\u0301cole")))); // NFD
    assertEquals(
        List.of("a", "①", "b"),
        texts(PreTokenizerPipeline.apply(whitespace, AlignedText.original("a①b"))));
    assertEquals(
        List.of("aⅧb"), texts(PreTokenizerPipeline.apply(whitespace, AlignedText.original("aⅧb"))));
    assertEquals(
        List.of("he\u200dllo"),
        texts(PreTokenizerPipeline.apply(whitespace, AlignedText.original("he\u200dllo"))));
  }

  @Test
  void onigRegexClassesUseUnicodeAndPosixDigit() throws Exception {
    assertEquals(
        List.of("a", "١٢", "b"),
        texts(
            PreTokenizerPipeline.apply(
                json(
                    "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\d+\"},"
                        + "\"behavior\":\"Isolated\"}"),
                AlignedText.original("a١٢b"))));
    assertEquals(
        List.of("aéb"),
        texts(
            PreTokenizerPipeline.apply(
                json(
                    "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\w+\"},"
                        + "\"behavior\":\"Isolated\"}"),
                AlignedText.original("aéb"))));
    assertEquals(
        List.of("a", "12", "b"),
        texts(
            PreTokenizerPipeline.apply(
                json(
                    "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"[[:digit:]]+\"},"
                        + "\"behavior\":\"Isolated\"}"),
                AlignedText.original("a12b"))));
    assertEquals(
        "#",
        NormalizerPipeline.apply(
                json("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"\\\\w+\"},\"content\":\"#\"}"),
                AlignedText.original("a你b"))
            .text());
  }

  @Test
  void bertCleanAndAccentRulesMatchHf() throws Exception {
    JsonNode unclean =
        json(
            "{\"type\":\"BertNormalizer\",\"clean_text\":false,"
                + "\"handle_chinese_chars\":false,\"lowercase\":false,\"strip_accents\":false}");
    assertEquals(
        "a\tb\nc\u00a0d",
        NormalizerPipeline.apply(unclean, AlignedText.original("a\tb\nc\u00a0d")).text());
    JsonNode clean =
        json(
            "{\"type\":\"BertNormalizer\",\"handle_chinese_chars\":false,"
                + "\"lowercase\":false,\"strip_accents\":false}");
    assertEquals(
        UNASSIGNED + "z",
        NormalizerPipeline.apply(clean, AlignedText.original(UNASSIGNED + "z")).text());
    JsonNode accents =
        json(
            "{\"type\":\"BertNormalizer\",\"handle_chinese_chars\":false,"
                + "\"lowercase\":false,\"strip_accents\":true}");
    assertEquals("कार", NormalizerPipeline.apply(accents, AlignedText.original("कार")).text());
    assertEquals("अः", NormalizerPipeline.apply(accents, AlignedText.original("अः")).text());
    assertEquals(
        "a" + COMBINING_ENCLOSING,
        NormalizerPipeline.apply(accents, AlignedText.original("a" + COMBINING_ENCLOSING)).text());
  }

  @Test
  void insertedAndComposedTextKeepsSourceOffsets() throws Exception {
    AlignedText prepended =
        NormalizerPipeline.apply(
            json("{\"type\":\"Prepend\",\"prepend\":\"_\"}"), AlignedText.original("hi"));
    assertEquals(0, prepended.units().getFirst().startByte());
    assertEquals(1, prepended.units().getFirst().endByte());
    AlignedText byteLevel =
        PreTokenizerPipeline.apply(
                json("{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"use_regex\":false}"),
                AlignedText.original("hi"))
            .getFirst();
    assertEquals(0, byteLevel.units().getFirst().startByte());
    assertEquals(1, byteLevel.units().getFirst().endByte());
    AlignedText composed =
        NormalizerPipeline.apply(
            json("{\"type\":\"NFC\"}"), AlignedText.original("e\u0301")); // NFD e-acute
    assertEquals(composed.offset(), new TokenOffset(0, 1));
    AlignedText replaced =
        NormalizerPipeline.apply(
            json("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"\\\\b\"},\"content\":\"|\"}"),
            AlignedText.original("ab cd"));
    AlignedText.Unit inserted =
        replaced.units().stream().filter(u -> u.value().equals("|")).toList().get(2);
    assertEquals(new TokenOffset(3, 3), new TokenOffset(inserted.startByte(), inserted.endByte()));
  }

  @Test
  void incrementalWordPieceCleanupStaysWithinTokens() throws Exception {
    JsonNode decoder = json("{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":true}");
    Map<String, Integer> vocab = Map.of("x", 0, "'", 1, "y", 2, " do not", 3);
    assertEquals("x ' y", incremental(decoder, vocab, List.of(0, 1, 2)));
    assertEquals("x ' y", DecoderPipeline.decode(decoder, List.of("x", "'", "y")));
    assertEquals(" don't", incremental(decoder, vocab, List.of(3)));
    assertEquals(" don't", DecoderPipeline.decode(decoder, List.of(" do not")));
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

  @Test
  void bertNormalizerRemovesControlsAndSpacesCompatibilityIdeographs() throws Exception {
    JsonNode bert =
        json("{\"type\":\"BertNormalizer\",\"lowercase\":false,\"strip_accents\":false}");
    assertEquals("ab", NormalizerPipeline.apply(bert, AlignedText.original("a\u000bb")).text());
    assertEquals(
        "ab",
        NormalizerPipeline.apply(bert, AlignedText.original("a\ue000b")).text()); // private use
    assertEquals(
        "a " + COMPAT_IDEOGRAPH + " b",
        NormalizerPipeline.apply(bert, AlignedText.original("a" + COMPAT_IDEOGRAPH + "b")).text());
  }

  @Test
  void decodersPreserveTokenBoundariesAndFirstTokenRules() throws Exception {
    JsonNode metaspace =
        json("{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"always\"}");
    assertEquals("a b", DecoderPipeline.decode(metaspace, List.of("▁▁a", "▁b")));
    JsonNode wordPiece = json("{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":false}");
    assertEquals(
        "##ings hello", DecoderPipeline.decode(wordPiece, List.of("##ing", "##s", "hello")));
    JsonNode stripFallback =
        json(
            "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"Strip\",\"content\":\""
                + " \",\"start\":1,\"stop\":0},{\"type\":\"ByteFallback\"}]}");
    assertEquals(
        "abc",
        DecoderPipeline.decode(
            json("{\"type\":\"Strip\",\"content\":\" \",\"start\":1,\"stop\":0}"),
            List.of(" a", " b", " c")));
    assertEquals("AB", DecoderPipeline.decode(stripFallback, List.of(" <0x41>", " <0x42>")));
    JsonNode byteLevelStrip =
        json(
            "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteLevel\"},"
                + "{\"type\":\"Strip\",\"content\":\""
                + " \",\"start\":1,\"stop\":0}]}");
    assertEquals("a b", DecoderPipeline.decode(byteLevelStrip, List.of("Ġa", "Ġb")));
    JsonNode byteLevelReplace =
        json(
            "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteLevel\"},"
                + "{\"type\":\"Replace\",\"pattern\":{\"String\":\"ab\"},\"content\":\"X\"}]}");
    assertEquals("X", DecoderPipeline.decode(byteLevelReplace, List.of("a", "b")));
    assertEquals(
        "\ufffd\ufffda", // two U+FFFD
        DecoderPipeline.decode(
            json("{\"type\":\"ByteFallback\"}"), List.of("<0xC3>", "<0x28>", "a")));
  }

  @Test
  void incrementalDecoderMatchesFirstTokenRules() throws Exception {
    JsonNode metaspace =
        json("{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"always\"}");
    assertEquals("a b", incremental(metaspace, Map.of("▁▁a", 0, "▁b", 1), List.of(0, 1)));
    JsonNode wordPiece = json("{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":false}");
    assertEquals(
        "##ings hello",
        incremental(wordPiece, Map.of("##ing", 0, "##s", 1, "hello", 2), List.of(0, 1, 2)));
    JsonNode fallbackMetaspace =
        json(
            "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteFallback\"},"
                + "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"always\"}]}");
    assertEquals(
        "\ufffd\ufffda", // two U+FFFD
        incremental(fallbackMetaspace, Map.of("<0xC3>", 0, "<0x28>", 1, "a", 2), List.of(0, 1, 2)));
  }

  @Test
  void bpeDropsUnknownWithoutUnkAndClampsFallbackOffsets() {
    TokenizerDefinition.Bpe noUnknown =
        new TokenizerDefinition.Bpe(
            Map.of("a", 0, "b", 1), Map.of(), null, false, false, "", "", false);
    List<TokenPiece> pieces = TokenizerModels.encode(noUnknown, AlignedText.original("azb"));
    assertEquals(List.of("a", "b"), pieces.stream().map(TokenPiece::text).toList());
    assertEquals(
        List.of(new TokenOffset(0, 1), new TokenOffset(2, 3)),
        pieces.stream().map(TokenPiece::offset).toList());

    TokenizerDefinition.Bpe fallback =
        new TokenizerDefinition.Bpe(
            Map.of("a", 0, "<0x58>", 1, "<0x62>", 2, "<0x59>", 3),
            Map.of(),
            null,
            false,
            true,
            "X",
            "Y",
            false);
    List<TokenPiece> encoded = TokenizerModels.encode(fallback, AlignedText.original("ab"));
    assertEquals(
        List.of("a", "<0x58>", "<0x62>", "<0x59>"),
        encoded.stream().map(TokenPiece::text).toList());
    assertEquals(4, encoded.size());
  }

  private static String incremental(
      JsonNode decoder, Map<String, Integer> vocab, List<Integer> ids) {
    TokenizerDefinition definition =
        new TokenizerDefinition(
            null,
            null,
            List.of(),
            new TokenizerDefinition.WordPiece(vocab, vocab.keySet().iterator().next(), "##", 100),
            decoder,
            List.of(),
            EncodingOptions.unbounded(false),
            false,
            false);
    RuntimeIncrementalDecoder incremental =
        new RuntimeIncrementalDecoder(new TokenizerRuntime(definition), false, decoder);
    StringBuilder result = new StringBuilder();
    ids.forEach(id -> result.append(incremental.append(id)));
    return result.append(incremental.finish()).toString();
  }

  private static List<String> texts(List<AlignedText> values) {
    return values.stream().map(AlignedText::text).toList();
  }

  private static JsonNode json(String value) throws Exception {
    return MAPPER.readTree(value);
  }
}
