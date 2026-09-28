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

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void adjacentSplitMatchesStaySeparateWhenMergedWithPrevious() throws Exception {
    assertEquals(
        List.of("ba", "a", "b"),
        texts(
            PreTokenizerPipeline.apply(
                json(
                    "{\"type\":\"Split\",\"pattern\":{\"String\":\"a\"},\"behavior\":\"MergedWithPrevious\"}"),
                AlignedText.original("baab"))));
  }

  @Test
  void metaspaceRecognizesExistingLeadingReplacement() throws Exception {
    assertEquals(
        List.of("▁abc"),
        texts(
            PreTokenizerPipeline.apply(
                json(
                    "{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"always\",\"split\":false}"),
                AlignedText.original("▁abc"))));
  }

  @Test
  void unicodeNormalizationOfLongDecomposedTextStaysBounded() throws Exception {
    JsonNode nfc = json("{\"type\":\"NFC\"}");
    String input = "Cafe\u0301 ".repeat(10_000);
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
        List.of("e\u0301cole"),
        texts(PreTokenizerPipeline.apply(whitespace, AlignedText.original("e\u0301cole"))));
    assertEquals(
        List.of("a", "①", "b"),
        texts(PreTokenizerPipeline.apply(whitespace, AlignedText.original("a①b"))));
  }

  @Test
  void bertNormalizerRemovesControlsAndSpacesCompatibilityIdeographs() throws Exception {
    JsonNode bert =
        json("{\"type\":\"BertNormalizer\",\"lowercase\":false,\"strip_accents\":false}");
    assertEquals("ab", NormalizerPipeline.apply(bert, AlignedText.original("a\u000bb")).text());
    assertEquals("ab", NormalizerPipeline.apply(bert, AlignedText.original("a\ue000b")).text());
    assertEquals(
        "a \uf900 b", NormalizerPipeline.apply(bert, AlignedText.original("a\uf900b")).text());
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
            "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteLevel\"},{\"type\":\"Strip\",\"content\":\""
                + " \",\"start\":1,\"stop\":0}]}");
    assertEquals("a b", DecoderPipeline.decode(byteLevelStrip, List.of("Ġa", "Ġb")));
    JsonNode byteLevelReplace =
        json(
            "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteLevel\"},{\"type\":\"Replace\",\"pattern\":{\"String\":\"ab\"},\"content\":\"X\"}]}");
    assertEquals("X", DecoderPipeline.decode(byteLevelReplace, List.of("a", "b")));
    assertEquals(
        "\ufffd\ufffda",
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
            "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteFallback\"},{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"always\"}]}");
    assertEquals(
        "\ufffd\ufffda",
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
