package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * {@link PreTokenizerPipeline} Split/Metaspace/Whitespace behavior: Onig/POSIX regex class
 * semantics, unicode whitespace handling, and where matches land relative to the previous one.
 */
class PreTokenizerPipelineTest {

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
  void splitAndReplaceTreatOnigWhitespaceAsUnicodeWhiteSpace() throws Exception {
    JsonNode split =
        MAPPER
            .readTree(
                Path.of(System.getProperty("jmlx.repository.root"))
                    .resolve(
                        "jmlx-tokenizer/src/test/resources/se/alipsa/jmlx/tokenizer/"
                            + "qwen2.5-0.5b-instruct.tokenizer.json")
                    .toFile())
            .path("pre_tokenizer")
            .path("pretokenizers")
            .get(0);
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

  private static List<String> texts(List<AlignedText> values) {
    return values.stream().map(AlignedText::text).toList();
  }

  private static JsonNode json(String value) throws Exception {
    return MAPPER.readTree(value);
  }
}
