package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * {@link DecoderPipeline} behavior: token-boundary and first-token rules for the supported
 * decoders, plus {@link RuntimeIncrementalDecoder} streaming equivalence with the buffered decode.
 */
class DecoderPipelineTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

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
  void llama2DecoderChainStreamsAndMatchesBufferedDecode() throws Exception {
    JsonNode decoder =
        MAPPER.readTree(
            """
            {"type":"Sequence","decoders":[
              {"type":"Replace","pattern":{"String":"▁"},"content":" "},
              {"type":"ByteFallback"},
              {"type":"Fuse"},
              {"type":"Strip","content":" ","start":1,"stop":0}]}
            """);
    Map<String, Integer> vocab = new LinkedHashMap<>();
    List<String> tokens = List.of("▁Hello", "▁wor", "ld", "<0xC3>", "<0xA9>", "!");
    for (String token : tokens) {
      vocab.put(token, vocab.size());
    }
    TokenizerDefinition definition =
        new TokenizerDefinition(
            null,
            null,
            List.of(),
            new TokenizerDefinition.WordPiece(vocab, "▁Hello", "##", 100),
            decoder,
            List.of(),
            EncodingOptions.unbounded(false),
            false,
            false);
    TokenizerRuntime runtime = new TokenizerRuntime(definition);
    RuntimeIncrementalDecoder incremental = new RuntimeIncrementalDecoder(runtime, false, decoder);
    assertTrue(incremental.streams());
    StringBuilder streamed = new StringBuilder();
    List<String> deltas = new ArrayList<>();
    for (int id = 0; id < tokens.size(); id++) {
      String delta = incremental.append(id);
      deltas.add(delta);
      streamed.append(delta);
    }
    streamed.append(incremental.finish());
    assertEquals("Hello worldé!", streamed.toString());
    assertEquals(runtime.decode(List.of(0, 1, 2, 3, 4, 5), false), streamed.toString());
    assertEquals("Hello", deltas.get(0));
    assertFalse(Collections.frequency(deltas, "") == deltas.size());
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

  private static JsonNode json(String value) throws Exception {
    return MAPPER.readTree(value);
  }
}
