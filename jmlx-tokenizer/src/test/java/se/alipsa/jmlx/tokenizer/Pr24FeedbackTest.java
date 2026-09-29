package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

class Pr24FeedbackTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @TempDir Path temporaryDirectory;

  private static List<String> bpe(TokenizerDefinition.Bpe model, String text) {
    return TokenizerModels.encode(model, AlignedText.original(text)).stream()
        .map(TokenPiece::text)
        .toList();
  }

  @Test
  void bpeMergeDropsContinuingSubwordPrefixOfRightSymbol() {
    TokenizerDefinition.Bpe model =
        new TokenizerDefinition.Bpe(
            Map.of("[UNK]", 0, "a", 1, "##b", 2, "ab", 3, "##c", 4, "abc", 5),
            Map.of("a ##b", 0, "ab ##c", 1),
            "[UNK]",
            false,
            false,
            "##",
            "",
            false);
    assertEquals(List.of("abc"), bpe(model, "abc"));
    assertEquals(List.of("ab"), bpe(model, "ab"));
  }

  @Test
  void bpeByteFallbackKeepsSubwordPrefixLikeHuggingFace() {
    TokenizerDefinition.Bpe model =
        new TokenizerDefinition.Bpe(
            Map.of("[UNK]", 0, "a", 1, "<0x62>", 2, "<0x23>", 3),
            Map.of(),
            "[UNK]",
            false,
            true,
            "##",
            "",
            false);
    assertEquals(List.of("a", "<0x23>", "<0x23>", "<0x62>"), bpe(model, "ab"));
  }

  @Test
  void singleWordRejectionDoesNotFallBackToShorterAddedToken() {
    List<AddedToken> tokens =
        List.of(
            new AddedToken(10, "ab", true, false, false, false, false),
            new AddedToken(11, "a", false, false, false, false, false));
    AddedTokenMatcher matcher = new AddedTokenMatcher(tokens, false, null);
    List<AddedTokenMatcher.Segment> segments = matcher.split(AlignedText.original("xabc"));
    assertEquals(1, segments.size());
    assertEquals("xabc", segments.get(0).text().text());
    assertEquals(null, segments.get(0).token());
  }

  @Test
  void addedTokenMatcherPrefersLongestAmongSharedFirstCodePoint() {
    List<AddedToken> tokens =
        List.of(new AddedToken(1, "<a>", true), new AddedToken(2, "<a><b>", true));
    AddedTokenMatcher matcher = new AddedTokenMatcher(tokens, false, null);
    List<AddedTokenMatcher.Segment> segments = matcher.split(AlignedText.original("x<a><b><a>"));
    assertEquals(
        List.of("x", "<a><b>", "<a>"), segments.stream().map(s -> s.text().text()).toList());
    assertEquals(2, segments.get(1).token().id());
  }

  @Test
  void splitInvertTrueIsRejectedAtLoad() throws Exception {
    ObjectNode root = (ObjectNode) MAPPER.readTree(wordPieceFixture().toFile());
    root.set(
        "pre_tokenizer",
        MAPPER.readTree(
            "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\w+\"},"
                + "\"behavior\":\"Isolated\",\"invert\":true}"));
    Path path = temporaryDirectory.resolve("invert.tokenizer.json");
    MAPPER.writeValue(path.toFile(), root);
    TokenizerException e = assertThrows(TokenizerException.class, () -> HfTokenizer.fromFile(path));
    assertTrue(e.getMessage().contains("invert"), e.getMessage());
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
    List<String> deltas = new java.util.ArrayList<>();
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

  private static Path wordPieceFixture() {
    return Path.of(System.getProperty("jmlx.repository.root"))
        .resolve("tools/tokenizer-oracle/fixtures/wordpiece.tokenizer.json");
  }
}
