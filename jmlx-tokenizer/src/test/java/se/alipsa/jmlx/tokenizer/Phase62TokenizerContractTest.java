package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeout;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

class Phase62TokenizerContractTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @TempDir Path temporaryDirectory;

  @Test
  void configuredDefaultsRemainComposableWithSpecialTokenChoice() throws Exception {
    ObjectNode root = (ObjectNode) MAPPER.readTree(wordPieceFixture().toFile());
    root.set(
        "truncation",
        MAPPER.readTree(
            """
            {"direction":"Right","max_length":5,"strategy":"LongestFirst","stride":0}
            """));
    root.set(
        "padding",
        MAPPER.readTree(
            """
            {"strategy":{"Fixed":7},"direction":"Right",
             "pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[PAD]"}
            """));
    Path tokenizerPath = temporaryDirectory.resolve("configured.json");
    MAPPER.writeValue(tokenizerPath.toFile(), root);
    HfTokenizer tokenizer = HfTokenizer.fromFile(tokenizerPath);

    TokenizerEncoding withSpecial = tokenizer.encodeWithDefaults("hello worlds! hello", true);
    TokenizerEncoding withoutSpecial = tokenizer.encodeWithDefaults("hello worlds! hello", false);

    assertEquals(List.of(2, 4, 5, 6, 3, 0, 0), withSpecial.ids());
    assertEquals(List.of(4, 5, 6, 7, 4, 0, 0), withoutSpecial.ids());
    assertEquals(List.of(1, 1, 1, 1, 1, 0, 0), withSpecial.attentionMask());
  }

  @Test
  void directoryLoadsMetadataAndReservedChatContext() throws Exception {
    Files.copy(wordPieceFixture(), temporaryDirectory.resolve("tokenizer.json"));
    Files.writeString(
        temporaryDirectory.resolve("tokenizer_config.json"),
        """
        {
          "bos_token": {"content":"[CLS]"},
          "eos_token": "[SEP]",
          "pad_token": "[PAD]",
          "unk_token": "[UNK]",
          "chat_template":
            "{{ bos_token }}{{ messages[0]['content'] }}{% if add_generation_prompt %}!{% endif %}"
        }
        """);
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(temporaryDirectory);

    assertEquals("[CLS]", tokenizer.metadata().bosToken().orElseThrow());
    assertEquals(
        "[CLS]hello!",
        tokenizer.renderChat(
            List.of(Map.of("role", "user", "content", "hello")),
            ChatTemplateOptions.defaults(true)));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ChatTemplateOptions("", false, Map.of("bos_token", "caller must not override")));
  }

  @Test
  void qwenDirectoryLoadsRootTemplateAndProducesCommittedPrompt() throws Exception {
    Path directory = Files.createDirectory(temporaryDirectory.resolve("qwen"));
    Path resources =
        repositoryRoot().resolve("jmlx-tokenizer/src/test/resources/se/alipsa/jmlx/tokenizer");
    Path tokenizerJson = resources.resolve("qwen2.5-0.5b-instruct.tokenizer.json");
    Files.copy(tokenizerJson, directory.resolve("tokenizer.json"));
    Files.copy(
        resources.resolve("qwen2.5-instruct-chat-template.jinja"),
        directory.resolve("chat_template.jinja"));
    Files.writeString(
        directory.resolve("tokenizer_config.json"),
        """
        {"eos_token":"<|im_end|>","chat_template":"this configured default is overridden"}
        """);
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    String rendered =
        tokenizer.renderChat(
            List.of(Map.of("role", "user", "content", "Hello")),
            ChatTemplateOptions.defaults(true));
    assertEquals(
        "<|im_start|>system\n"
            + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        rendered);
    assertEquals(
        HfTokenizer.fromFile(tokenizerJson).encode(rendered, false),
        tokenizer.encode(rendered, false));
    assertEquals(List.of("default"), tokenizer.metadata().chatTemplateNames());
  }

  @Test
  void llamaDirectoryLoadsConfiguredTokensAndRootTemplate() throws Exception {
    final Path directory = Files.createDirectory(temporaryDirectory.resolve("llama"));
    Path resources =
        repositoryRoot().resolve("jmlx-tokenizer/src/test/resources/se/alipsa/jmlx/tokenizer");
    ObjectNode tokenizerJson =
        (ObjectNode) MAPPER.readTree(resources.resolve("llama3-style.tokenizer.json").toFile());
    addSpecialToken(tokenizerJson, 128001, "<|eot_id|>");
    addSpecialToken(tokenizerJson, 128002, "<|start_header_id|>");
    addSpecialToken(tokenizerJson, 128003, "<|end_header_id|>");
    MAPPER.writeValue(directory.resolve("tokenizer.json").toFile(), tokenizerJson);
    Files.copy(
        resources.resolve("llama3-instruct-chat-template.jinja"),
        directory.resolve("chat_template.jinja"));
    Files.writeString(
        directory.resolve("tokenizer_config.json"),
        """
        {"bos_token":"<|begin_of_text|>","eos_token":{"content":"<|eot_id|>"}}
        """);
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    assertEquals(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + "Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        tokenizer.renderChat(
            List.of(Map.of("role", "user", "content", "Hello")),
            ChatTemplateOptions.defaults(true)));
    assertEquals("<|eot_id|>", tokenizer.metadata().eosToken().orElseThrow());
  }

  private static void addSpecialToken(ObjectNode tokenizerJson, int id, String content) {
    ObjectNode token = MAPPER.createObjectNode();
    token.put("id", id);
    token.put("content", content);
    token.put("single_word", false);
    token.put("lstrip", false);
    token.put("rstrip", false);
    token.put("normalized", false);
    token.put("special", true);
    tokenizerJson.withArray("added_tokens").add(token);
  }

  @Test
  void incrementalByteLevelDecodeMatchesFullDecodeAndOrderedIdRules() {
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(
            repositoryRoot()
                .resolve(
                    "jmlx-tokenizer/src/test/resources/se/alipsa/jmlx/tokenizer/"
                        + "qwen2.5-0.5b-instruct.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("🫠", false);
    IncrementalTokenDecoder decoder = tokenizer.newIncrementalDecoder(true);
    StringBuilder streamed = new StringBuilder();
    ids.forEach(id -> streamed.append(decoder.append(id)));
    streamed.append(decoder.finish());
    assertEquals(tokenizer.decode(ids, true), streamed.toString());
    assertEquals("🫠", streamed.toString());
    assertTrue(ids.size() > 1, "fixture must split the UTF-8 scalar across tokens");
  }

  @Test
  void templateOnlyKnownIdIsDecodedBeforeAboveBaseRangeDrop() {
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(
            repositoryRoot()
                .resolve(
                    "jmlx-tokenizer/src/test/resources/se/alipsa/jmlx/tokenizer/"
                        + "llama3-style-template-id-absent-from-vocab.tokenizer.json"));
    int id = 999999;
    String expected = tokenizer.decode(List.of(id), false);
    IncrementalTokenDecoder decoder = tokenizer.newIncrementalDecoder(false);
    assertEquals(expected, decoder.append(id) + decoder.finish());
  }

  @Test
  void unsupportedComponentsAndMalformedFallbackTokensFailWithTheirPath() throws Exception {
    ObjectNode wordPiece = (ObjectNode) MAPPER.readTree(wordPieceFixture().toFile());
    wordPiece.set(
        "normalizer",
        MAPPER.readTree("{\"type\":\"Precompiled\",\"precompiled_charsmap\":\"AA==\"}"));
    Path precompiled = temporaryDirectory.resolve("precompiled.json");
    MAPPER.writeValue(precompiled.toFile(), wordPiece);
    TokenizerException unsupported =
        assertThrows(TokenizerException.class, () -> HfTokenizer.fromFile(precompiled));
    assertTrue(unsupported.getMessage().contains("normalizer.type 'Precompiled'"));

    String malformed =
        Files.readString(
                repositoryRoot()
                    .resolve("tools/tokenizer-oracle/fixtures/metaspace-bpe.tokenizer.json"))
            .replace("<0xE2>", "<0xGG>");
    Path malformedPath = temporaryDirectory.resolve("malformed-fallback.json");
    Files.writeString(malformedPath, malformed);
    TokenizerException invalid =
        assertThrows(TokenizerException.class, () -> HfTokenizer.fromFile(malformedPath));
    assertTrue(invalid.getMessage().contains("malformed byte fallback token '<0xGG>'"));
  }

  @Test
  void unigramTrieKeepsLongInputBounded() {
    HfTokenizer tokenizer =
        HfTokenizer.fromFile(
            repositoryRoot().resolve("tools/tokenizer-oracle/fixtures/unigram.tokenizer.json"));
    assertTimeout(
        Duration.ofSeconds(5),
        () -> assertEquals(20_001, tokenizer.encode("Hello ".repeat(20_000), false).size()));
  }

  @Test
  void incrementalByteLevelDecodeMatchesOneShotLossyUtf8AcrossByteSplits() throws Exception {
    final HfTokenizer tokenizer = HfTokenizer.fromFile(writeByteVocabularyTokenizer());
    List<byte[]> samples = new java.util.ArrayList<>();
    byte[] scalar = "A🫠B".getBytes(StandardCharsets.UTF_8);
    for (int length = 0; length <= scalar.length; length++) {
      samples.add(java.util.Arrays.copyOf(scalar, length));
    }
    samples.add(new byte[] {(byte) 0xf0, 0x28, (byte) 0x8c, 0x28});
    Random random = new Random(620L);
    for (int sample = 0; sample < 100; sample++) {
      byte[] bytes = new byte[random.nextInt(32)];
      random.nextBytes(bytes);
      samples.add(bytes);
    }
    for (byte[] bytes : samples) {
      IncrementalTokenDecoder decoder = tokenizer.newIncrementalDecoder(false);
      StringBuilder actual = new StringBuilder();
      for (byte value : bytes) {
        actual.append(decoder.append(value & 0xff));
      }
      actual.append(decoder.finish());
      assertEquals(new String(bytes, StandardCharsets.UTF_8), actual.toString());
    }
  }

  private Path writeByteVocabularyTokenizer() throws Exception {
    Map<String, Integer> vocabulary = new LinkedHashMap<>();
    for (int value = 0; value < 256; value++) {
      vocabulary.put(ByteLevelCoding.encode(new byte[] {(byte) value}), value);
    }
    Path path = temporaryDirectory.resolve("byte-vocabulary.json");
    Files.writeString(
        path,
        """
        {"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
         "normalizer":null,
         "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,
                          "trim_offsets":true,"use_regex":false},
         "post_processor":null,"decoder":{"type":"ByteLevel"},
         "model":{"type":"BPE","dropout":null,"unk_token":null,
                  "continuing_subword_prefix":"","end_of_word_suffix":"",
                  "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                  "vocab":%s,"merges":[]}}
        """
            .formatted(MAPPER.writeValueAsString(vocabulary)));
    return path;
  }

  private static Path wordPieceFixture() {
    return repositoryRoot().resolve("tools/tokenizer-oracle/fixtures/wordpiece.tokenizer.json");
  }

  private static Path repositoryRoot() {
    return Path.of(
        java.util.Objects.requireNonNull(
            System.getProperty("jmlx.repository.root"),
            "jmlx.repository.root must be set by build.gradle"));
  }
}
