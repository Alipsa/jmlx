package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.Template;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Pinned family bundles checked against independently rendered Hugging Face chat references. */
class Phase63FamilyTokenizerTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final List<String> FAMILIES = List.of("mistral", "gemma", "phi3", "mixtral");

  @Test
  void chatInputBundlesMatchRecordedProvenance() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    JsonNode sources =
        MAPPER
            .readTree(root.resolve("tools/hf-reference/provenance.json").toFile())
            .required("chat_sources");
    for (String family : FAMILIES) {
      for (String filename : List.of("tokenizer.json", "tokenizer_config.json")) {
        String key = family + "/" + filename;
        byte[] bytes =
            Files.readAllBytes(
                root.resolve("jmlx-tokenizer/src/test/resources/families").resolve(key));
        String actual =
            java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        assertEquals(sources.required(key).asString(), actual, key);
      }
    }
  }

  @Test
  void familyEncodingMatchesPinnedTokenizersOracle() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    Path fixtures = root.resolve("tools/tokenizer-oracle/fixtures");
    JsonNode input = MAPPER.readTree(fixtures.resolve("phase6-3-families.input.json").toFile());
    JsonNode expected =
        MAPPER.readTree(fixtures.resolve("phase6-3-families.expected.json").toFile());
    assertEquals(FAMILIES.size(), input.required("fixtures").size());
    for (int i = 0; i < FAMILIES.size(); i++) {
      JsonNode source = input.required("fixtures").get(i);
      JsonNode reference = expected.required("fixtures").get(i);
      assertEquals(source.required("name").asString(), reference.required("name").asString());
      HfTokenizer tokenizer =
          HfTokenizer.fromFile(fixtures.resolve(source.required("tokenizer").asString()));
      for (int j = 0; j < source.required("cases").size(); j++) {
        JsonNode testCase = source.required("cases").get(j);
        JsonNode golden = reference.required("cases").get(j);
        String label =
            source.required("name").asString() + ":" + testCase.required("name").asString();
        TokenizerEncoding encoding =
            tokenizer.encode(
                testCase.required("text").asString(),
                new EncodingOptions(
                    testCase.required("addSpecialTokens").booleanValue(),
                    Truncation.disabled(),
                    Padding.disabled()));
        assertEquals(integers(golden.required("ids")), encoding.ids(), label);
        assertEquals(
            golden.required("decoded").asString(), tokenizer.decode(encoding.ids(), true), label);
      }
    }
  }

  @Test
  void familyChatMatchesHuggingFaceGolden() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    for (String family : FAMILIES) {
      Path directory = root.resolve("jmlx-tokenizer/src/test/resources/families").resolve(family);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
      JsonNode golden =
          MAPPER.readTree(
              root.resolve("tools/hf-reference/goldens/chat-" + family + ".json").toFile());
      assertEquals(family, golden.required("family").asString());
      assertTrue(tokenizer.vocabSize() > 0, family);
      assertFalse(golden.required("cases").isEmpty(), family);
      for (JsonNode testCase : golden.required("cases")) {
        String label = family + ":" + testCase.required("name").asString();
        boolean addGenerationPrompt = testCase.required("add_generation_prompt").booleanValue();
        List<Map<String, Object>> messages = messages(testCase.required("messages"));
        String rendered =
            tokenizer.renderChat(messages, ChatTemplateOptions.defaults(addGenerationPrompt));
        assertEquals(testCase.required("rendered").asString(), rendered, label);
        List<Integer> expectedIds = integers(testCase.required("ids"));
        assertEquals(expectedIds, tokenizer.encode(rendered, false), label);
        // A chat request uses OMIT: the template already owns BOS.
        int bosId = tokenizer.bosTokenId(tokenizer.metadata().bosToken().orElseThrow()).getAsInt();
        if (rendered.startsWith(tokenizer.metadata().bosToken().orElseThrow())) {
          assertEquals(bosId, expectedIds.getFirst(), label);
          assertEquals(1, expectedIds.stream().filter(id -> id == bosId).count(), label);
        }
        assertEquals(expectedIds.size() + 1, tokenizer.encode(rendered, true).size(), label);
        if (family.equals("mistral") || family.equals("mixtral")) {
          String corpusSource =
              Files.readString(
                  root.resolve(
                      "jmlx-jinja/src/test/resources/model-templates/"
                          + "mistral-7b-instruct-v0.3.jinja"));
          Template corpusTemplate = ChatTemplateRenderer.parse(corpusSource);
          assertEquals(
              testCase.required("rendered").asString(),
              ChatTemplateRenderer.render(
                  corpusTemplate,
                  messages,
                  addGenerationPrompt,
                  tokenizer.metadata().bosToken().orElseThrow(),
                  tokenizer.metadata().eosToken().orElseThrow(),
                  Map.of()),
              label);
        }
      }
    }
  }

  private static List<Map<String, Object>> messages(JsonNode nodes) {
    List<Map<String, Object>> result = new ArrayList<>();
    for (JsonNode node : nodes) {
      result.add(
          Map.of(
              "role", node.required("role").asString(),
              "content", node.required("content").asString()));
    }
    return List.copyOf(result);
  }

  private static List<Integer> integers(JsonNode nodes) {
    List<Integer> result = new ArrayList<>();
    nodes.forEach(node -> result.add(node.intValue()));
    return List.copyOf(result);
  }
}
