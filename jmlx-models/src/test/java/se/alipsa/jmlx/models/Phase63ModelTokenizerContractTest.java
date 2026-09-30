package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

class Phase63ModelTokenizerContractTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void eachFamilyChatRequestUsesTemplateIdsWithoutDuplicateBos() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    for (String family : List.of("mistral", "gemma", "phi3", "mixtral")) {
      HfTokenizer tokenizer =
          HfTokenizer.fromDirectory(
              root.resolve("jmlx-tokenizer/src/test/resources/families").resolve(family));
      JsonNode golden =
          MAPPER.readTree(
              root.resolve("tools/hf-reference/goldens/chat-" + family + ".json").toFile());
      JsonNode first = golden.path("cases").get(0);
      List<Map<String, Object>> messages = new ArrayList<>();
      for (JsonNode message : first.path("messages")) {
        messages.add(
            Map.of(
                "role",
                message.path("role").asString(),
                "content",
                message.path("content").asString()));
      }
      GenerationRequest request =
          GenerationRequest.chat(
              tokenizer,
              messages,
              ChatTemplateOptions.defaults(first.path("add_generation_prompt").asBoolean()),
              GenerationConfig.greedyDefaults(1, Set.of()),
              CancellationToken.NONE);
      int[] expected = new int[first.path("ids").size()];
      for (int i = 0; i < expected.length; i++) {
        expected[i] = first.path("ids").get(i).asInt();
      }
      assertArrayEquals(expected, request.promptTokenIds(), family);
      assertEquals(PromptSpecialTokens.OMIT, request.promptSpecialTokens(), family);
      ArchitectureDescriptor descriptor =
          ArchitectureMappings.parse(
              MAPPER.readTree(
                  root.resolve("tools/hf-reference/goldens/checkpoints/" + family + "/config.json")
                      .toFile()));
      assertTrue(tokenizer.vocabSize() <= descriptor.dimensions().vocabSize(), family);
    }
  }
}
