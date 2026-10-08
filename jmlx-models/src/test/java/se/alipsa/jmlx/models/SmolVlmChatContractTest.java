package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Structured-chat generation requests for the pinned SmolVLM-256M bundle, verified through {@link
 * GenerationRequest#chat} against the committed Hugging Face golden. No model execution: the
 * decoder stays text-only, and this test pins the prompt side of the vision contract.
 */
class SmolVlmChatContractTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final ChatTemplateOptions OPTIONS = ChatTemplateOptions.defaults(true);

  private static HfTokenizer smolvlm() {
    return HfTokenizer.fromDirectory(
        Path.of(
            System.getProperty("jmlx.repository.root"),
            "jmlx-tokenizer",
            "src",
            "test",
            "resources",
            "families",
            "smolvlm"));
  }

  @Test
  void structuredChatRequestsMatchThePinnedGolden() throws Exception {
    HfTokenizer tokenizer = smolvlm();
    JsonNode golden =
        MAPPER.readTree(
            Path.of(
                    System.getProperty("jmlx.repository.root"),
                    "tools",
                    "hf-reference",
                    "goldens",
                    "chat-smolvlm.json")
                .toFile());
    assertEquals("smolvlm", golden.required("family").asString());
    for (JsonNode caseNode : golden.required("cases")) {
      String name = caseNode.required("name").asString();
      GenerationRequest request =
          GenerationRequest.chat(
              tokenizer,
              messages(caseNode.required("messages")),
              OPTIONS,
              GenerationConfig.greedyDefaults(1, Set.of()),
              CancellationToken.NONE);
      assertArrayEquals(ints(caseNode.required("ids")), request.promptTokenIds(), name);
      assertEquals(PromptSpecialTokens.OMIT, request.promptSpecialTokens(), name);
      assertNotNull(request.tokenizer(), name);
      assertTrue(request.images().isEmpty(), name);
    }
  }

  @Test
  void imagesAttachToTheRenderedRequestAsPositionalPlaceholders() throws Exception {
    HfTokenizer tokenizer = smolvlm();
    JsonNode golden =
        MAPPER.readTree(
            Path.of(
                    System.getProperty("jmlx.repository.root"),
                    "tools",
                    "hf-reference",
                    "goldens",
                    "chat-smolvlm.json")
                .toFile());
    JsonNode caseNode = null;
    for (JsonNode node : golden.required("cases")) {
      if (node.required("name").asString().equals("two_images_adjacent")) {
        caseNode = node;
      }
    }
    assertNotNull(caseNode);
    GenerationRequest request =
        GenerationRequest.chat(
                tokenizer,
                messages(caseNode.required("messages")),
                OPTIONS,
                GenerationConfig.greedyDefaults(1, Set.of()),
                CancellationToken.NONE)
            .withImages(List.of(new RgbImage(2, 2, new byte[12]), new RgbImage(1, 1, new byte[3])));
    assertEquals(2, request.images().size());
    assertArrayEquals(
        ints(caseNode.required("ids")), request.promptTokenIds(), "two_images_adjacent");
  }

  /** Converts the golden's message records into the map shape renderChat accepts. */
  private static List<Map<String, Object>> messages(JsonNode node) {
    List<Map<String, Object>> result = new ArrayList<>();
    for (JsonNode message : node) {
      Map<String, Object> map = new LinkedHashMap<>();
      map.put("role", message.required("role").asString());
      JsonNode content = message.required("content");
      if (content.isTextual()) {
        map.put("content", content.asString());
      } else {
        List<Map<String, Object>> parts = new ArrayList<>();
        for (JsonNode part : content) {
          Map<String, Object> partMap = new LinkedHashMap<>();
          for (var field : part.properties()) {
            partMap.put(field.getKey(), field.getValue().asString());
          }
          parts.add(partMap);
        }
        map.put("content", parts);
      }
      result.add(map);
    }
    return result;
  }

  private static int[] ints(JsonNode values) {
    List<Integer> result = new ArrayList<>();
    values.forEach(value -> result.add(value.intValue()));
    return result.stream().mapToInt(Integer::intValue).toArray();
  }
}
