package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Structured chat content ({@code {"type": "text", "text": ...}} and {@code {"type": "image"}}
 * parts) rendered through the committed SmolVLM-256M bundle, verified against the pinned Hugging
 * Face golden. String content keeps its existing behavior: the template iterates the content, and
 * under a template like SmolVLM's that only prints structured text parts, a plain string renders as
 * nothing. The golden's {@code string_content_dropped_by_template} case pins that, so callers must
 * send text parts for text.
 */
class StructuredChatContentTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final ChatTemplateOptions OPTIONS = ChatTemplateOptions.defaults(true);

  private static HfTokenizer smolvlm() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    return HfTokenizer.fromDirectory(
        root.resolve("jmlx-tokenizer/src/test/resources/families/smolvlm"));
  }

  private static JsonNode golden() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    return MAPPER.readTree(root.resolve("tools/hf-reference/goldens/chat-smolvlm.json").toFile());
  }

  @Test
  void structuredAndStringContentMatchThePinnedGolden() throws Exception {
    HfTokenizer tokenizer = smolvlm();
    JsonNode golden = golden();
    assertEquals("smolvlm", golden.required("family").asString());
    assertTrue(golden.required("cases").size() >= 2);
    for (JsonNode caseNode : golden.required("cases")) {
      String name = caseNode.required("name").asString();
      String rendered = tokenizer.renderChat(messages(caseNode.required("messages")), OPTIONS);
      assertEquals(caseNode.required("rendered").asString(), rendered, name);
      assertEquals(ints(caseNode.required("ids")), tokenizer.encode(rendered, false), name);
    }
  }

  @Test
  void imagePartsAreOrderPreservingPlaceholders() throws Exception {
    HfTokenizer tokenizer = smolvlm();
    // Two image parts render two markers in order; the optional image value is ignored entirely.
    // The first part is an image, so the role separator has no space.
    String rendered =
        tokenizer.renderChat(
            List.of(
                Map.of(
                    "role",
                    "user",
                    "content",
                    List.of(
                        Map.of("type", "image", "image", "not fetched"),
                        Map.of("type", "text", "text", " between"),
                        Map.of("type", "image")))),
            OPTIONS);
    assertEquals("<|im_start|>User:<image> between<image><end_of_utterance>\nAssistant:", rendered);
  }

  @Test
  void rejectsMalformedStructuredContent() throws Exception {
    HfTokenizer tokenizer = smolvlm();
    assertEquals(
        "HfTokenizer.renderChat: content parts must be objects",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(Map.of("role", "user", "content", List.of("not a map"))), OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: unsupported content part type video (only text and image)",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(Map.of("role", "user", "content", List.of(Map.of("type", "video")))),
                    OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: unsupported content part type null (only text and image)",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(Map.of("role", "user", "content", List.of(Map.of("kind", "text")))),
                    OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: content part contains unsupported key 'url'",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(
                        Map.of(
                            "role",
                            "user",
                            "content",
                            List.of(Map.of("type", "image", "url", "https://example.com/a.png")))),
                    OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: content part contains unsupported key 'text'",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(
                        Map.of(
                            "role",
                            "user",
                            "content",
                            List.of(Map.of("type", "image", "text", "not a text part")))),
                    OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: text content parts require a 'text' string value",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(Map.of("role", "user", "content", List.of(Map.of("type", "text")))),
                    OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: text content parts require a 'text' string value",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(
                        Map.of(
                            "role", "user", "content", List.of(Map.of("type", "text", "text", 7)))),
                    OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: content list must not be empty",
        message(
            () ->
                tokenizer.renderChat(
                    List.of(Map.of("role", "user", "content", List.<Object>of())), OPTIONS)));
    assertEquals(
        "HfTokenizer.renderChat: message content must be text or a list of text/image parts",
        message(
            () -> tokenizer.renderChat(List.of(Map.of("role", "user", "content", 42)), OPTIONS)));
  }

  private static String message(Executable rendering) {
    return assertThrows(TokenizerException.class, rendering).getMessage();
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

  private static List<Integer> ints(JsonNode values) {
    List<Integer> result = new ArrayList<>();
    values.forEach(value -> result.add(value.intValue()));
    return result;
  }
}
