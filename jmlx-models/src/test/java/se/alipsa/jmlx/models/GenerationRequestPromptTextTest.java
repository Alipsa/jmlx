package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

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

/**
 * Prompt-text retention: tokenizer-backed requests keep the source text their IDs were encoded from
 * ({@code text} keeps the raw prompt, {@code chat} keeps the rendered template output),
 * pre-tokenized requests keep none, and every copy method carries the retained text forward.
 */
class GenerationRequestPromptTextTest {

  private static HfTokenizer mistralTokenizer() {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    return HfTokenizer.fromDirectory(
        root.resolve("jmlx-tokenizer/src/test/resources/families/mistral"));
  }

  private static GenerationConfig config() {
    return GenerationConfig.greedyDefaults(3, Set.of());
  }

  @Test
  void textRequestsRetainTheRawPrompt() {
    GenerationRequest request =
        GenerationRequest.text(
            mistralTokenizer(),
            "Say hello.",
            PromptSpecialTokens.ADD,
            config(),
            CancellationToken.NONE);
    assertEquals("Say hello.", request.promptText());
  }

  @Test
  void chatRequestsRetainTheRenderedTemplateOutput() {
    HfTokenizer tokenizer = mistralTokenizer();
    List<Map<String, Object>> messages = List.of(Map.of("role", "user", "content", "hello"));
    GenerationRequest request =
        GenerationRequest.chat(
            tokenizer,
            messages,
            ChatTemplateOptions.defaults(true),
            config(),
            CancellationToken.NONE);
    assertEquals(
        tokenizer.renderChat(messages, ChatTemplateOptions.defaults(true)), request.promptText());
  }

  @Test
  void chatRequestsDeepCopyMessagesWithoutRejectingNullsOrReorderingKeys() {
    HfTokenizer tokenizer = mistralTokenizer();
    Map<String, Object> message = new LinkedHashMap<>();
    message.put("role", "user");
    message.put("name", null);
    message.put("content", "hello");
    List<Map<String, Object>> messages = new ArrayList<>(List.of(message));
    GenerationRequest request =
        GenerationRequest.chat(
            tokenizer,
            messages,
            ChatTemplateOptions.defaults(true),
            config(),
            CancellationToken.NONE);
    // The request owns its copy: a later caller mutation changes nothing.
    message.put("role", "sabotage");
    Map<String, Object> retained = request.chatMessages().get(0);
    assertEquals("user", retained.get("role"));
    // Null values (legal in chat fields other than content) survive the copy ...
    assertNull(retained.get("name"));
    // ... and the insertion order a template iterating the message can observe is preserved
    // (Map.copyOf would reorder and reject the null; the copy must do neither).
    assertEquals(List.of("role", "name", "content"), List.copyOf(retained.keySet()));
  }

  @Test
  void preTokenizedRequestsCarryNoText() {
    GenerationRequest request =
        new GenerationRequest(new int[] {4, 5}, config(), CancellationToken.NONE);
    assertNull(request.promptText());
  }

  @Test
  void copyMethodsCarryTheRetainedText() {
    GenerationRequest text =
        GenerationRequest.text(
            mistralTokenizer(),
            "Say hello.",
            PromptSpecialTokens.OMIT,
            config(),
            CancellationToken.NONE);
    GenerationRequest withImages = text.withImages(List.of(new RgbImage(1, 1, new byte[3])));
    GenerationRequest withPolicy = text.withCachePolicy(GenerationCachePolicy.slidingWindow(8));
    GenerationRequest both = withImages.withCachePolicy(GenerationCachePolicy.full(32));

    for (GenerationRequest copy : List.of(withImages, withPolicy, both)) {
      assertEquals("Say hello.", copy.promptText());
    }
    // Pre-tokenized copies stay text-free.
    GenerationRequest preTokenized =
        new GenerationRequest(new int[] {4}, config(), CancellationToken.NONE)
            .withImages(List.of(new RgbImage(1, 1, new byte[3])))
            .withCachePolicy(GenerationCachePolicy.slidingWindow(8));
    assertNull(preTokenized.promptText());
  }
}
