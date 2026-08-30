package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ChatTemplateRendererTest {

  private String readFixture(String name) throws IOException {
    try {
      return Files.readString(Path.of(getClass().getResource(name).toURI()));
    } catch (URISyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  @Test
  void llama3TemplateWrapsMessagesWithHeaderTagsAndBosToken() throws IOException {
    String template = readFixture("llama3-instruct-chat-template.jinja");
    String result =
        ChatTemplateRenderer.render(
            template,
            List.of(Map.of("role", "user", "content", "Hello")),
            true,
            "<|begin_of_text|>",
            "<|eot_id|>",
            Map.of());
    assertTrue(result.startsWith("<|begin_of_text|>"));
    assertTrue(result.contains("<|start_header_id|>user<|end_header_id|>"));
    assertTrue(result.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"));
  }

  @Test
  void qwenTemplateInsertsDefaultSystemPromptWhenNoneProvided() throws IOException {
    String template = readFixture("qwen2.5-instruct-chat-template.jinja");
    String result =
        ChatTemplateRenderer.render(
            template,
            List.of(Map.of("role", "user", "content", "Hello")),
            true,
            null,
            "<|im_end|>",
            Map.of());
    assertTrue(result.contains("You are Qwen, created by Alibaba Cloud."));
    assertTrue(result.contains("<|im_start|>user\nHello<|im_end|>"));
    assertTrue(result.endsWith("<|im_start|>assistant\n"));
  }
}
