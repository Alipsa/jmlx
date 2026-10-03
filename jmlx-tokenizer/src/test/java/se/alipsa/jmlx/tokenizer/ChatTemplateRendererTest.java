package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.Template;

class ChatTemplateRendererTest {

  @Test
  void allowsParsedTemplatesToBeReused() {
    Template template = ChatTemplateRenderer.parse("{{ messages[0]['content'] }}");
    assertEquals(
        "Hello",
        ChatTemplateRenderer.render(
            template, List.of(Map.of("content", "Hello")), false, null, null, Map.of()));
  }

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
    // Byte-verbatim, not startsWith/contains/endsWith: those would pass through a whitespace bug
    // (e.g. a missing/extra blank line around the header separators) undetected.
    assertEquals(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + "Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        result);
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
    assertEquals(
        "<|im_start|>system\n"
            + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n"
            + "Hello<|im_end|>\n"
            + "<|im_start|>assistant\n",
        result);
  }

  @Test
  void explicitClockPinsAllRenderOverloads() {
    String source = "{{ strftime_now('%Y-%m-%d') }}|{{ messages[0]['content'] }}";
    Template parsed = ChatTemplateRenderer.parse(source);
    var options =
        se.alipsa.jmlx.jinja.RenderOptions.builder()
            .clock(
                java.time.Clock.fixed(
                    java.time.Instant.parse("2024-01-01T00:30:00Z"), java.time.ZoneOffset.UTC))
            .zoneId(java.time.ZoneId.of("America/Los_Angeles"))
            .build();
    List<Map<String, Object>> messages = List.of(Map.of("content", "hi"));
    Map<String, Object> context = Map.of("messages", messages);
    assertEquals("2023-12-31|hi", ChatTemplateRenderer.render(source, context, options));
    assertEquals("2023-12-31|hi", ChatTemplateRenderer.render(parsed, context, options));
    assertEquals(
        "2023-12-31|hi",
        ChatTemplateRenderer.render(source, messages, false, null, null, Map.of(), options));
    assertEquals(
        "2023-12-31|hi",
        ChatTemplateRenderer.render(parsed, messages, false, null, null, Map.of(), options));
  }

  @Test
  void defaultOptionsFollowChangedSystemZone() {
    java.util.TimeZone original = java.util.TimeZone.getDefault();
    try {
      String template = "{{ strftime_now('%Y-%m-%d %H') }}";
      java.util.TimeZone.setDefault(java.util.TimeZone.getTimeZone("GMT+09:00"));
      assertEquals(
          java.time.ZonedDateTime.now()
              .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH")),
          ChatTemplateRenderer.render(template, Map.of()));
      java.util.TimeZone.setDefault(java.util.TimeZone.getTimeZone("GMT-08:00"));
      assertEquals(
          java.time.ZonedDateTime.now()
              .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH")),
          ChatTemplateRenderer.render(template, Map.of()));
    } finally {
      java.util.TimeZone.setDefault(original);
    }
  }

  @Test
  void templatesThatDateThemselvesWithStrftimeNowRender() {
    // Llama 3.x chat templates call strftime_now("%d %b %Y"); the renderer supplies a clock.
    String template = "{{ strftime_now('%Y') }}|{{ messages[0]['content'] }}";
    List<Map<String, Object>> messages = List.of(Map.of("role", "user", "content", "hi"));
    String rendered =
        ChatTemplateRenderer.render(
            template, java.util.Map.of("messages", messages, "add_generation_prompt", false));
    assertEquals(java.time.Year.now().toString() + "|hi", rendered);
    assertEquals(
        rendered,
        ChatTemplateRenderer.render(
            ChatTemplateRenderer.parse(template),
            messages,
            false,
            "<s>",
            "</s>",
            java.util.Map.of()));
  }
}
