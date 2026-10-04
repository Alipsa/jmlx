package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.jinja.RenderOptions;

/**
 * The high-level {@code HfTokenizer.renderChat} path honors {@link ChatTemplateOptions}' explicit
 * render options: a pinned clock and zone for reproducible prompts, and the system clock when none
 * are supplied.
 */
class ChatTemplateRenderOptionsTest {

  @TempDir Path temporaryDirectory;

  @Test
  void renderChatHonorsThePinnedClockAndZoneFromChatTemplateOptions() throws Exception {
    HfTokenizer tokenizer = tokenizerWithDateTemplate();
    List<Map<String, Object>> messages = List.of(Map.of("role", "user", "content", "hi"));
    assertEquals(
        "2024-01-01 hi",
        tokenizer.renderChat(
            messages, new ChatTemplateOptions("", true, Map.of(), fixedClock(ZoneOffset.UTC))));
    // The zone is honored end to end: 00:30 UTC is still 2023-12-31 in Los Angeles.
    assertEquals(
        "2023-12-31 hi",
        tokenizer.renderChat(
            messages,
            new ChatTemplateOptions(
                "", true, Map.of(), fixedClock(ZoneId.of("America/Los_Angeles")))));
  }

  @Test
  void renderChatWithoutRenderOptionsFollowsTheSystemClock() throws Exception {
    HfTokenizer tokenizer = tokenizerWithDateTemplate();
    String rendered =
        tokenizer.renderChat(
            List.of(Map.of("role", "user", "content", "hi")), ChatTemplateOptions.defaults(true));
    // Format-only assertion: comparing against a separately read Year.now() would desync at a
    // year boundary; the pinned-clock test above covers the exact-value path.
    assertTrue(
        rendered.matches("\\d{4}-\\d{2}-\\d{2} hi"), "unexpected rendered date: " + rendered);
  }

  @Test
  void explicitRenderOptionsWithoutClockOrZoneStillRenderStrftimeNow() throws Exception {
    HfTokenizer tokenizer = tokenizerWithDateTemplate();
    // The reported regression: passing RenderOptions only to raise a limit used to leave
    // strftime_now without a clock (jinja requires both clock and zone at first use) and fail a
    // template that renders fine by default.
    String rendered =
        tokenizer.renderChat(
            List.of(Map.of("role", "user", "content", "hi")),
            new ChatTemplateOptions(
                "", true, Map.of(), RenderOptions.builder().maxSteps(500_000).build()));
    assertTrue(
        rendered.matches("\\d{4}-\\d{2}-\\d{2} hi"), "unexpected rendered date: " + rendered);
  }

  @Test
  void clockOnlyRenderOptionsKeepTheirPinnedClockAcrossTheZoneTopUp() {
    // A supplied clock with no zone keeps the clock's instant; the zone is topped up with the
    // system default, which is exactly what the renderer documents.
    Clock fixed = Clock.fixed(Instant.parse("2024-01-01T00:30:00Z"), ZoneOffset.UTC);
    String rendered =
        ChatTemplateRenderer.render(
            "{{ strftime_now('%Y-%m-%d') }}",
            Map.of(), RenderOptions.builder().clock(fixed).build());
    assertEquals(
        java.time.ZonedDateTime.ofInstant(
                Instant.parse("2024-01-01T00:30:00Z"), ZoneId.systemDefault())
            .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd")),
        rendered);
  }

  @Test
  void topUpPreservesLimitsAndHostFunctionsOfPartialRenderOptions() {
    String rendered =
        ChatTemplateRenderer.render(
            "{{ shout('hi') }} {{ strftime_now('%Y') }}",
            Map.of(),
            RenderOptions.builder()
                .maxSteps(500_000)
                .hostFunction("shout", arguments -> String.valueOf(arguments.get(0)).toUpperCase())
                .build());
    assertTrue(rendered.matches("HI \\d{4}"), "unexpected rendered prompt: " + rendered);
  }

  @Test
  void legacyOptionsShapesLeaveRenderOptionsNull() {
    assertNull(new ChatTemplateOptions("", false, Map.of()).renderOptions());
    assertNull(new ChatTemplateOptions("", false, Map.of(), null).renderOptions());
    assertNull(ChatTemplateOptions.defaults(true).renderOptions());
  }

  private static RenderOptions fixedClock(ZoneId zone) {
    return RenderOptions.builder()
        .clock(Clock.fixed(Instant.parse("2024-01-01T00:30:00Z"), ZoneOffset.UTC))
        .zoneId(zone)
        .build();
  }

  private HfTokenizer tokenizerWithDateTemplate() throws Exception {
    Path directory = Files.createDirectory(temporaryDirectory.resolve("dated"));
    Files.copy(wordPieceFixture(), directory.resolve("tokenizer.json"));
    Files.writeString(
        directory.resolve("chat_template.jinja"),
        "{{ strftime_now('%Y-%m-%d') }} {{ messages[0]['content'] }}");
    return HfTokenizer.fromDirectory(directory);
  }

  private static Path wordPieceFixture() {
    return Path.of(System.getProperty("jmlx.repository.root"))
        .resolve("tools/tokenizer-oracle/fixtures/wordpiece.tokenizer.json");
  }
}
