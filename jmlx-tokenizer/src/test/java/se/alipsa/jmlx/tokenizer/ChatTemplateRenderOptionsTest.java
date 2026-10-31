package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
  void clockOnlyRenderOptionsKeepTheirPinnedClockZone() {
    // A supplied clock with no zone keeps the clock's own zone, not the host's default, so a
    // pinned clock renders the same prompt on every host: the same instant reads a different date
    // under a different pinned zone.
    Instant instant = Instant.parse("2024-01-01T00:30:00Z");
    String utc =
        ChatTemplateRenderer.render(
            "{{ strftime_now('%Y-%m-%d') }}",
            Map.of(), RenderOptions.builder().clock(Clock.fixed(instant, ZoneOffset.UTC)).build());
    String losAngeles =
        ChatTemplateRenderer.render(
            "{{ strftime_now('%Y-%m-%d') }}",
            Map.of(),
            RenderOptions.builder()
                .clock(Clock.fixed(instant, ZoneId.of("America/Los_Angeles")))
                .build());
    assertEquals("2024-01-01", utc);
    assertEquals("2023-12-31", losAngeles);
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

  @Test
  void extraContextIsDeepCopiedAtConstruction() {
    // A caller mutation of a nested value (a tools list) after construction must not change what
    // a later render sees: the request's retained options are part of its build-time snapshot.
    List<String> tools = new java.util.ArrayList<>(List.of("search"));
    Map<String, Object> context = new java.util.LinkedHashMap<>();
    context.put("tools", tools);
    final ChatTemplateOptions options = new ChatTemplateOptions("", false, context);
    tools.add("sabotage");
    tools.remove("search");
    context.put("tools", "sabotage");
    assertEquals(List.of("search"), options.extraContext().get("tools"));

    // Arrays are copied into unmodifiable lists (the renderer converts both arrays and lists to
    // the same array value), so a caller mutation of the original array must not reach a later
    // render, and the retained copy cannot itself be mutated through the accessor.
    String[] categories = {"weather", "news"};
    Map<String, Object> arrayContext = new java.util.LinkedHashMap<>();
    arrayContext.put("categories", categories);
    ChatTemplateOptions arrayOptions = new ChatTemplateOptions("", false, arrayContext);
    categories[0] = "sabotage";
    assertEquals(List.of("weather", "news"), arrayOptions.extraContext().get("categories"));
    assertThrows(
        UnsupportedOperationException.class,
        () -> ((List<String>) arrayOptions.extraContext().get("categories")).add("sabotage"));
  }

  @Test
  void concretelyTypedObjectArrayIsCopiedWithoutArrayStoreException() {
    // A concretely typed object array (an ArrayList[]) is a context value the renderer accepts
    // (it converts any array), and its element copy comes back as an unmodifiable wrapper, which
    // the original element type cannot hold — so the copy converts the array to a list rather
    // than storing elements back into the typed array (which would throw ArrayStoreException).
    java.util.ArrayList<String> inner = new java.util.ArrayList<>(List.of("a"));
    Object[] typed = new java.util.ArrayList[] {inner};
    Map<String, Object> context = new java.util.LinkedHashMap<>();
    context.put("t", typed);
    ChatTemplateOptions options = new ChatTemplateOptions("", false, context);
    assertEquals(List.of(List.of("a")), options.extraContext().get("t"));
  }

  @Test
  void cyclicContextIsRejectedWithAClearError() {
    // The recursive copy runs before the render, so without a cycle check a self-referencing
    // value would overflow the stack here; the renderer rejects cycles with a clear error, and
    // the copy must do the same rather than never reach it.
    List<Object> cycle = new java.util.ArrayList<>();
    cycle.add(cycle);
    Map<String, Object> context = new java.util.LinkedHashMap<>();
    context.put("c", cycle);
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class, () -> new ChatTemplateOptions("", false, context));
    assertTrue(error.getMessage().contains("cycle"), "unexpected: " + error.getMessage());
  }

  @Test
  void repeatedReferencesAreCopiedOnceAndShared() {
    // The renderer converts each host object only once; the copy must match that, so a value
    // referenced several times is copied once and the references stay the same object.
    List<String> shared = new java.util.ArrayList<>(List.of("x"));
    Map<String, Object> context = new java.util.LinkedHashMap<>();
    context.put("a", shared);
    context.put("b", shared);
    ChatTemplateOptions options = new ChatTemplateOptions("", false, context);
    assertSame(options.extraContext().get("a"), options.extraContext().get("b"));
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
