package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.Phi3VGeometryPlan;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Pure Phi-3.5-Vision prompt expansion against the committed Hugging Face WP3 golden and against
 * synthetic marker/tokenizer combinations. No model execution and no pixels: the expander is text-
 * and ID-space arithmetic, and the image geometries come from the processor's shared pure geometry
 * plan.
 */
class Phi3VPromptExpanderTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  // The golden's two image shapes: a 336x336 square (2x2 crop grid, 757 tokens) and a 672x336
  // wide image (1x2 grid, 457), per the reference token formula.
  private static final Phi3VGeometryPlan SQUARE =
      new Phi3VGeometryPlan(336, 336, false, 672, 672, 672, 672, 2, 2, 4, 4, 5, 757);
  private static final Phi3VGeometryPlan WIDE =
      new Phi3VGeometryPlan(672, 336, false, 672, 336, 672, 336, 1, 2, 2, 4, 5, 457);

  @Test
  void goldenExpansionsMatchThePinnedReference() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    int checked = 0;
    for (JsonNode caseNode : golden().required("cases")) {
      String caseName = caseNode.required("name").asString();
      List<Phi3VGeometryPlan> geometries = geometries(caseNode);
      for (var variant : caseNode.required("variants").properties()) {
        String name = caseName + "/" + variant.getKey();
        JsonNode value = variant.getValue();
        boolean prompt = variant.getKey().equals("add_generation_prompt_true");
        String rendered =
            tokenizer.renderChat(
                messages(caseNode.required("messages")), ChatTemplateOptions.defaults(prompt));
        String goldenRendered = value.required("rendered").asString();
        assertEquals(goldenRendered, rendered, name);
        assertEquals(value.required("rendered_sha256").asString(), sha256(rendered), name);
        JsonNode chunks = value.required("chunks");
        JsonNode chunkIds = value.required("chunk_ids");
        for (int i = 0; i < chunks.size(); i++) {
          List<Integer> ids = tokenizer.encode(chunks.get(i).asString(), true);
          int[] expected = ints(chunkIds.get(i));
          int[] actual = ids.stream().mapToInt(Integer::intValue).toArray();
          assertArrayEquals(expected, actual, name + " chunk " + i);
        }
        Phi3VPromptPlan plan = Phi3VPromptExpander.expandRendered(tokenizer, rendered, geometries);
        assertEquals(goldenRendered, plan.renderedText(), name);
        assertArrayEquals(ints(value.required("expanded_ids")), plan.expandedIds(), name);
        assertEquals(destinationNodes(value), plan.images(), name);
        checked++;
      }
    }
    assertEquals(10, checked);
  }

  @Test
  void expandFlattensImagePartsToTheGoldenPrompt() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    int checked = 0;
    for (JsonNode caseNode : golden().required("cases")) {
      String caseName = caseNode.required("name").asString();
      List<Phi3VGeometryPlan> geometries = geometries(caseNode);
      for (var variant : caseNode.required("variants").properties()) {
        String name = caseName + "/" + variant.getKey();
        JsonNode value = variant.getValue();
        boolean prompt = variant.getKey().equals("add_generation_prompt_true");
        Phi3VPromptPlan plan =
            Phi3VPromptExpander.expand(
                tokenizer,
                partMessages(caseNode.required("messages")),
                ChatTemplateOptions.defaults(prompt),
                geometries);
        assertEquals(value.required("rendered").asString(), plan.renderedText(), name);
        assertArrayEquals(ints(value.required("expanded_ids")), plan.expandedIds(), name);
        assertEquals(destinationNodes(value), plan.images(), name);
        checked++;
      }
    }
    assertEquals(10, checked);
  }

  @Test
  void expandRejectsMarkerTextWrittenInPlainText() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    ChatTemplateOptions options = ChatTemplateOptions.defaults(false);
    assertRejection(
        "plain string content must not contain image markers",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, List.of(textMessage("A" + marker(1))), options, List.of(SQUARE)));
    assertRejection(
        "text content part must not contain image markers",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, partMessage(textPart("A" + marker(1))), options, List.of(SQUARE)));
  }

  @Test
  void expandRejectsImagePartGeometryCountMismatches() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    ChatTemplateOptions options = ChatTemplateOptions.defaults(false);
    assertRejection(
        "prompt contains 2 image part(s) but 1 image(s) were provided",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer,
                partMessage(imagePart(), imagePart(), textPart("A")),
                options,
                List.of(SQUARE)));
    assertRejection(
        "prompt contains 0 image part(s) but 1 image(s) were provided",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, partMessage(textPart("A")), options, List.of(SQUARE)));
  }

  @Test
  void expandRejectsMalformedContentParts() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    ChatTemplateOptions options = ChatTemplateOptions.defaults(false);
    assertRejection(
        "content parts must be objects",
        () -> Phi3VPromptExpander.expand(tokenizer, partMessage("not-a-map"), options, List.of()));
    assertRejection(
        "unsupported content part type audio",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, partMessage(Map.of("type", "audio")), options, List.of()));
    Map<String, Object> extraKey = new LinkedHashMap<>();
    extraKey.put("type", "text");
    extraKey.put("text", "A");
    extraKey.put("extra", 1);
    assertRejection(
        "content part contains unsupported key 'extra'",
        () -> Phi3VPromptExpander.expand(tokenizer, partMessage(extraKey), options, List.of()));
    assertRejection(
        "text content parts require a 'text' string value",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, partMessage(Map.of("type", "text")), options, List.of()));
    assertRejection(
        "content list must not be empty",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer,
                List.of(Map.of("role", "user", "content", List.of())),
                options,
                List.of()));
    assertRejection(
        "message content must be text or a list of text/image parts",
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, List.of(Map.of("role", "user", "content", 42)), options, List.of()));
  }

  @Test
  void expandRenderedRejectsInvalidMarkerSequences() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    assertRejection(
        "marker index 0 is not 1-based",
        () -> Phi3VPromptExpander.expandRendered(tokenizer, "<|image_0|>", List.of(SQUARE)));
    assertRejection(
        "marker index 2 at position 0 skips 1",
        () -> Phi3VPromptExpander.expandRendered(tokenizer, marker(2), List.of(SQUARE)));
    assertRejection(
        "marker index 1 at position 1 duplicates marker 1 (marker indices must be distinct)",
        () ->
            Phi3VPromptExpander.expandRendered(
                tokenizer, marker(1) + marker(1), List.of(SQUARE, WIDE)));
    assertRejection(
        "marker index 1 at position 2 duplicates marker 1 (marker indices must be distinct)",
        () ->
            Phi3VPromptExpander.expandRendered(
                tokenizer, marker(1) + marker(2) + marker(1), List.of(SQUARE, WIDE, SQUARE)));
    assertRejection(
        "marker index 3 at position 1 skips 2",
        () ->
            Phi3VPromptExpander.expandRendered(
                tokenizer, marker(1) + marker(3), List.of(SQUARE, WIDE)));
    assertRejection(
        "marker index has more than 9 digits",
        () ->
            Phi3VPromptExpander.expandRendered(tokenizer, "<|image_1234567890|>", List.of(SQUARE)));
  }

  @Test
  void expandRenderedRejectsMarkerGeometryCountMismatches() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    assertRejection(
        "prompt contains 1 image marker(s) but 0 image(s) were provided",
        () -> Phi3VPromptExpander.expandRendered(tokenizer, "A" + marker(1), List.of()));
    assertRejection(
        "prompt contains 0 image marker(s) but 1 image(s) were provided",
        () -> Phi3VPromptExpander.expandRendered(tokenizer, "A", List.of(SQUARE)));
    assertRejection(
        "prompt contains 2 image marker(s) but 1 image(s) were provided",
        () ->
            Phi3VPromptExpander.expandRendered(tokenizer, marker(1) + marker(2), List.of(SQUARE)));
  }

  @Test
  void expandAndExpandRenderedRejectNullArguments() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    List<Phi3VGeometryPlan> none = List.of();
    List<Map<String, Object>> messages = partMessage(textPart("A"));
    ChatTemplateOptions options = ChatTemplateOptions.defaults(false);
    assertThrows(
        NullPointerException.class,
        () -> Phi3VPromptExpander.expand(null, messages, options, none));
    assertThrows(
        NullPointerException.class,
        () -> Phi3VPromptExpander.expand(tokenizer, null, options, none));
    assertThrows(
        NullPointerException.class,
        () -> Phi3VPromptExpander.expand(tokenizer, messages, null, none));
    assertThrows(
        NullPointerException.class,
        () -> Phi3VPromptExpander.expand(tokenizer, messages, options, null));
    assertThrows(
        NullPointerException.class,
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, Collections.<Map<String, Object>>singletonList(null), options, none));
    assertThrows(
        NullPointerException.class,
        () ->
            Phi3VPromptExpander.expand(
                tokenizer, messages, options, Collections.<Phi3VGeometryPlan>singletonList(null)));
    assertThrows(
        NullPointerException.class, () -> Phi3VPromptExpander.expandRendered(null, "A", none));
    assertThrows(
        NullPointerException.class,
        () -> Phi3VPromptExpander.expandRendered(tokenizer, null, none));
    assertThrows(
        NullPointerException.class, () -> Phi3VPromptExpander.expandRendered(tokenizer, "A", null));
    assertThrows(
        NullPointerException.class,
        () ->
            Phi3VPromptExpander.expandRendered(
                tokenizer, "A", Collections.<Phi3VGeometryPlan>singletonList(null)));
  }

  @Test
  void literalImageTokenIsNotAMarker() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    String literal = "<|image|>";
    assertEquals(List.of(1, 32044), tokenizer.encode(literal, true));
    String text = "A" + literal + "B";
    Phi3VPromptPlan plan = Phi3VPromptExpander.expandRendered(tokenizer, text, List.of());
    assertTrue(plan.images().isEmpty());
    assertEquals(text, plan.renderedText());
    int[] expected = tokenizer.encode(text, true).stream().mapToInt(Integer::intValue).toArray();
    assertArrayEquals(expected, plan.expandedIds());
  }

  @Test
  void emptyPromptExpandsToThePerEncodingBos() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    Phi3VPromptPlan plan = Phi3VPromptExpander.expandRendered(tokenizer, "", List.of());
    assertArrayEquals(new int[] {1}, plan.expandedIds());
    assertTrue(plan.images().isEmpty());
    assertEquals("", plan.renderedText());
  }

  @Test
  void planRejectsInvalidDestinations() {
    assertThrows(IllegalArgumentException.class, () -> new Phi3VImageDestination(-1, 2));
    assertThrows(IllegalArgumentException.class, () -> new Phi3VImageDestination(0, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VPromptPlan("t", new int[] {1, 2}, List.of(new Phi3VImageDestination(0, 3))));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new Phi3VPromptPlan(
                "t",
                new int[10],
                List.of(new Phi3VImageDestination(0, 2), new Phi3VImageDestination(1, 2))));
  }

  @Test
  void planReportsDestinationsAndCopies() {
    Phi3VImageDestination a = new Phi3VImageDestination(2, 3);
    Phi3VImageDestination b = new Phi3VImageDestination(11, 2);
    int[] ids = new int[] {1, -1, -1, -1, 2, -2, 3, 4, 5, 6, 7, -2, -2};
    Phi3VPromptPlan plan = new Phi3VPromptPlan("t", ids.clone(), List.of(a, b));
    assertEquals(13, plan.expandedLength());
    assertEquals(5, plan.totalSentinelRows());
    assertArrayEquals(new int[] {2, 3, 4, 11, 12}, plan.sentinelDestinations());
    assertEquals(13, b.sentinelEnd());
    int[] copy = plan.expandedIds();
    copy[0] = 99;
    assertEquals(1, plan.expandedIds()[0]);
    Phi3VPromptPlan same = new Phi3VPromptPlan("t", ids.clone(), List.of(a, b));
    assertEquals(plan, same);
    assertEquals(plan.hashCode(), same.hashCode());
    assertNotEquals(
        plan, new Phi3VPromptPlan("u", ids.clone(), List.of(a, b)), "different rendered text");
    int[] differentIds = ids.clone();
    differentIds[12] = -3;
    assertNotEquals(
        plan,
        new Phi3VPromptPlan("t", differentIds, List.of(a, b)),
        "different expanded IDs must differ");
    assertNotEquals(
        plan,
        new Phi3VPromptPlan("t", ids.clone(), List.of(a, new Phi3VImageDestination(11, 1))),
        "different destinations must differ");
  }

  @Test
  void verifySentinelDestinationsRejectsUnplannedOrWrongSentinels() {
    Phi3VPromptPlan wrong =
        new Phi3VPromptPlan("t", new int[] {1, 2, 3}, List.of(new Phi3VImageDestination(1, 2)));
    assertThrows(
        IllegalStateException.class, () -> Phi3VPromptExpander.verifySentinelDestinations(wrong));
    Phi3VPromptPlan extra =
        new Phi3VPromptPlan("t", new int[] {-1, -2}, List.of(new Phi3VImageDestination(0, 1)));
    assertThrows(
        IllegalStateException.class, () -> Phi3VPromptExpander.verifySentinelDestinations(extra));
  }

  @Test
  void promptTokensValidateAndComposeMarkers() {
    assertEquals("<|" + "image_1" + "|>", Phi3VPromptTokens.marker(1));
    assertEquals(-1, Phi3VPromptTokens.sentinel(1));
    assertEquals(-2, Phi3VPromptTokens.sentinel(2));
    assertThrows(IllegalArgumentException.class, () -> Phi3VPromptTokens.marker(0));
    assertThrows(IllegalArgumentException.class, () -> Phi3VPromptTokens.sentinel(0));
    assertTrue(Phi3VPromptTokens.MARKER_PATTERN.matcher(marker(123)).find());
    assertTrue(Phi3VPromptTokens.MARKER_PATTERN.matcher("A" + marker(1) + "B").find());
    assertTrue(!Phi3VPromptTokens.MARKER_PATTERN.matcher("<|image|>").find());
    assertTrue(!Phi3VPromptTokens.MARKER_PATTERN.matcher("image_1").find());
  }

  /** Composes one marker literal from parts; the source never holds a complete marker. */
  private static String marker(int index) {
    return "<|image_" + index + "|>";
  }

  private static Path goldenPath() {
    return Path.of(
            System.getProperty("jmlx.repository.root"),
            "tools",
            "hf-reference",
            "goldens",
            "phi3v-wp3-prompt.json")
        .toAbsolutePath();
  }

  private static Path fixtureDir() {
    return Path.of(
            System.getProperty("jmlx.repository.root"),
            "jmlx-models",
            "src",
            "test",
            "resources",
            "phi3v-tokens")
        .toAbsolutePath();
  }

  private static JsonNode golden() throws Exception {
    return MAPPER.readTree(goldenPath().toFile());
  }

  private static HfTokenizer fixtureTokenizer() throws Exception {
    return HfTokenizer.fromDirectory(fixtureDir());
  }

  private static int[] ints(JsonNode values) {
    int[] result = new int[values.size()];
    for (int i = 0; i < values.size(); i++) {
      result[i] = values.get(i).asInt();
    }
    return result;
  }

  private static List<Phi3VImageDestination> destinationNodes(JsonNode value) {
    List<Phi3VImageDestination> result = new ArrayList<>();
    for (JsonNode node : value.required("image_destinations")) {
      result.add(
          new Phi3VImageDestination(
              node.required("start").asInt(), node.required("count").asInt()));
    }
    return result;
  }

  private static List<Phi3VGeometryPlan> geometries(JsonNode caseNode) {
    List<Phi3VGeometryPlan> result = new ArrayList<>();
    for (JsonNode tokens : caseNode.required("num_img_tokens")) {
      result.add(geometryFor(tokens.asInt()));
    }
    return result;
  }

  private static Phi3VGeometryPlan geometryFor(int numImgTokens) {
    if (numImgTokens == SQUARE.numImgTokens()) {
      return SQUARE;
    }
    if (numImgTokens == WIDE.numImgTokens()) {
      return WIDE;
    }
    throw new IllegalStateException(
        "golden image token count " + numImgTokens + " has no geometry");
  }

  private static String sha256(String text) throws Exception {
    byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
    byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
    StringBuilder hex = new StringBuilder(digest.length * 2);
    for (byte value : digest) {
      hex.append(String.format("%02x", value));
    }
    return hex.toString();
  }

  private static void assertRejection(String expected, Executable action) {
    IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, action);
    assertTrue(
        failure.getMessage().contains(expected), "unexpected message: " + failure.getMessage());
  }

  /** The golden's messages as-is: plain string content (marker text included). */
  private static List<Map<String, Object>> messages(JsonNode node) {
    List<Map<String, Object>> result = new ArrayList<>();
    for (JsonNode message : node) {
      Map<String, Object> copy = new LinkedHashMap<>();
      for (var field : message.properties()) {
        copy.put(field.getKey(), field.getValue().asString());
      }
      result.add(copy);
    }
    return result;
  }

  /**
   * The golden's messages with image parts instead of marker text: each marker becomes an image
   * part, the surrounding text stays in text parts, so {@code expand} renders byte-identical prompt
   * text.
   */
  private static List<Map<String, Object>> partMessages(JsonNode node) {
    List<Map<String, Object>> result = new ArrayList<>();
    for (JsonNode message : node) {
      String content = message.required("content").asString();
      List<Object> parts = new ArrayList<>();
      Matcher matcher = Phi3VPromptTokens.MARKER_PATTERN.matcher(content);
      int lastEnd = 0;
      while (matcher.find()) {
        String before = content.substring(lastEnd, matcher.start());
        if (!before.isEmpty()) {
          parts.add(textPart(before));
        }
        parts.add(imagePart());
        lastEnd = matcher.end();
      }
      String after = content.substring(lastEnd);
      if (!after.isEmpty()) {
        parts.add(textPart(after));
      }
      if (parts.isEmpty()) {
        parts.add(textPart(""));
      }
      Map<String, Object> copy = new LinkedHashMap<>();
      copy.put("role", message.required("role").asString());
      copy.put("content", parts);
      result.add(copy);
    }
    return result;
  }

  private static Map<String, Object> textMessage(String text) {
    return Map.of("role", "user", "content", text);
  }

  private static List<Map<String, Object>> partMessage(Object... parts) {
    Map<String, Object> message = new LinkedHashMap<>();
    message.put("role", "user");
    message.put("content", List.of(parts));
    return List.of(message);
  }

  private static Map<String, Object> textPart(String text) {
    Map<String, Object> part = new LinkedHashMap<>();
    part.put("type", "text");
    part.put("text", text);
    return part;
  }

  private static Map<String, Object> imagePart() {
    return Map.of("type", "image");
  }
}
