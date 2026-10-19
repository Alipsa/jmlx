package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.TokenizerException;
import se.alipsa.jmlx.vision.Qwen3VlGeometryPlan;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Pure Qwen3-VL prompt expansion and three-axis position planning against the committed Hugging
 * Face WP3 golden and against synthetic marker/tokenizer combinations. No model execution and no
 * pixels: the expander and the planner are ID-space arithmetic, and the image geometries come from
 * the processor's shared pure geometry plan.
 */
class Qwen3VlPromptExpanderTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  // Synthetic marker IDs; the committed fixture tokenizer decodes them to the Qwen3-VL contract
  // strings, mirroring the pinned artifact's 151655/151656/151652/151653.
  private static final int IMAGE = 10;
  private static final int VIDEO = 11;
  private static final int START = 12;
  private static final int END = 13;

  private static Path goldenPath() {
    return Path.of(
            System.getProperty("jmlx.repository.root"),
            "tools",
            "hf-reference",
            "goldens",
            "qwen3-vl-wp3-prompt.json")
        .toAbsolutePath();
  }

  private static JsonNode golden() throws Exception {
    return MAPPER.readTree(goldenPath().toFile());
  }

  private static int[] ints(JsonNode values) {
    List<Integer> result = new ArrayList<>();
    values.forEach(value -> result.add(value.intValue()));
    return result.stream().mapToInt(Integer::intValue).toArray();
  }

  private static Qwen3VlPromptTokens tokens() {
    return new Qwen3VlPromptTokens(IMAGE, VIDEO, START, END);
  }

  private static Qwen3VlPromptTokens tokens(int video) {
    return new Qwen3VlPromptTokens(IMAGE, video, START, END);
  }

  /** A consistent synthetic geometry: LLM grid one over the given gridH x gridW, merge side. */
  private static Qwen3VlGeometryPlan geometry(int gridH, int gridW, int merge) {
    int raw = gridH * gridW;
    int merged = raw / (merge * merge);
    return new Qwen3VlGeometryPlan(gridW, gridH, gridW, gridH, 1, gridH, gridW, raw, merged, 864);
  }

  private static Qwen3VlPromptTokens goldenTokens() throws Exception {
    JsonNode node = golden().required("tokens");
    return new Qwen3VlPromptTokens(
        node.required("image").asInt(),
        node.required("video").asInt(),
        node.required("vision_start").asInt(),
        node.required("vision_end").asInt());
  }

  @Test
  void goldenExpansionsAndPositionsMatchThePinnedReference() throws Exception {
    Qwen3VlPromptTokens tokens = goldenTokens();
    for (JsonNode caseNode : golden().required("cases")) {
      String caseName = caseNode.required("name").asString();
      for (var variant : caseNode.required("variants").properties()) {
        String name = caseName + "/" + variant.getKey();
        JsonNode value = variant.getValue();
        int[] unexpanded = ints(value.required("unexpanded_ids"));
        List<Qwen3VlGeometryPlan> geometries = new ArrayList<>();
        for (JsonNode image : value.required("images")) {
          JsonNode grid = image.required("grid_thw");
          geometries.add(
              new Qwen3VlGeometryPlan(
                  image.required("input_width").asInt(),
                  image.required("input_height").asInt(),
                  image.required("resized_width").asInt(),
                  image.required("resized_height").asInt(),
                  grid.get(0).asInt(),
                  grid.get(1).asInt(),
                  grid.get(2).asInt(),
                  image.required("raw_patch_rows").asInt(),
                  image.required("merged_row_count").asInt(),
                  image.required("patch_stride").asInt()));
        }
        Qwen3VlPromptPlan plan = Qwen3VlPromptExpander.expand(unexpanded, tokens, geometries);
        assertArrayEquals(unexpanded, plan.unexpandedIds(), name);
        assertArrayEquals(ints(value.required("expanded_ids")), plan.expandedIds(), name);
        JsonNode positions = value.required("positions");
        assertArrayEquals(ints(positions.required("t")), plan.temporalPositions(), name);
        assertArrayEquals(ints(positions.required("h")), plan.heightPositions(), name);
        assertArrayEquals(ints(positions.required("w")), plan.widthPositions(), name);
        assertEquals(value.required("delta").asInt(), plan.positionDelta(), name);
      }
    }
  }

  @Test
  void singleImageCasesReproduceTheWp1ProbeNumbers() throws Exception {
    Qwen3VlPromptTokens tokens = goldenTokens();
    for (String caseName : List.of("single_image_256x256", "single_image_64x32")) {
      JsonNode caseNode = caseByName(caseName);
      for (var variant : caseNode.required("variants").properties()) {
        String name = caseName + "/" + variant.getKey();
        if (!variant.getKey().equals("add_generation_prompt_true")) {
          continue; // the WP1 probe ran the generation-prompt variant
        }
        JsonNode value = variant.getValue();
        Qwen3VlPromptPlan plan =
            Qwen3VlPromptExpander.expand(
                ints(value.required("unexpanded_ids")), tokens, geometries(value));
        if (caseName.equals("single_image_256x256")) {
          assertEquals(82, plan.expandedLength(), name);
          assertEquals(-56, plan.positionDelta(), name);
        } else {
          assertEquals(90, plan.expandedLength(), name);
          assertEquals(-60, plan.positionDelta(), name);
        }
      }
    }
  }

  private static JsonNode caseByName(String name) throws Exception {
    for (JsonNode caseNode : golden().required("cases")) {
      if (caseNode.required("name").asString().equals(name)) {
        return caseNode;
      }
    }
    throw new IllegalStateException("golden has no case named " + name);
  }

  private static List<Qwen3VlGeometryPlan> geometries(JsonNode value) {
    List<Qwen3VlGeometryPlan> geometries = new ArrayList<>();
    for (JsonNode image : value.required("images")) {
      JsonNode grid = image.required("grid_thw");
      geometries.add(
          new Qwen3VlGeometryPlan(
              image.required("input_width").asInt(),
              image.required("input_height").asInt(),
              image.required("resized_width").asInt(),
              image.required("resized_height").asInt(),
              grid.get(0).asInt(),
              grid.get(1).asInt(),
              grid.get(2).asInt(),
              image.required("raw_patch_rows").asInt(),
              image.required("merged_row_count").asInt(),
              image.required("patch_stride").asInt()));
    }
    return geometries;
  }

  @Test
  void wrapperExpandsToTheMergedFeatureCount() {
    Qwen3VlPromptPlan plan =
        Qwen3VlPromptExpander.expand(
            new int[] {1, 2, START, IMAGE, END, 3, 4}, tokens(), List.of(geometry(4, 6, 2)));
    assertArrayEquals(
        new int[] {1, 2, START, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, END, 3, 4},
        plan.expandedIds());
    assertEquals(12, plan.expandedLength());
    assertEquals(7, plan.unexpandedLength());
    assertEquals(6, plan.totalFeatureRows());
    assertEquals(List.of(new Qwen3VlImagePlacement(3, 6)), plan.images());
    assertArrayEquals(new int[] {3, 4, 5, 6, 7, 8}, plan.featureDestinations());
    // Text 0..2; the LLM 1x2x3 image block at 3..8: T constant, H 3,3,3,4,4,4, W 3,4,5,3,4,5;
    // trailing text resumes at the block maximum plus one.
    assertArrayEquals(new int[] {0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8}, plan.temporalPositions());
    assertArrayEquals(new int[] {0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8}, plan.heightPositions());
    assertArrayEquals(new int[] {0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8}, plan.widthPositions());
    // The block's largest coordinate (8) plus one minus the 12-token length: -3.
    assertEquals(-3, plan.positionDelta());
  }

  @Test
  void twoImagesWithDifferentGridsPlanTextBetweenThemAndANegativeDelta() {
    Qwen3VlPromptPlan plan =
        Qwen3VlPromptExpander.expand(
            new int[] {START, IMAGE, END, 7, START, IMAGE, END},
            tokens(),
            List.of(geometry(4, 6, 2), geometry(2, 4, 2)));
    assertEquals(13, plan.expandedLength());
    assertEquals(
        List.of(new Qwen3VlImagePlacement(1, 6), new Qwen3VlImagePlacement(10, 2)), plan.images());
    // Image 1 (LLM 1x2x3) at 1..6, the token at 7 resumes at 4, image 2 (LLM 1x1x2) at 10..11
    // resumes at 7, the trailing end token at 12 resumes at 9.
    assertArrayEquals(new int[] {0, 1, 1, 1, 1, 1, 1, 4, 5, 6, 7, 7, 9}, plan.temporalPositions());
    assertArrayEquals(new int[] {0, 1, 1, 1, 2, 2, 2, 4, 5, 6, 7, 7, 9}, plan.heightPositions());
    assertArrayEquals(new int[] {0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9}, plan.widthPositions());
    assertEquals(-3, plan.positionDelta());
    // Decode continues from the logical cache position plus the delta, never the KV length.
    assertEquals(10, plan.ropePositionAt(13));
    assertEquals(9, plan.ropePositionAt(12));
  }

  @Test
  void textOnlyPromptPlansSequentialPositionsAndZeroDelta() {
    Qwen3VlPromptPlan plan = Qwen3VlPromptExpander.expand(new int[] {1, 2, 3}, tokens(), List.of());
    assertArrayEquals(new int[] {1, 2, 3}, plan.expandedIds());
    assertArrayEquals(new int[] {0, 1, 2}, plan.temporalPositions());
    assertArrayEquals(new int[] {0, 1, 2}, plan.heightPositions());
    assertArrayEquals(new int[] {0, 1, 2}, plan.widthPositions());
    assertEquals(0, plan.positionDelta());
    assertEquals(List.of(), plan.images());
    assertEquals(0, plan.totalFeatureRows());
    assertArrayEquals(new int[0], plan.featureDestinations());
    assertEquals(0, plan.ropePositionAt(0));
  }

  @Test
  void singleTokenGridExpandsToAWrapperOfTheSameLength() {
    // grid 2x2 with merge 2 is a single merged token: the wrapper's three tokens stay three.
    Qwen3VlPromptPlan plan =
        Qwen3VlPromptExpander.expand(
            new int[] {START, IMAGE, END}, tokens(), List.of(geometry(2, 2, 2)));
    assertArrayEquals(new int[] {START, IMAGE, END}, plan.expandedIds());
    assertEquals(List.of(new Qwen3VlImagePlacement(1, 1)), plan.images());
    assertArrayEquals(new int[] {0, 1, 2}, plan.temporalPositions());
    assertArrayEquals(new int[] {0, 1, 2}, plan.heightPositions());
    assertArrayEquals(new int[] {0, 1, 2}, plan.widthPositions());
    assertEquals(0, plan.positionDelta());
  }

  @Test
  void videoTokensAreRejectedWhenDeclared() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () -> Qwen3VlPromptExpander.expand(new int[] {1, VIDEO, 2}, tokens(), List.of()));
    assertEquals(
        "prompt contains the video token (ID 11); video is not supported", failure.getMessage());
  }

  @Test
  void anUndeclaredVideoIdIsAnOrdinaryToken() {
    Qwen3VlPromptPlan plan =
        Qwen3VlPromptExpander.expand(new int[] {1, VIDEO, 2}, tokens(-1), List.of());
    assertArrayEquals(new int[] {1, VIDEO, 2}, plan.expandedIds());
  }

  @Test
  void anIncompleteWrapperIsRejected() {
    for (int[] prompt :
        new int[][] {
          {1, START}, {START, IMAGE}, {START, IMAGE, 5}, {START, 5, END}, {START, IMAGE, IMAGE, END}
        }) {
      int index = prompt[0] == START ? 0 : 1;
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () -> Qwen3VlPromptExpander.expand(prompt, tokens(), List.of(geometry(2, 2, 2))));
      assertEquals(
          "vision start token (ID 12) at index "
              + index
              + " is not the start of a complete vision start, image, vision end wrapper",
          failure.getMessage());
    }
  }

  @Test
  void nestedWrappersAreRejected() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlPromptExpander.expand(
                    new int[] {START, START, IMAGE, END}, tokens(), List.of(geometry(2, 2, 2))));
    assertTrue(
        failure
            .getMessage()
            .startsWith("vision start token (ID 12) at index 0 is not the start of a complete"));
  }

  @Test
  void bareImageTokensAreRejected() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () -> Qwen3VlPromptExpander.expand(new int[] {1, IMAGE, 2}, tokens(), List.of()));
    assertEquals(
        "bare image token (ID 10) at index 1 is not wrapped in vision start/end tokens",
        failure.getMessage());
  }

  @Test
  void consecutiveImageTokensAreRejectedAsPreExpanded() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlPromptExpander.expand(new int[] {1, IMAGE, IMAGE, 2}, tokens(), List.of()));
    assertEquals(
        "consecutive image tokens (ID 10) at indices 1 and 2 look like an already-expanded "
            + "placeholder; pass the unexpanded prompt",
        failure.getMessage());
  }

  @Test
  void anUnmatchedVisionEndIsRejected() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () -> Qwen3VlPromptExpander.expand(new int[] {1, END}, tokens(), List.of()));
    assertEquals("unmatched vision end token (ID 13) at index 1", failure.getMessage());
  }

  @Test
  void wrapperCountMismatchesAreRejected() {
    for (int wrappers : new int[] {0, 1, 2, 3}) {
      int[] prompt = new int[3 * wrappers];
      for (int i = 0; i < wrappers; i++) {
        prompt[i * 3] = START;
        prompt[i * 3 + 1] = IMAGE;
        prompt[i * 3 + 2] = END;
      }
      for (int images = 0; images <= 3; images++) {
        if (wrappers == images) {
          continue;
        }
        List<Qwen3VlGeometryPlan> geometries = new ArrayList<>();
        for (int i = 0; i < images; i++) {
          geometries.add(geometry(2, 2, 2));
        }
        IllegalArgumentException failure =
            assertThrows(
                IllegalArgumentException.class,
                () -> Qwen3VlPromptExpander.expand(prompt, tokens(), geometries));
        assertEquals(
            "prompt contains "
                + wrappers
                + " image placeholder(s) but "
                + images
                + " image(s) were provided",
            failure.getMessage());
      }
    }
  }

  @Test
  void nonDivisibleGridsAreRejectedBeforeExpansion() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlPromptExpander.expand(
                    new int[] {START, IMAGE, END}, tokens(), List.of(geometry(10, 9, 3))));
    assertEquals("image grid 10x9 is not divisible by its merge size 3", failure.getMessage());
  }

  @Test
  void anOversizedExpandedLengthIsRejected() {
    // Two 32768x32768 grids with merge 1: 6 + 2 * (2^30 - 1) tokens, past a signed 32-bit int.
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlPromptExpander.expand(
                    new int[] {START, IMAGE, END, START, IMAGE, END},
                    tokens(),
                    List.of(geometry(32768, 32768, 1), geometry(32768, 32768, 1))));
    assertEquals(
        "expanded prompt length 2147483652 overflows a signed 32-bit integer",
        failure.getMessage());
  }

  @Test
  void nullArgumentsAreRejected() {
    assertThrows(
        NullPointerException.class, () -> Qwen3VlPromptExpander.expand(null, tokens(), List.of()));
    assertThrows(
        NullPointerException.class,
        () -> Qwen3VlPromptExpander.expand(new int[] {1}, null, List.of()));
    assertThrows(
        NullPointerException.class,
        () -> Qwen3VlPromptExpander.expand(new int[] {1}, tokens(), null));
    List<Qwen3VlGeometryPlan> withNull = new ArrayList<>();
    withNull.add(geometry(2, 2, 2));
    withNull.add(null);
    assertThrows(
        NullPointerException.class,
        () -> Qwen3VlPromptExpander.expand(new int[] {1}, tokens(), withNull));
  }

  @Test
  void thePlannerRejectsOversizedImageRunsAndMismatches() {
    IllegalArgumentException oversized =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlPositionPlanner.plan(
                    new int[] {IMAGE}, tokens(), List.of(new int[] {1, 46341, 46341})));
    assertEquals(
        "image run (1, 46341, 46341) of 2147488281 tokens overflows a signed 32-bit integer",
        oversized.getMessage());

    IllegalArgumentException missing =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlPositionPlanner.plan(
                    new int[] {1, 2, 3}, tokens(), List.of(new int[] {1, 2, 2})));
    assertEquals(
        "expanded prompt holds no image token (ID 10) at or after position 0; the image grids "
            + "disagree with the expanded prompt",
        missing.getMessage());

    IllegalArgumentException malformed =
        assertThrows(
            IllegalArgumentException.class,
            () -> Qwen3VlPositionPlanner.plan(new int[] {IMAGE}, tokens(), List.of(new int[] {1})));
    assertEquals("llm grids are positive (t, h, w) triples: [1]", malformed.getMessage());

    // The image token at position 1 leaves only two tokens of room, but the grid holds four:
    // the mismatch is rejected before any array is written.
    IllegalArgumentException pastTheEnd =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlPositionPlanner.plan(
                    new int[] {1, IMAGE, 2}, tokens(), List.of(new int[] {1, 2, 2})));
    assertEquals(
        "image grid (1, 2, 2) at token 1 extends past the expanded prompt (3 tokens)",
        pastTheEnd.getMessage());
  }

  @Test
  void thePlannerHandlesTextOnlyAndEmptyPrompts() {
    Qwen3VlRopePlan textOnly =
        Qwen3VlPositionPlanner.plan(new int[] {1, 2, 3}, tokens(), List.of());
    assertEquals(
        new Qwen3VlRopePlan(new int[] {0, 1, 2}, new int[] {0, 1, 2}, new int[] {0, 1, 2}, 0),
        textOnly);
    Qwen3VlRopePlan empty = Qwen3VlPositionPlanner.plan(new int[0], tokens(), List.of());
    assertEquals(new Qwen3VlRopePlan(new int[0], new int[0], new int[0], 0), empty);
  }

  @Test
  void theTokensRecordValidatesItsIds() {
    assertEquals(new Qwen3VlPromptTokens(IMAGE, VIDEO, START, END), tokens());
    assertDoesNotThrow(() -> new Qwen3VlPromptTokens(IMAGE, -1, START, END));
    assertEquals(
        "image, vision start and vision end token IDs must be non-negative: -1, 12, 13",
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen3VlPromptTokens(-1, VIDEO, START, END))
            .getMessage());
    assertEquals(
        "videoTokenId must be -1 (no video token) or a token ID: -2",
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen3VlPromptTokens(IMAGE, -2, START, END))
            .getMessage());
    assertEquals(
        "image, vision start and vision end token IDs must be distinct: 12, 12, 13",
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen3VlPromptTokens(START, VIDEO, START, END))
            .getMessage());
    assertEquals(
        "videoTokenId must be distinct from the image markers: 10",
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen3VlPromptTokens(IMAGE, IMAGE, START, END))
            .getMessage());
  }

  private static Path fixturePath() {
    return Path.of(
            System.getProperty("jmlx.repository.root"),
            "jmlx-models",
            "src",
            "test",
            "resources",
            "qwen3vl-tokens",
            "tokenizer.json")
        .toAbsolutePath();
  }

  private static HfTokenizer fixtureTokenizer() throws Exception {
    return HfTokenizer.fromFile(fixturePath());
  }

  /** The fixture with one added token (and its model.vocab entry) removed, in {@code tempDir}. */
  private static HfTokenizer fixtureTokenizerMissingAddedToken(int id, Path tempDir)
      throws Exception {
    ObjectNode root = (ObjectNode) MAPPER.readTree(fixturePath().toFile()).deepCopy();
    String sentinel = null;
    for (JsonNode entry : root.required("added_tokens")) {
      if (entry.required("id").asInt() == id) {
        sentinel = entry.required("content").asString();
      }
    }
    if (sentinel == null) {
      throw new IllegalStateException("fixture has no added token with ID " + id);
    }
    ((ObjectNode) root.required("model").required("vocab")).remove(sentinel);
    ((ArrayNode) root.required("added_tokens"))
        .removeIf(entry -> entry.required("id").asInt() == id);
    Path path = tempDir.resolve("tokenizer.json");
    MAPPER.writeValue(path.toFile(), root);
    return HfTokenizer.fromFile(path);
  }

  @Test
  void resolveAcceptsConfigIdsAgreeingWithTheFixtureVocabulary() throws Exception {
    assertEquals(
        new Qwen3VlPromptTokens(IMAGE, VIDEO, START, END),
        Qwen3VlPromptTokens.resolve(fixtureTokenizer(), IMAGE, VIDEO, START, END));
  }

  @Test
  void resolveRejectsAConfigIdThatDisagreesWithTheVocabulary() throws Exception {
    for (String role : List.of("image", "video", "vision start", "vision end")) {
      int configImage = role.equals("image") ? 99 : IMAGE;
      int configVideo = role.equals("video") ? 99 : VIDEO;
      int configStart = role.equals("vision start") ? 99 : START;
      int configEnd = role.equals("vision end") ? 99 : END;
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  Qwen3VlPromptTokens.resolve(
                      fixtureTokenizer(), configImage, configVideo, configStart, configEnd));
      assertTrue(failure.getMessage().contains(role), role + ": " + failure.getMessage());
      assertTrue(failure.getMessage().contains("99"), role + ": " + failure.getMessage());
    }
    assertThrows(
        NullPointerException.class,
        () -> Qwen3VlPromptTokens.resolve(null, IMAGE, VIDEO, START, END));
  }

  @Test
  void resolvePropagatesTheTokenizerExceptionForAMissingSentinel(@TempDir Path tempDir)
      throws Exception {
    HfTokenizer tokenizer = fixtureTokenizerMissingAddedToken(END, tempDir);
    assertThrows(
        TokenizerException.class,
        () -> Qwen3VlPromptTokens.resolve(tokenizer, IMAGE, VIDEO, START, END));
  }

  @Test
  void resolveSkipsTheVideoLookupWhenTheConfigDeclaresNoVideoToken(@TempDir Path tempDir)
      throws Exception {
    // The config's -1 must not require the video sentinel in the vocabulary at all: it must
    // resolve through a fixture variant that has no video token added.
    HfTokenizer noVideoTokenizer = fixtureTokenizerMissingAddedToken(VIDEO, tempDir);
    Qwen3VlPromptTokens resolved =
        Qwen3VlPromptTokens.resolve(noVideoTokenizer, IMAGE, -1, START, END);
    assertEquals(new Qwen3VlPromptTokens(IMAGE, -1, START, END), resolved);
    // The -1 also resolves against the full fixture, where the video sentinel is present: the
    // config's declaration wins over the vocabulary.
    assertEquals(
        new Qwen3VlPromptTokens(IMAGE, -1, START, END),
        Qwen3VlPromptTokens.resolve(fixtureTokenizer(), IMAGE, -1, START, END));
    // The other three roles still resolve and cross-check with the config while the video
    // role is absent from it.
    for (String role : List.of("image", "vision start", "vision end")) {
      int configImage = role.equals("image") ? 99 : IMAGE;
      int configStart = role.equals("vision start") ? 99 : START;
      int configEnd = role.equals("vision end") ? 99 : END;
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  Qwen3VlPromptTokens.resolve(
                      noVideoTokenizer, configImage, -1, configStart, configEnd));
      assertTrue(failure.getMessage().contains(role), role + ": " + failure.getMessage());
      assertTrue(failure.getMessage().contains("99"), role + ": " + failure.getMessage());
    }
  }

  @Test
  void theFixtureDecodesEachSentinelIdBackToItsContractConstant() throws Exception {
    HfTokenizer tokenizer = fixtureTokenizer();
    assertEquals(Qwen3VlPromptTokens.IMAGE_TOKEN_CONTENT, tokenizer.decode(List.of(IMAGE), false));
    assertEquals(Qwen3VlPromptTokens.VIDEO_TOKEN_CONTENT, tokenizer.decode(List.of(VIDEO), false));
    assertEquals(
        Qwen3VlPromptTokens.VISION_START_TOKEN_CONTENT, tokenizer.decode(List.of(START), false));
    assertEquals(
        Qwen3VlPromptTokens.VISION_END_TOKEN_CONTENT, tokenizer.decode(List.of(END), false));
  }

  @Test
  void thePlacementRecordValidatesItsRange() {
    assertEquals(9, new Qwen3VlImagePlacement(3, 6).featureEnd());
    assertEquals(
        "featureStart must be non-negative: -1",
        assertThrows(IllegalArgumentException.class, () -> new Qwen3VlImagePlacement(-1, 1))
            .getMessage());
    assertEquals(
        "featureCount must be positive: 0",
        assertThrows(IllegalArgumentException.class, () -> new Qwen3VlImagePlacement(0, 0))
            .getMessage());
  }

  @Test
  void theRopePlanRecordValidatesItsAxes() {
    assertEquals(
        "the three position axes must be equally long: 2, 1, 2",
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen3VlRopePlan(new int[] {0, 1}, new int[] {0}, new int[] {0, 1}, 0))
            .getMessage());
    assertEquals(
        "widthPositions positions must be non-negative: -1",
        assertThrows(
                IllegalArgumentException.class,
                () -> new Qwen3VlRopePlan(new int[] {0}, new int[] {0}, new int[] {-1}, 0))
            .getMessage());
  }

  @Test
  void thePlanConstructorRejectsInconsistentPositionsAndDelta() {
    assertEquals(
        "heightPositions must be as long as the expanded prompt: 1 vs 2",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new Qwen3VlPromptPlan(
                        new int[] {1},
                        new int[] {1, 2},
                        List.of(),
                        new int[] {0, 1},
                        new int[] {0},
                        new int[] {0, 1},
                        0))
            .getMessage());
    assertEquals(
        "heightPositions positions must be non-negative: -1",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new Qwen3VlPromptPlan(
                        new int[] {1},
                        new int[] {1, 2},
                        List.of(),
                        new int[] {0, 1},
                        new int[] {0, -1},
                        new int[] {0, 1},
                        0))
            .getMessage());
    assertEquals(
        "positionDelta must equal the largest coordinate plus one minus the expanded length: 5 vs"
            + " 0",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new Qwen3VlPromptPlan(
                        new int[] {1},
                        new int[] {1, 2, 3},
                        List.of(),
                        new int[] {0, 1, 2},
                        new int[] {0, 1, 2},
                        new int[] {0, 1, 2},
                        5))
            .getMessage());
    assertEquals(
        "positionDelta must be zero for an empty prompt: 3",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new Qwen3VlPromptPlan(
                        new int[0], new int[0], List.of(), new int[0], new int[0], new int[0], 3))
            .getMessage());
  }

  @Test
  void thePlanConstructorRejectsInconsistentDestinations() {
    assertEquals(
        "feature destination 2 is out of bounds for the expanded prompt (2 tokens)",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new Qwen3VlPromptPlan(
                        new int[] {1},
                        new int[] {1, 2},
                        List.of(new Qwen3VlImagePlacement(1, 2)),
                        new int[] {0, 1},
                        new int[] {0, 1},
                        new int[] {0, 1},
                        0))
            .getMessage());
    assertEquals(
        "feature destinations must be strictly increasing: 1 after 2",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new Qwen3VlPromptPlan(
                        new int[] {1},
                        new int[] {1, 2, 3},
                        List.of(new Qwen3VlImagePlacement(2, 1), new Qwen3VlImagePlacement(1, 1)),
                        new int[] {0, 1, 2},
                        new int[] {0, 1, 2},
                        new int[] {0, 1, 2},
                        0))
            .getMessage());
    assertEquals(
        "feature destination run "
            + new Qwen3VlImagePlacement(Integer.MAX_VALUE - 1, 10)
            + " overflows a signed 32-bit integer",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new Qwen3VlPromptPlan(
                        new int[] {1},
                        new int[] {1, 2, 3, 4},
                        List.of(new Qwen3VlImagePlacement(Integer.MAX_VALUE - 1, 10)),
                        new int[] {0, 1, 2, 3},
                        new int[] {0, 1, 2, 3},
                        new int[] {0, 1, 2, 3},
                        0))
            .getMessage());
  }

  @Test
  void theFeatureInvariantCatchesUndeliveredImageTokens() {
    Qwen3VlPromptPlan undelivered =
        new Qwen3VlPromptPlan(
            new int[] {1, 2, 3},
            new int[] {1, IMAGE, 2, IMAGE, 3},
            List.of(new Qwen3VlImagePlacement(1, 1)),
            new int[] {0, 1, 2, 3, 4},
            new int[] {0, 1, 2, 3, 4},
            new int[] {0, 1, 2, 3, 4},
            0);
    assertEquals(
        "expanded prompt holds 2 image tokens but 1 feature rows are planned",
        assertThrows(
                IllegalStateException.class,
                () -> Qwen3VlPromptExpander.verifyFeatureDestinations(undelivered, IMAGE))
            .getMessage());

    Qwen3VlPromptPlan misdirected =
        new Qwen3VlPromptPlan(
            new int[] {1, 2, 3},
            new int[] {1, IMAGE, 3, IMAGE, 5},
            List.of(new Qwen3VlImagePlacement(1, 2)),
            new int[] {0, 1, 2, 3, 4},
            new int[] {0, 1, 2, 3, 4},
            new int[] {0, 1, 2, 3, 4},
            0);
    assertEquals(
        "feature destination 2 does not hold the image token",
        assertThrows(
                IllegalStateException.class,
                () -> Qwen3VlPromptExpander.verifyFeatureDestinations(misdirected, IMAGE))
            .getMessage());
  }

  @Test
  void thePlanBudgetArithmeticMatchesTheSevenPointThreeFamily() {
    Qwen3VlPromptPlan plan =
        Qwen3VlPromptExpander.expand(
            new int[] {1, 2, START, IMAGE, END, 3, 4}, tokens(), List.of(geometry(4, 6, 2)));
    assertEquals(0L, plan.requiredCacheCapacity(0));
    assertEquals(16L, plan.requiredCacheCapacity(5));
    assertEquals(
        "maxNewTokens must be non-negative: -1",
        assertThrows(IllegalArgumentException.class, () -> plan.requiredCacheCapacity(-1))
            .getMessage());

    assertDoesNotThrow(() -> plan.requireCacheCapacity(KVCachePolicy.full(), 5));
    assertDoesNotThrow(() -> plan.requireCacheCapacity(KVCachePolicy.slidingWindow(4), 5));
    assertEquals(
        "generation (expanded prompt 12 tokens, unexpanded 7 tokens) exceeds capacity",
        assertThrows(
                IllegalArgumentException.class,
                () -> plan.requireCacheCapacity(KVCachePolicy.full(3), 5))
            .getMessage());
    assertThrows(NullPointerException.class, () -> plan.requireCacheCapacity(null, 5));
  }

  @Test
  void theDecodeContinuationRejectsNegativePositions() {
    Qwen3VlPromptPlan plan =
        Qwen3VlPromptExpander.expand(
            new int[] {START, IMAGE, END, 7, START, IMAGE, END},
            tokens(),
            List.of(geometry(4, 6, 2), geometry(2, 4, 2)));
    assertEquals(
        "cachePosition must be non-negative: -1",
        assertThrows(IllegalArgumentException.class, () -> plan.ropePositionAt(-1)).getMessage());
    assertEquals(
        "RoPE position at cache position 2 is negative: -1 (delta -3); decode starts at cache "
            + "position 13",
        assertThrows(IllegalArgumentException.class, () -> plan.ropePositionAt(2)).getMessage());
  }

  @Test
  void thePlanOwnsItsArraysAndList() {
    int[] unexpanded = {1, 2, START, IMAGE, END, 3, 4};
    List<Qwen3VlGeometryPlan> geometries = new ArrayList<>();
    geometries.add(geometry(4, 6, 2));
    Qwen3VlPromptPlan plan = Qwen3VlPromptExpander.expand(unexpanded, tokens(), geometries);
    unexpanded[0] = 999;
    geometries.clear();
    assertArrayEquals(new int[] {1, 2, START, IMAGE, END, 3, 4}, plan.unexpandedIds());
    assertEquals(1, plan.images().size());

    int[] expanded = plan.expandedIds();
    expanded[0] = 999;
    int[] temporal = plan.temporalPositions();
    temporal[0] = 999;
    int[] height = plan.heightPositions();
    height[0] = 999;
    int[] width = plan.widthPositions();
    width[0] = 999;
    int[] unexpandedAgain = plan.unexpandedIds();
    assertEquals(1, unexpandedAgain[0]);
    assertArrayEquals(
        new int[] {1, 2, START, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, END, 3, 4},
        plan.expandedIds());
    assertEquals(0, plan.temporalPositions()[0]);
    assertEquals(0, plan.heightPositions()[0]);
    assertEquals(0, plan.widthPositions()[0]);
  }

  @Test
  void thePlanComparesByValue() {
    Qwen3VlPromptPlan plan =
        Qwen3VlPromptExpander.expand(
            new int[] {1, 2, START, IMAGE, END, 3, 4}, tokens(), List.of(geometry(4, 6, 2)));
    Qwen3VlPromptPlan equal =
        new Qwen3VlPromptPlan(
            new int[] {1, 2, START, IMAGE, END, 3, 4},
            plan.expandedIds(),
            plan.images(),
            plan.temporalPositions(),
            plan.heightPositions(),
            plan.widthPositions(),
            plan.positionDelta());
    assertEquals(plan, equal);
    assertEquals(plan.hashCode(), equal.hashCode());
    assertNotNull(plan.toString());
    assertTrue(plan.toString().startsWith("Qwen3VlPromptPlan["));
    // The delta is fixed by the axes (the constructor enforces it), so the inequality case varies
    // the placements instead: a valid destination run that is different from the plan's.
    Qwen3VlPromptPlan differentImages =
        new Qwen3VlPromptPlan(
            plan.unexpandedIds(),
            plan.expandedIds(),
            List.of(new Qwen3VlImagePlacement(4, 6)),
            plan.temporalPositions(),
            plan.heightPositions(),
            plan.widthPositions(),
            plan.positionDelta());
    assertNotEquals(plan, differentImages);
  }
}
