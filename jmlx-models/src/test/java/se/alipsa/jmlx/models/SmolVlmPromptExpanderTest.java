package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.SmolVlmGeometryPlan;
import se.alipsa.jmlx.vision.SmolVlmImageProcessor;
import se.alipsa.jmlx.vision.SmolVlmProcessorConfig;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Pure SmolVLM prompt expansion against the committed Hugging Face chat golden and against
 * synthetic marker/tokenizer combinations. No model execution and no pixels: the expander is
 * ID-space arithmetic, and the image grids come from the processor's shared pure geometry plan.
 */
class SmolVlmPromptExpanderTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private record Fixtures(
      SmolVlmPromptTokens tokens, SmolVlmImageProcessor processor, int tokensPerTile) {}

  private static Path family() {
    return Path.of(
            System.getProperty("jmlx.repository.root"),
            "jmlx-tokenizer",
            "src",
            "test",
            "resources",
            "families",
            "smolvlm")
        .toAbsolutePath();
  }

  private static JsonNode golden() throws Exception {
    return MAPPER.readTree(
        Path.of(
                System.getProperty("jmlx.repository.root"),
                "tools",
                "hf-reference",
                "goldens",
                "chat-smolvlm.json")
            .toFile());
  }

  /** Looks a token up in the family's {@code added_tokens.json} (a content to ID map). */
  private static int idOf(JsonNode addedTokens, String content) {
    JsonNode value = addedTokens.path(content);
    return value.isInt() ? value.asInt() : -1;
  }

  private static int[] ints(JsonNode values) {
    List<Integer> result = new ArrayList<>();
    values.forEach(value -> result.add(value.intValue()));
    return result.stream().mapToInt(Integer::intValue).toArray();
  }

  /** Resolves the special IDs from the tokenizer's own metadata, never from the golden. */
  private static Fixtures pinned() throws Exception {
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(family());
    // The newlines are ordinary BPE pieces of the same tokenizer, resolved at load time.
    assertEquals(List.of(198), tokenizer.encode("\n", false), "lone newline piece");
    assertEquals(List.of(1116), tokenizer.encode("\n\n", false), "doubled newline piece");
    JsonNode added = MAPPER.readTree(family().resolve("added_tokens.json").toFile());
    int image = idOf(added, "<image>");
    int fake = idOf(added, "<fake_token_around_image>");
    int global = idOf(added, "<global-img>");
    int video = idOf(added, "<video>");
    assertTrue(image > 0 && fake > 0 && global > 0);
    assertEquals(-1, video, "the pinned SmolVLM-256M tokenizer declares no video token");
    // Verify every marker of the block, not just its ends: a contiguous block ordered
    // differently would defeat the expander's base-offset arithmetic.
    int base = idOf(added, "<row_1_col_1>") - 1;
    assertTrue(base > 0);
    for (int row = 1; row <= SmolVlmPromptTokens.MARKER_GRID_SIDE; row++) {
      for (int col = 1; col <= SmolVlmPromptTokens.MARKER_GRID_SIDE; col++) {
        assertEquals(
            base + (row - 1) * SmolVlmPromptTokens.MARKER_GRID_SIDE + col,
            idOf(added, "<row_" + row + "_col_" + col + ">"),
            "<row_" + row + "_col_" + col + ">");
      }
    }
    int tokensPerTile =
        MAPPER
            .readTree(family().resolve("processor_config.json").toFile())
            .required("image_seq_len")
            .asInt();
    assertEquals(64, tokensPerTile);
    SmolVlmProcessorConfig config =
        SmolVlmProcessorConfig.fromJson(
            Files.readAllBytes(family().resolve("preprocessor_config.json")));
    return new Fixtures(
        new SmolVlmPromptTokens(
            image, video, fake, global, base, new int[] {198}, new int[] {1116}),
        new SmolVlmImageProcessor(config),
        tokensPerTile);
  }

  @Test
  void goldenExpansionsMatchThePinnedReference() throws Exception {
    Fixtures fixtures = pinned();
    for (JsonNode caseNode : golden().required("cases")) {
      String name = caseNode.required("name").asString();
      int[] unexpanded = ints(caseNode.required("ids"));
      List<SmolVlmImageGrid> grids = new ArrayList<>();
      for (JsonNode image : caseNode.path("images")) {
        // The grid comes from the shared pure geometry plan, not from the golden's name.
        SmolVlmGeometryPlan geometry =
            fixtures
                .processor()
                .geometry(image.required("height").asInt(), image.required("width").asInt());
        // Names are "tile-{rows}x{cols}-{width}x{height}", e.g. "tile-4x4-100x80".
        String tileName = image.required("name").asString();
        String gridPart = tileName.substring(5, tileName.indexOf('-', 5));
        int gridSplit = gridPart.indexOf('x');
        assertEquals(Integer.parseInt(gridPart.substring(0, gridSplit)), geometry.rows(), tileName);
        assertEquals(
            Integer.parseInt(gridPart.substring(gridSplit + 1)), geometry.cols(), tileName);
        grids.add(new SmolVlmImageGrid(geometry.rows(), geometry.cols()));
      }
      SmolVlmPromptPlan plan =
          SmolVlmPromptExpander.expand(
              unexpanded, fixtures.tokens(), List.copyOf(grids), fixtures.tokensPerTile());
      assertArrayEquals(unexpanded, plan.unexpandedIds(), name);
      assertEquals(unexpanded.length, plan.unexpandedLength(), name);
      if (caseNode.has("expanded_ids")) {
        assertArrayEquals(ints(caseNode.required("expanded_ids")), plan.expandedIds(), name);
        assertEquals(plan.expandedIds().length, plan.expandedLength(), name);
      } else {
        // No images: the degenerate plan retains the prompt unchanged.
        assertArrayEquals(unexpanded, plan.expandedIds(), name);
        assertEquals(0, plan.totalFeatureRows(), name);
      }
    }
  }

  @Test
  void featureDestinationsCoverEveryImageTokenExactlyOnce() throws Exception {
    Fixtures fixtures = pinned();
    JsonNode caseNode = null;
    for (JsonNode node : golden().required("cases")) {
      if (node.required("name").asString().equals("single_image_4x4")) {
        caseNode = node;
      }
    }
    assertNotNull(caseNode);
    int[] unexpanded = ints(caseNode.required("ids"));
    SmolVlmGeometryPlan geometry = fixtures.processor().geometry(80, 100);
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(
            unexpanded,
            fixtures.tokens(),
            List.of(new SmolVlmImageGrid(geometry.rows(), geometry.cols())),
            fixtures.tokensPerTile());
    assertEquals(17, plan.images().getFirst().tileCount(), "16 crops plus the global thumbnail");
    assertEquals(17 * 64, plan.totalFeatureRows());

    int[] destinations = plan.featureDestinations();
    assertEquals(17 * 64, destinations.length);
    int imageToken = fixtures.tokens().imageTokenId();
    int expected = 0;
    for (int destination : destinations) {
      assertTrue(destination > expected, "destinations must be strictly increasing");
      expected = destination;
      assertEquals(imageToken, plan.expandedIds()[destination]);
    }
    // Every image token in the expanded prompt is a destination, no more and no less.
    int imageTokens = 0;
    for (int id : plan.expandedIds()) {
      if (id == imageToken) {
        imageTokens++;
      }
    }
    assertEquals(destinations.length, imageTokens);
  }

  private static final int IMAGE = 10;
  private static final int FAKE = 11;
  private static final int GLOBAL = 12;
  private static final int BASE = 100;
  private static final int ROW_NL = 20;
  private static final int DOUBLE_NL = 21;
  private static final int TPP = 4;

  private static SmolVlmPromptTokens synthetic(int video) {
    return new SmolVlmPromptTokens(
        IMAGE, video, FAKE, GLOBAL, BASE, new int[] {ROW_NL}, new int[] {DOUBLE_NL});
  }

  @Test
  void unsplitImageExpandsToFakeGlobalRunFake() {
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(
            new int[] {1, 2, IMAGE, 3, 4}, synthetic(-1), List.of(new SmolVlmImageGrid(0, 0)), TPP);
    assertArrayEquals(
        new int[] {1, 2, FAKE, GLOBAL, IMAGE, IMAGE, IMAGE, IMAGE, FAKE, 3, 4}, plan.expandedIds());
    // The region opens with fake@2 and global@3, so the image run starts at 4.
    assertArrayEquals(new int[] {4}, plan.images().getFirst().tileStarts());
    assertArrayEquals(new int[] {4, 5, 6, 7}, plan.featureDestinations());
  }

  @Test
  void splitImageEmitsRowMajorTilesRowNewlinesAndMergedGlobalNewline() {
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(
            new int[] {IMAGE}, synthetic(-1), List.of(new SmolVlmImageGrid(2, 3)), TPP);
    int[] expected =
        new int[] {
          FAKE, BASE + 1, IMAGE, IMAGE, IMAGE, IMAGE, FAKE, BASE + 2, IMAGE, IMAGE, IMAGE, IMAGE,
          FAKE, BASE + 3, IMAGE, IMAGE, IMAGE, IMAGE, ROW_NL, FAKE, BASE + 7, IMAGE, IMAGE, IMAGE,
          IMAGE, FAKE, BASE + 8, IMAGE, IMAGE, IMAGE, IMAGE, FAKE, BASE + 9, IMAGE, IMAGE, IMAGE,
          IMAGE, DOUBLE_NL, FAKE, GLOBAL, IMAGE, IMAGE, IMAGE, IMAGE, FAKE
        };
    assertArrayEquals(expected, plan.expandedIds());
    assertArrayEquals(new int[] {2, 8, 14, 21, 27, 33, 40}, plan.images().getFirst().tileStarts());
    assertEquals(7 * TPP, plan.totalFeatureRows());
  }

  @Test
  void imagesAreExpandedInPromptOrderWithTheirOwnGrids() {
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(
            new int[] {IMAGE, 5, IMAGE},
            synthetic(-1),
            List.of(new SmolVlmImageGrid(0, 0), new SmolVlmImageGrid(1, 2)),
            TPP);
    assertEquals(2, plan.images().size());
    assertEquals(1, plan.images().getFirst().tileCount());
    assertEquals(3, plan.images().get(1).tileCount());
    // The first region occupies positions 0-6, the separator sits at 7, and the second region
    // opens with fake@8 and row/col@9, so its first tile starts at 10.
    assertEquals(2, plan.images().getFirst().tileStarts()[0]);
    assertEquals(10, plan.images().get(1).tileStarts()[0]);
    assertTrue(
        plan.featureDestinations()[0]
            < plan.featureDestinations()[plan.featureDestinations().length - 1]);
  }

  @Test
  void placeholderCountMismatchIsRejectedNamingBothSides() {
    SmolVlmPromptTokens tokens = synthetic(-1);
    IllegalArgumentException twoToOne =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                SmolVlmPromptExpander.expand(
                    new int[] {1, IMAGE, 2, IMAGE, 3},
                    tokens,
                    List.of(new SmolVlmImageGrid(0, 0)),
                    TPP));
    assertEquals(
        "prompt contains 2 <image> placeholder(s) but 1 image(s) were provided",
        twoToOne.getMessage());
    IllegalArgumentException oneToTwo =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                SmolVlmPromptExpander.expand(
                    new int[] {1, IMAGE, 2},
                    tokens,
                    List.of(new SmolVlmImageGrid(0, 0), new SmolVlmImageGrid(1, 1)),
                    TPP));
    assertEquals(
        "prompt contains 1 <image> placeholder(s) but 2 image(s) were provided",
        oneToTwo.getMessage());
    IllegalArgumentException none =
        assertThrows(
            IllegalArgumentException.class,
            () -> SmolVlmPromptExpander.expand(new int[] {1, IMAGE, 2}, tokens, List.of(), TPP));
    assertEquals(
        "prompt contains 1 <image> placeholder(s) but 0 image(s) were provided", none.getMessage());
  }

  @Test
  void processorOnlyMarkersAreRejectedWithDistinctMessages() {
    SmolVlmPromptTokens tokens = synthetic(-1);
    List<SmolVlmImageGrid> grid = List.of(new SmolVlmImageGrid(0, 0));
    String fake =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    SmolVlmPromptExpander.expand(new int[] {1, FAKE, IMAGE, 2}, tokens, grid, TPP))
            .getMessage();
    String column =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    SmolVlmPromptExpander.expand(
                        new int[] {1, BASE + 5, IMAGE, 2}, tokens, grid, TPP))
            .getMessage();
    String global =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    SmolVlmPromptExpander.expand(
                        new int[] {1, GLOBAL, IMAGE, 2}, tokens, grid, TPP))
            .getMessage();
    assertTrue(fake.contains("fake") && fake.contains(String.valueOf(FAKE)));
    assertTrue(column.contains("row/column") && column.contains(String.valueOf(BASE + 5)));
    assertTrue(global.contains("global") && global.contains(String.valueOf(GLOBAL)));
    assertNotEquals(fake, column);
    assertNotEquals(column, global);
    // The base itself is not a marker and passes through untouched.
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(new int[] {BASE, IMAGE}, tokens, grid, TPP);
    assertEquals(BASE, plan.expandedIds()[0]);
  }

  @Test
  void markersAreRejectedEvenWithoutImagesToExpand() {
    SmolVlmPromptTokens tokens = synthetic(-1);
    assertThrows(
        IllegalArgumentException.class,
        () -> SmolVlmPromptExpander.expand(new int[] {1, FAKE, 2}, tokens, List.of(), TPP));
  }

  @Test
  void videoTokenIsRejectedOnlyWhenTheTokenizerDeclaresOne() {
    SmolVlmPromptTokens withVideo = synthetic(99);
    IllegalArgumentException video =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                SmolVlmPromptExpander.expand(
                    new int[] {1, 99, IMAGE, 2},
                    withVideo,
                    List.of(new SmolVlmImageGrid(0, 0)),
                    TPP));
    assertEquals(
        "prompt contains the video token (ID 99); video is not supported", video.getMessage());
    // With no declared video token the same ID is an ordinary token.
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(
            new int[] {1, 99, IMAGE, 2}, synthetic(-1), List.of(new SmolVlmImageGrid(0, 0)), TPP);
    assertEquals(99, plan.expandedIds()[1]);
  }

  @Test
  void gridsOutsideTheMarkerSpaceAreRejected() {
    SmolVlmPromptTokens tokens = synthetic(-1);
    IllegalArgumentException rows =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                SmolVlmPromptExpander.expand(
                    new int[] {IMAGE}, tokens, List.of(new SmolVlmImageGrid(7, 2)), TPP));
    assertTrue(rows.getMessage().contains("7x2"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            SmolVlmPromptExpander.expand(
                new int[] {IMAGE}, tokens, List.of(new SmolVlmImageGrid(2, 7)), TPP));
    assertThrows(
        IllegalArgumentException.class,
        () -> SmolVlmPromptExpander.expand(new int[] {1, 2}, tokens, List.of(), 0));
  }

  @Test
  void budgetUsesTheExpandedLengthAndIsZeroForPromptOnly() {
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(
            new int[] {1, 2, IMAGE, 3}, synthetic(-1), List.of(new SmolVlmImageGrid(2, 3)), TPP);
    int expanded = plan.expandedIds().length;
    // 3 surrounding tokens plus the (2 x 3) grid's 45-token region.
    assertEquals(48, expanded);
    assertEquals(0L, plan.requiredCacheCapacity(0), "prompt-only needs no cache");
    assertEquals(expanded, plan.requiredCacheCapacity(1));
    assertEquals(expanded + 10 - 1, plan.requiredCacheCapacity(10));
    assertThrows(IllegalArgumentException.class, () -> plan.requiredCacheCapacity(-1));

    long exact = plan.requiredCacheCapacity(10);
    plan.requireCacheCapacity(KVCachePolicy.full((int) exact), 10);
    plan.requireCacheCapacity(KVCachePolicy.full(), 10);
    plan.requireCacheCapacity(KVCachePolicy.slidingWindow(4), 1000);
    plan.requireCacheCapacity(KVCachePolicy.full(), 0);
    IllegalArgumentException over =
        assertThrows(
            IllegalArgumentException.class,
            () -> plan.requireCacheCapacity(KVCachePolicy.full((int) exact - 1), 10));
    assertEquals(
        "generation (expanded prompt "
            + expanded
            + " tokens, unexpanded 4 tokens) exceeds capacity",
        over.getMessage());
  }

  @Test
  void tokensRecordValidatesDistinctAndNonOverlappingIds() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new SmolVlmPromptTokens(FAKE, -1, FAKE, GLOBAL, BASE, new int[] {1}, new int[] {1}));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new SmolVlmPromptTokens(
                IMAGE, IMAGE, FAKE, GLOBAL, BASE, new int[] {1}, new int[] {1}));
    assertThrows(
        IllegalArgumentException.class,
        () -> new SmolVlmPromptTokens(IMAGE, -2, FAKE, GLOBAL, BASE, new int[] {1}, new int[] {1}));
    // The marker block must not overlap the special tokens (the base itself may).
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new SmolVlmPromptTokens(
                IMAGE, -1, FAKE, BASE + 10, BASE, new int[] {1}, new int[] {1}));
    assertDoesNotThrow(
        () -> new SmolVlmPromptTokens(IMAGE, -1, FAKE, BASE, BASE, new int[] {1}, new int[] {1}));
    assertThrows(
        IllegalArgumentException.class,
        () -> new SmolVlmPromptTokens(IMAGE, -1, FAKE, GLOBAL, BASE, new int[0], new int[] {1}));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new SmolVlmPromptTokens(IMAGE, -1, FAKE, GLOBAL, BASE, new int[] {-1}, new int[] {1}));
    assertThrows(
        IllegalArgumentException.class,
        () -> new SmolVlmPromptTokens(IMAGE, -1, FAKE, GLOBAL, -1, new int[] {1}, new int[] {1}));
    assertEquals(BASE + 2, synthetic(-1).rowColTokenId(1, 2));
    assertEquals(BASE + 36, synthetic(-1).rowColTokenId(6, 6));
    assertThrows(IllegalArgumentException.class, () -> synthetic(-1).rowColTokenId(0, 1));
    assertThrows(IllegalArgumentException.class, () -> synthetic(-1).rowColTokenId(7, 1));
  }

  @Test
  void gridAndPlacementRecordsValidateShape() {
    assertThrows(IllegalArgumentException.class, () -> new SmolVlmImageGrid(1, 0));
    assertThrows(IllegalArgumentException.class, () -> new SmolVlmImageGrid(-1, 2));
    assertEquals(1, new SmolVlmImageGrid(0, 0).tileCount());
    assertEquals(5, new SmolVlmImageGrid(2, 2).tileCount());
    assertThrows(IllegalArgumentException.class, () -> new SmolVlmImagePlacement(new int[0]));
    assertThrows(IllegalArgumentException.class, () -> new SmolVlmImagePlacement(new int[] {5, 5}));
    assertThrows(IllegalArgumentException.class, () -> new SmolVlmImagePlacement(new int[] {9, 4}));
    assertDoesNotThrow(() -> new SmolVlmImagePlacement(new int[] {1, 4, 9}));
  }

  @Test
  void newlineRunsCollidingWithSpecialTokensAreRejected() {
    int[] rowNewline = new int[] {ROW_NL};
    String imageCollision =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new SmolVlmPromptTokens(
                        IMAGE, -1, FAKE, GLOBAL, BASE, new int[] {IMAGE}, rowNewline))
            .getMessage();
    assertTrue(imageCollision.contains("rowNewlineTokenIds"));
    assertTrue(imageCollision.contains("the image token"));
    assertTrue(imageCollision.contains(String.valueOf(IMAGE)));
    String markerCollision =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new SmolVlmPromptTokens(
                        IMAGE, -1, FAKE, GLOBAL, BASE, rowNewline, new int[] {BASE + 12}))
            .getMessage();
    assertTrue(markerCollision.contains("globalBlockNewlineTokenIds"));
    assertTrue(markerCollision.contains("a row/column marker"));
    // A declared video token collides; without one, the same ID is an ordinary token.
    String videoCollision =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    new SmolVlmPromptTokens(
                        IMAGE, 99, FAKE, GLOBAL, BASE, new int[] {99}, rowNewline))
            .getMessage();
    assertTrue(videoCollision.contains("the video token"));
    assertDoesNotThrow(
        () -> new SmolVlmPromptTokens(IMAGE, -1, FAKE, GLOBAL, BASE, new int[] {99}, rowNewline));
    assertThrows(
        IllegalArgumentException.class,
        () -> new SmolVlmPromptTokens(IMAGE, -1, FAKE, GLOBAL, BASE, new int[] {FAKE}, rowNewline));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new SmolVlmPromptTokens(IMAGE, -1, FAKE, GLOBAL, BASE, rowNewline, new int[] {GLOBAL}));
  }

  @Test
  void recordAccessorsReturnDefensiveCopies() {
    int[] prompt = new int[] {1, 2, IMAGE, 3};
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(
            prompt, synthetic(-1), List.of(new SmolVlmImageGrid(2, 3)), TPP);
    prompt[0] = 999;
    assertEquals(1, plan.unexpandedIds()[0], "the plan must own a copy of the prompt");

    int[] expanded = plan.expandedIds();
    expanded[0] = 999;
    assertEquals(1, plan.expandedIds()[0], "an expandedIds() mutation must not stick");
    int[] unexpanded = plan.unexpandedIds();
    unexpanded[0] = 999;
    assertEquals(1, plan.unexpandedIds()[0], "an unexpandedIds() mutation must not stick");
    int[] tileStarts = plan.images().getFirst().tileStarts();
    tileStarts[0] = 999;
    assertNotEquals(999, plan.images().getFirst().tileStarts()[0]);
    int[] rowNewline = synthetic(-1).rowNewlineTokenIds();
    rowNewline[0] = 999;
    assertEquals(ROW_NL, synthetic(-1).rowNewlineTokenIds()[0]);
    int[] globalNewline = synthetic(-1).globalBlockNewlineTokenIds();
    globalNewline[0] = 999;
    assertEquals(DOUBLE_NL, synthetic(-1).globalBlockNewlineTokenIds()[0]);
  }

  @Test
  void planMinimumLengthCheckUsesLongArithmetic() {
    // tokensPerTile + 3 must not wrap in int arithmetic: the wrap would make the minimum
    // check pass where it must reject.
    SmolVlmImagePlacement placement = new SmolVlmImagePlacement(new int[] {1});
    IllegalArgumentException over =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new SmolVlmPromptPlan(
                    new int[0], new int[] {1}, Integer.MAX_VALUE, List.of(placement)));
    assertTrue(over.getMessage().contains("2147483650"));
  }
}
