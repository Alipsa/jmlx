package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessor;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessorResult;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;

/**
 * Validation coverage for the WP5 vision tower's argument and key-mapping checks (PR #58 review
 * finding 4): the public {@link Qwen3VlVisionTower#visionKeyToTowerKey(String, int, int[])}
 * branches, {@link Qwen3VlVisionTower#visionWeights(Map, int, int[])}'s skip/transpose/duplicate
 * behavior, the constructor's missing/extra/non-floating-weight, deepstack-index, position-grid and
 * merge-width rejections, {@code runImage}'s geometry rejections, and the {@link
 * Qwen3VlVisionTower#rotaryCoordinates(int, int, int)}, {@link
 * Qwen3VlVisionTower#positionLookup(int, int, int)} and {@link
 * Qwen3VlVisionTower#linspaceValue(int, int, int)} argument checks.
 *
 * <p>Three of {@code runImage}'s documented rejections are untestable and intentionally not
 * covered: {@code gridThw()[0] != 1}, a {@code rawPatchRows()} that disagrees with the grid, and a
 * {@code values()} length that disagrees with {@code rawPatchRows * patchStride}. The {@code
 * Qwen3VlImageProcessorResult} sole constructor hard-codes the grid to {@code [1, h, w]} and its
 * {@code Qwen3VlGeometryPlan} validates {@code rawPatchRows == gridH * gridW} and the values
 * length, so no malformed result object can exist through the public API; those checks remain
 * defensive only.
 *
 * <p>Gated like the other WP5 tower suites because the constructor and {@code runImage} paths
 * allocate MLX arrays; the pure-Java mapping and coordinate methods are covered by the same class.
 */
@EnabledIfNativeAvailable
class Qwen3VlVisionTowerValidationTest {

  // Tiny consistent geometry: hidden 8 with 2 heads (headDim 4), a 4x4 position grid, depth 2
  // with one DeepStack index, merge 2 (m = 8 * 4 = 32). The processor-matching conv weight is
  // [hidden, temporal, patch, patch, channels] = [8, 2, 16, 16, 3] (the 16x16 / temporal-2
  // patch stride 1536 of the pinned processor chain).
  private static final int HIDDEN = 8;
  private static final int NUM_HEADS = 2;
  private static final int INTERMEDIATE = 8;
  private static final int OUT_HIDDEN = 8;
  private static final int GRID_SIDE = 4;
  private static final int DEPTH = 2;
  private static final int[] DEEPSTACK = {1};
  private static final int MERGE_BLOCK = HIDDEN * 2 * 2;
  private static final int IMAGE = 64;

  @ParameterizedTest
  @CsvSource({
    "model.visual.patch_embed.proj.weight, patchEmbedding.weight",
    "model.visual.patch_embed.proj.bias, patchEmbedding.bias",
    "model.visual.pos_embed.weight, positionEmbedding.weight",
    "model.visual.blocks.0.norm1.weight, layer0.norm1.weight",
    "model.visual.blocks.2.norm2.bias, layer2.norm2.bias",
    "model.visual.blocks.1.attn.qkv.weight, layer1.attention.qkv.weight",
    "model.visual.blocks.1.attn.qkv.bias, layer1.attention.qkv.bias",
    "model.visual.blocks.1.attn.proj.weight, layer1.attention.proj.weight",
    "model.visual.blocks.1.attn.proj.bias, layer1.attention.proj.bias",
    "model.visual.blocks.0.mlp.linear_fc1.weight, layer0.mlp.0.weight",
    "model.visual.blocks.0.mlp.linear_fc2.bias, layer0.mlp.2.bias",
    "model.visual.merger.norm.weight, merger.norm.weight",
    "model.visual.merger.linear_fc1.bias, merger.linearFc1.bias",
    "model.visual.merger.linear_fc2.weight, merger.linearFc2.weight",
    "model.visual.deepstack_merger_list.0.norm.bias, deepstackMerger0.norm.bias",
    "model.visual.deepstack_merger_list.1.linear_fc2.weight, deepstackMerger1.linearFc2.weight",
  })
  void checkpointKeysMapToTowerKeys(String checkpointKey, String expected) {
    // Depth 3 with a length-2 DeepStack list: every in-range index maps, so an off-by-one in the
    // block/merger bounds would surface here.
    assertEquals(
        expected, Qwen3VlVisionTower.visionKeyToTowerKey(checkpointKey, 3, new int[] {1, 5}));
  }

  @ParameterizedTest
  @ValueSource(
      strings = {
        "model.language_model.embed_tokens.weight", // non-visual
        "visual.patch_embed.proj.weight", // missing model.visual. prefix
        "model.visual.patch_embed.proj.gain", // unknown patch-embed sub-key
        "model.visual.pos_embed.bias", // pos_embed maps weight only
        "model.visual.blocks.3.norm1.weight", // block index out of range (depth 3)
        "model.visual.blocks.x.norm1.weight", // non-numeric block index
        "model.visual.blocks.-1.norm1.weight", // negative block index
        "model.visual.blocks.99999999999999999999.norm1.weight", // overflowing block index
        "model.visual.blocks..norm1.weight", // empty block index
        "model.visual.blocks.1.", // block key without a sub-key
        "model.visual.blocks.0.attn.gate.weight", // unknown attention sub-key
        "model.visual.blocks.0.attn.qkv.gain", // unknown attention tail
        "model.visual.blocks.0.mlp.linear_fc3.weight", // unknown MLP sub-key
        "model.visual.blocks.0.other.weight", // unknown block sub-key
        "model.visual.merger.linear_fc3.weight", // unknown merger sub-key
        "model.visual.merger.gain", // unknown merger tail
        "model.visual.deepstack_merger_list.2.norm.weight", // merger index out of range (2)
        "model.visual.deepstack_merger_list.x.norm.weight", // non-numeric merger index
        "model.visual.deepstack_merger_list.99999999999999999999.norm.weight", // overflowing index
        "model.visual.deepstack_merger_list.0.other.weight", // unknown merger sub-key
      })
  void unknownAndOutOfRangeKeysMapToNull(String checkpointKey) {
    assertNull(Qwen3VlVisionTower.visionKeyToTowerKey(checkpointKey, 3, new int[] {1, 5}));
  }

  @Test
  void visionKeyToTowerKeyRejectsNullArguments() {
    assertEquals(
        "key",
        assertThrows(
                NullPointerException.class,
                () -> Qwen3VlVisionTower.visionKeyToTowerKey(null, DEPTH, DEEPSTACK))
            .getMessage());
    assertEquals(
        "deepstackIndexes",
        assertThrows(
                NullPointerException.class,
                () ->
                    Qwen3VlVisionTower.visionKeyToTowerKey(
                        "model.visual.pos_embed.weight", DEPTH, null))
            .getMessage());
  }

  @Test
  void visionWeightsSkipsNonVisualAndOutOfRangeKeysAndTransposesConvWeight() {
    try (MLXScope scope = new MLXScope()) {
      float[] input = new float[24];
      for (int i = 0; i < input.length; i++) {
        input[i] = i + 1f;
      }
      Map<String, MLXArray> checkpoint = new LinkedHashMap<>();
      // PyTorch layout [out, in, T, H, W] = [2, 3, 1, 2, 2], distinct values per position.
      checkpoint.put(
          "model.visual.patch_embed.proj.weight",
          MLX.array(scope, input, new int[] {2, 3, 1, 2, 2}));
      checkpoint.put("model.visual.blocks.1.norm1.bias", full(scope, HIDDEN));
      checkpoint.put(
          "model.language_model.embed_tokens.weight", full(scope, new int[] {4, HIDDEN}));
      // Current behavior: well-formed keys whose block or merger index is out of the configured
      // range are silently dropped. PR #58 review finding 5 changes visionWeights to throw for
      // those keys; update this expectation when that lands.
      checkpoint.put("model.visual.blocks.9.norm1.weight", full(scope, HIDDEN));
      checkpoint.put("model.visual.deepstack_merger_list.5.norm.weight", full(scope, HIDDEN));

      Map<String, MLXArray> weights =
          Qwen3VlVisionTower.visionWeights(checkpoint, DEPTH, DEEPSTACK);

      assertEquals(2, weights.size(), "only the in-range visual keys survive");
      assertArrayEquals(
          new String[] {"patchEmbedding.weight", "layer1.norm1.bias"}, weights.keySet().toArray());
      // jmlx's channels-last layout [out, T, H, W, in], transposed {0, 2, 3, 4, 1}: output
      // [o, t, h, w, c] reads input [o, c, t, h, w]; with T = 1 the t term is always zero.
      float[] expected = new float[24];
      for (int o = 0; o < 2; o++) {
        for (int h = 0; h < 2; h++) {
          for (int w = 0; w < 2; w++) {
            for (int c = 0; c < 3; c++) {
              expected[o * 12 + h * 6 + w * 3 + c] = input[o * 12 + c * 4 + h * 2 + w];
            }
          }
        }
      }
      assertArrayEquals(expected, weights.get("patchEmbedding.weight").toFloatArray(), 0f);
    }
  }

  @Test
  void visionWeightsRejectsDuplicateTowerKeys() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> checkpoint = new LinkedHashMap<>();
      checkpoint.put("model.visual.blocks.0.norm1.weight", full(scope, HIDDEN));
      // "00" parses to the same index as "0", so both keys collide on the same tower key.
      checkpoint.put("model.visual.blocks.00.norm1.weight", full(scope, HIDDEN));
      IllegalArgumentException e =
          assertThrows(
              IllegalArgumentException.class,
              () -> Qwen3VlVisionTower.visionWeights(checkpoint, DEPTH, DEEPSTACK));
      assertEquals(
          "duplicate vision tower key \"layer0.norm1.weight\" from checkpoint key"
              + " \"model.visual.blocks.00.norm1.weight\"",
          e.getMessage());
    }
  }

  @Test
  void constructorRejectsMissingWeight() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      weights.remove("merger.norm.weight");
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          DEEPSTACK,
          "missing vision tower weight \"merger.norm.weight\"");
    }
  }

  @Test
  void constructorRejectsExtraWeight() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      weights.put("patchEmbedding.extra", full(scope, HIDDEN));
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          DEEPSTACK,
          "unexpected vision tower weights: [patchEmbedding.extra]");
    }
  }

  @Test
  void constructorRejectsNonFloatingWeight() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      weights.put(
          "positionEmbedding.weight",
          MLX.full(scope, new int[] {GRID_SIDE * GRID_SIDE, HIDDEN}, 1f, DType.INT32));
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          DEEPSTACK,
          "vision tower weights must be floating: positionEmbedding.weight");
    }
  }

  @Test
  void constructorRejectsInvalidDeepstackIndexes() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      expectConstructorMessage(
          scope, weights, NUM_HEADS, new int[] {}, "deepstackIndexes must not be empty");
      expectConstructorMessage(
          scope, weights, NUM_HEADS, new int[] {5}, "deepstack index 0 = 5 is out of range [0, 2)");
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          new int[] {-1},
          "deepstack index 0 = -1 is out of range [0, 2)");
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          new int[] {1, 1},
          "deepstack indexes must be strictly increasing; index 1 has 1 after 1");
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          new int[] {1, 0},
          "deepstack indexes must be strictly increasing; index 1 has 0 after 1");
    }
  }

  @Test
  void constructorRejectsLayerlessTower() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      weights.remove("layer0.norm1.weight");
      expectConstructorMessage(
          scope, weights, NUM_HEADS, DEEPSTACK, "the vision tower needs at least one layer");
    }
  }

  @Test
  void constructorRejectsNonSquarePositionGrid() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      weights.put("positionEmbedding.weight", full(scope, 12, HIDDEN));
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          DEEPSTACK,
          "positionEmbedding.weight must contain a square grid, got 12 rows");
    }
  }

  @Test
  void constructorRejectsNonSquareMergeWidth() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      weights.put("merger.linearFc1.weight", full(scope, 24, 24));
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          DEEPSTACK,
          "merger width m = 24 must be the hidden size times a square merge^2, got merge^2 = 3");
      weights.put("merger.linearFc1.weight", full(scope, 20, 20));
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          DEEPSTACK,
          "merger width m = 20 must be divisible by hidden size 8");
      weights.put("merger.linearFc1.weight", full(scope, 24, 16));
      expectConstructorMessage(
          scope,
          weights,
          NUM_HEADS,
          DEEPSTACK,
          "merger.linearFc1.weight must be square [m, m], got shape [24, 16]");
    }
  }

  @Test
  void constructorRejectsInvalidHeadCounts() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      expectConstructorMessage(
          scope, weights, 0, DEEPSTACK, "the vision tower needs at least one attention head");
      expectConstructorMessage(
          scope, weights, 3, DEEPSTACK, "hidden size 8 is not divisible by 3 attention heads");
    }
  }

  @Test
  void constructorRejectsNullArguments() {
    try (MLXScope scope = new MLXScope()) {
      assertEquals(
          "weights",
          assertThrows(
                  NullPointerException.class,
                  () -> new Qwen3VlVisionTower(scope, NUM_HEADS, null, DEEPSTACK))
              .getMessage());
      Map<String, MLXArray> weights = baseWeights(scope);
      assertEquals(
          "deepstackIndexes",
          assertThrows(
                  NullPointerException.class,
                  () -> new Qwen3VlVisionTower(scope, NUM_HEADS, weights, null))
              .getMessage());
    }
  }

  @Test
  void runImageRejectsPatchStrideMismatch() {
    Qwen3VlImageProcessorResult image = processedImage();
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      // The tower's 4x4 temporal-1 patch embed gives an expected stride of 3 * 1 * 4 * 4 = 48,
      // disagreeing with the processor's 1536.
      weights.put("patchEmbedding.weight", full(scope, HIDDEN, 1, 4, 4, 3));
      Qwen3VlVisionTower tower = buildTower(scope, weights);
      IllegalArgumentException e =
          assertThrows(IllegalArgumentException.class, () -> tower.forward(scope, List.of(image)));
      assertEquals(
          "image patch stride 1536 does not match the tower's 48 (inChannels * temporal *"
              + " patchHeight * patchWidth)",
          e.getMessage());
    }
  }

  @Test
  void runImageRejectsGridNotMultipleOfMerge() {
    Qwen3VlImageProcessorResult image = processedImage();
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> weights = baseWeights(scope);
      // Merge size 3 (m = 8 * 9 = 72): the processor's 4x4 grid is not a multiple of 3.
      putMergeWidth(weights, scope, 72);
      Qwen3VlVisionTower tower = buildTower(scope, weights);
      IllegalArgumentException e =
          assertThrows(IllegalArgumentException.class, () -> tower.forward(scope, List.of(image)));
      assertEquals("image grid 4x4 must be multiples of the spatial merge size 3", e.getMessage());
    }
  }

  @Test
  void forwardRejectsUnrelatedTargetScopeAndEmptyImages() {
    Qwen3VlImageProcessorResult image = processedImage();
    try (MLXScope towerScope = new MLXScope();
        MLXScope stranger = new MLXScope()) {
      Qwen3VlVisionTower tower = buildTower(towerScope, baseWeights(towerScope));
      IllegalArgumentException e =
          assertThrows(
              IllegalArgumentException.class, () -> tower.forward(stranger, List.of(image)));
      assertTrue(
          e.getMessage()
              .startsWith(
                  "target scope must be the tower's module scope or a descendant of it, got "),
          e.getMessage());
      assertEquals(
          "at least one image is required",
          assertThrows(IllegalArgumentException.class, () -> tower.forward(towerScope, List.of()))
              .getMessage());
    }
  }

  @Test
  void forwardRejectsNullArguments() {
    Qwen3VlImageProcessorResult image = processedImage();
    try (MLXScope scope = new MLXScope()) {
      Qwen3VlVisionTower tower = buildTower(scope, baseWeights(scope));
      assertEquals(
          "target",
          assertThrows(NullPointerException.class, () -> tower.forward(null, List.of(image)))
              .getMessage());
      assertEquals(
          "images",
          assertThrows(NullPointerException.class, () -> tower.forward(scope, null)).getMessage());
      assertEquals(
          "images element",
          assertThrows(
                  NullPointerException.class,
                  // Arrays.asList, not List.of: List.of rejects the null at creation with an
                  // unattributed message; the tower's own element check must fire instead.
                  () -> tower.forward(scope, Arrays.asList((Qwen3VlImageProcessorResult) null)))
              .getMessage());
    }
  }

  @Test
  void rotaryCoordinatesRejectsInvalidArguments() {
    assertEquals(
        "merge must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.rotaryCoordinates(8, 8, 0))
            .getMessage());
    assertEquals(
        "merge must be positive",
        assertThrows(
                IllegalArgumentException.class,
                () -> Qwen3VlVisionTower.rotaryCoordinates(8, 8, -1))
            .getMessage());
    assertEquals(
        "h and w must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.rotaryCoordinates(0, 4, 2))
            .getMessage());
    assertEquals(
        "h and w must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.rotaryCoordinates(4, 0, 2))
            .getMessage());
    assertEquals(
        "h = 6 and w = 8 must be multiples of merge = 4",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.rotaryCoordinates(6, 8, 4))
            .getMessage());
    assertEquals(
        "h = 8 and w = 7 must be multiples of merge = 2",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.rotaryCoordinates(8, 7, 2))
            .getMessage());
  }

  @Test
  void rotaryCoordinatesEmitMergeBlockOrder() {
    int[][] coordinates = Qwen3VlVisionTower.rotaryCoordinates(8, 8, 2);
    assertEquals(2, coordinates.length, "rows and cols");
    assertEquals(64, coordinates[0].length, "row count");
    assertEquals(64, coordinates[1].length, "col count");
    // The reference loop order documented in the method's Javadoc: the first eight coordinates
    // are (0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3), the last four (6,6),(6,7),(7,6),(7,7).
    assertArrayEquals(new int[] {0, 0, 1, 1, 0, 0, 1, 1}, Arrays.copyOf(coordinates[0], 8));
    assertArrayEquals(new int[] {0, 1, 0, 1, 2, 3, 2, 3}, Arrays.copyOf(coordinates[1], 8));
    assertArrayEquals(new int[] {6, 6, 7, 7}, Arrays.copyOfRange(coordinates[0], 60, 64));
    assertArrayEquals(new int[] {6, 7, 6, 7}, Arrays.copyOfRange(coordinates[1], 60, 64));
    for (int blockRow = 0; blockRow < 4; blockRow++) {
      for (int blockCol = 0; blockCol < 4; blockCol++) {
        for (int intraRow = 0; intraRow < 2; intraRow++) {
          for (int intraCol = 0; intraCol < 2; intraCol++) {
            int m = (blockRow * 4 + blockCol) * 4 + intraRow * 2 + intraCol;
            assertEquals(blockRow * 2 + intraRow, coordinates[0][m], "row " + m);
            assertEquals(blockCol * 2 + intraCol, coordinates[1][m], "col " + m);
          }
        }
      }
    }
  }

  @Test
  void positionLookupRejectsInvalidArguments() {
    assertEquals(
        "h, w and gridSide must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.positionLookup(0, 4, 4))
            .getMessage());
    assertEquals(
        "h, w and gridSide must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.positionLookup(4, 0, 4))
            .getMessage());
    assertEquals(
        "h, w and gridSide must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.positionLookup(4, 4, 0))
            .getMessage());
  }

  @Test
  void positionLookupComputesBilinearCorners() {
    // Exact 4x4 grid over itself: every patch sits on a grid point, so the weights are {1, 0,
    // 0, 0}; the corner indices are still the floor/ceil grid points, with the ceiling clipping
    // at gridSide - 1 on the bottom and right edges (where it collapses back onto the floor).
    Qwen3VlVisionTower.PositionLookup exact = Qwen3VlVisionTower.positionLookup(4, 4, 4);
    for (int ph = 0; ph < 4; ph++) {
      for (int pw = 0; pw < 4; pw++) {
        int k = ph * 4 + pw;
        int ceilH = Math.min(ph + 1, 3);
        int ceilW = Math.min(pw + 1, 3);
        int[] expectedCorners =
            new int[] {ph * 4 + pw, ph * 4 + ceilW, ceilH * 4 + pw, ceilH * 4 + ceilW};
        assertArrayEquals(expectedCorners, corners(exact, k), "exact corners " + ph + "x" + pw);
        assertArrayEquals(
            new float[] {1f, 0f, 0f, 0f},
            cornerWeights(exact, k),
            "exact weights " + ph + "x" + pw);
      }
    }
    // Fractional 3x3 grid over the 4x4 table: linspace(0, 3, 3) = {0, 1.5, 3}, so the middle
    // patch sits at (1.5, 1.5) and takes all four corners equally, while the top-left and
    // bottom-right patches sit exactly on grid points (the bottom-right at the clipped (3, 3)).
    Qwen3VlVisionTower.PositionLookup fractional = Qwen3VlVisionTower.positionLookup(3, 3, 4);
    assertArrayEquals(new int[] {5, 6, 9, 10}, corners(fractional, 4));
    assertArrayEquals(new float[] {0.25f, 0.25f, 0.25f, 0.25f}, cornerWeights(fractional, 4));
    assertArrayEquals(new int[] {0, 1, 4, 5}, corners(fractional, 0));
    assertArrayEquals(new float[] {1f, 0f, 0f, 0f}, cornerWeights(fractional, 0));
    assertArrayEquals(new int[] {15, 15, 15, 15}, corners(fractional, 8));
    assertArrayEquals(new float[] {1f, 0f, 0f, 0f}, cornerWeights(fractional, 8));
  }

  @Test
  void linspaceValueRejectsInvalidArguments() {
    assertEquals(
        "side and n must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.linspaceValue(0, 4, 0))
            .getMessage());
    assertEquals(
        "side and n must be positive",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.linspaceValue(4, 0, 0))
            .getMessage());
    assertEquals(
        "i = -1 is out of range [0, 4)",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.linspaceValue(4, 4, -1))
            .getMessage());
    assertEquals(
        "i = 4 is out of range [0, 4)",
        assertThrows(
                IllegalArgumentException.class, () -> Qwen3VlVisionTower.linspaceValue(4, 4, 4))
            .getMessage());
    assertEquals(0f, Qwen3VlVisionTower.linspaceValue(4, 1, 0), 0f, "the n = 1 fill guard");
    float[] uniform = new float[4];
    for (int i = 0; i < 4; i++) {
      uniform[i] = Qwen3VlVisionTower.linspaceValue(4, 4, i);
    }
    assertArrayEquals(new float[] {0f, 1f, 2f, 3f}, uniform, 0f);
  }

  /** The all-expected-key weight map of the tiny geometry, constant 0.001 float32. */
  private static Map<String, MLXArray> baseWeights(MLXScope scope) {
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    put(weights, scope, "patchEmbedding.weight", HIDDEN, 2, 16, 16, 3);
    put(weights, scope, "patchEmbedding.bias", HIDDEN);
    put(weights, scope, "positionEmbedding.weight", GRID_SIDE * GRID_SIDE, HIDDEN);
    for (int i = 0; i < DEPTH; i++) {
      String layer = "layer" + i + ".";
      put(weights, scope, layer + "attention.qkv.weight", 3 * HIDDEN, HIDDEN);
      put(weights, scope, layer + "attention.qkv.bias", 3 * HIDDEN);
      put(weights, scope, layer + "attention.proj.weight", HIDDEN, HIDDEN);
      put(weights, scope, layer + "attention.proj.bias", HIDDEN);
      for (String norm : new String[] {"norm1", "norm2"}) {
        put(weights, scope, layer + norm + ".weight", HIDDEN);
        put(weights, scope, layer + norm + ".bias", HIDDEN);
      }
      put(weights, scope, layer + "mlp.0.weight", INTERMEDIATE, HIDDEN);
      put(weights, scope, layer + "mlp.0.bias", INTERMEDIATE);
      put(weights, scope, layer + "mlp.2.weight", HIDDEN, INTERMEDIATE);
      put(weights, scope, layer + "mlp.2.bias", HIDDEN);
    }
    put(weights, scope, "merger.norm.weight", HIDDEN);
    put(weights, scope, "merger.norm.bias", HIDDEN);
    put(weights, scope, "merger.linearFc1.weight", MERGE_BLOCK, MERGE_BLOCK);
    put(weights, scope, "merger.linearFc1.bias", MERGE_BLOCK);
    put(weights, scope, "merger.linearFc2.weight", OUT_HIDDEN, MERGE_BLOCK);
    put(weights, scope, "merger.linearFc2.bias", OUT_HIDDEN);
    for (int j = 0; j < DEEPSTACK.length; j++) {
      String prefix = "deepstackMerger" + j + ".";
      put(weights, scope, prefix + "norm.weight", MERGE_BLOCK);
      put(weights, scope, prefix + "norm.bias", MERGE_BLOCK);
      put(weights, scope, prefix + "linearFc1.weight", MERGE_BLOCK, MERGE_BLOCK);
      put(weights, scope, prefix + "linearFc1.bias", MERGE_BLOCK);
      put(weights, scope, prefix + "linearFc2.weight", OUT_HIDDEN, MERGE_BLOCK);
      put(weights, scope, prefix + "linearFc2.bias", OUT_HIDDEN);
    }
    return weights;
  }

  /**
   * Rewires the main merger and the first DeepStack merger to the non-default merge-block width
   * {@code m} (kept consistent across the two so the constructor accepts the tower).
   */
  private static void putMergeWidth(Map<String, MLXArray> weights, MLXScope scope, int m) {
    weights.put("merger.linearFc1.weight", full(scope, m, m));
    weights.put("merger.linearFc1.bias", full(scope, m));
    weights.put("merger.linearFc2.weight", full(scope, OUT_HIDDEN, m));
    weights.put("deepstackMerger0.norm.weight", full(scope, m));
    weights.put("deepstackMerger0.norm.bias", full(scope, m));
    weights.put("deepstackMerger0.linearFc1.weight", full(scope, m, m));
    weights.put("deepstackMerger0.linearFc1.bias", full(scope, m));
    weights.put("deepstackMerger0.linearFc2.weight", full(scope, OUT_HIDDEN, m));
  }

  private static void put(Map<String, MLXArray> weights, MLXScope scope, String key, int... shape) {
    weights.put(key, full(scope, shape));
  }

  private static MLXArray full(MLXScope scope, int... shape) {
    return MLX.full(scope, shape, 0.001f, DType.FLOAT32);
  }

  private static Qwen3VlVisionTower buildTower(MLXScope scope, Map<String, MLXArray> weights) {
    return new Qwen3VlVisionTower(scope, NUM_HEADS, weights, DEEPSTACK);
  }

  private static void expectConstructorMessage(
      MLXScope scope,
      Map<String, MLXArray> weights,
      int numHeads,
      int[] deepstackIndexes,
      String message) {
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> new Qwen3VlVisionTower(scope, numHeads, weights, deepstackIndexes));
    assertEquals(message, e.getMessage());
  }

  /** One 64x64 processor output: grid (1, 4, 4), 16 raw patch rows of stride 1536. */
  private static Qwen3VlImageProcessorResult processedImage() {
    byte[] pixels = new byte[IMAGE * IMAGE * 3];
    for (int i = 0; i < pixels.length; i++) {
      pixels[i] = (byte) (i * 7 + 13);
    }
    Qwen3VlImageProcessorResult result = processor().process(new RgbImage(IMAGE, IMAGE, pixels));
    assertArrayEquals(new int[] {1, 4, 4}, result.gridThw(), "grid");
    assertEquals(16, result.rawPatchRows(), "raw patch rows");
    assertEquals(1536, result.patchStride(), "patch stride");
    return result;
  }

  /** The 2B artifact's slow-processor options (the golden/peak tests' pinned chain). */
  private static Qwen3VlImageProcessor processor() {
    return new Qwen3VlImageProcessor(
        Qwen3VlProcessorConfig.builder()
            .patchSize(16)
            .temporalPatchSize(2)
            .mergeSize(2)
            .minPixels(3136)
            .maxPixels(4096 * 16 * 16)
            .imageMean(new float[] {0.5f, 0.5f, 0.5f})
            .imageStd(new float[] {0.5f, 0.5f, 0.5f})
            .build());
  }

  private static int[] corners(Qwen3VlVisionTower.PositionLookup lookup, int k) {
    return new int[] {
      lookup.cornerIndices()[0][k],
      lookup.cornerIndices()[1][k],
      lookup.cornerIndices()[2][k],
      lookup.cornerIndices()[3][k]
    };
  }

  private static float[] cornerWeights(Qwen3VlVisionTower.PositionLookup lookup, int k) {
    return new float[] {
      lookup.cornerWeights()[0][k],
      lookup.cornerWeights()[1][k],
      lookup.cornerWeights()[2][k],
      lookup.cornerWeights()[3][k]
    };
  }
}
