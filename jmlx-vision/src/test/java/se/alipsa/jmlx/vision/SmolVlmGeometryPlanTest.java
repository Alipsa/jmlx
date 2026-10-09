package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * The pure geometry plan against the pinned SmolVLM-256M configuration, including the three image
 * sizes of the committed chat golden, and agreement with the processor's pixel chain.
 */
class SmolVlmGeometryPlanTest {

  private static SmolVlmProcessorConfig pinnedConfig() {
    try {
      return SmolVlmProcessorConfig.fromJson(
          OracleFixtures.configText().getBytes(StandardCharsets.UTF_8));
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static SmolVlmProcessorConfig config(String replacementFrom, String replacementTo) {
    try {
      String json = OracleFixtures.configText();
      return SmolVlmProcessorConfig.fromJson(
          json.replace(replacementFrom, replacementTo).getBytes(StandardCharsets.UTF_8));
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static RgbImage blank(int width, int height) {
    return new RgbImage(width, height, new byte[width * height * 3]);
  }

  @Test
  void pinnedConfigPlansTheChatGoldenGrids() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(pinnedConfig());

    // tile-4x4-100x80: stage 1 1638x2048 (even-rounded short edge); stage 2 recomputes the
    // short edge from the rounded aspect (2048/1638 -> 1638) and rounds it up to 2048.
    SmolVlmGeometryPlan p44 = processor.geometry(80, 100);
    assertEquals(new SmolVlmGeometryPlan(1638, 2048, 2048, 2048, 4, 4), p44);
    assertTrue(p44.split());
    assertEquals(17, p44.tileCount());

    // tile-1x4-2048x512 and tile-4x1-512x2048: the stage-1 longest edge is already 2048, so
    // stage 1 and stage 2 agree and only one axis exceeds the 512 tile.
    assertEquals(
        new SmolVlmGeometryPlan(512, 2048, 512, 2048, 1, 4), processor.geometry(512, 2048));
    assertEquals(
        new SmolVlmGeometryPlan(2048, 512, 2048, 512, 4, 1), processor.geometry(2048, 512));
    assertEquals(5, processor.geometry(512, 2048).tileCount());
  }

  @Test
  void smallImagesUpscaleToTheLongestEdgeAndSplit() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(pinnedConfig());

    // 512x512 upscales to 2048x2048: the longest edge is always exactly size.longest_edge after
    // stage 1, so every split image has at least four tiles along its longest axis.
    assertEquals(
        new SmolVlmGeometryPlan(2048, 2048, 2048, 2048, 4, 4), processor.geometry(512, 512));
    assertEquals(new SmolVlmGeometryPlan(2048, 2048, 2048, 2048, 4, 4), processor.geometry(10, 10));
  }

  @Test
  void noSplitAndNoResizeVariants() {
    // Splitting disabled: (0, 0) grid, no stage-2 size, one tile.
    SmolVlmImageProcessor noSplit =
        new SmolVlmImageProcessor(
            config("\"do_image_splitting\": true", "\"do_image_splitting\": false"));
    SmolVlmGeometryPlan plan = noSplit.geometry(80, 100);
    assertEquals(new SmolVlmGeometryPlan(1638, 2048, 0, 0, 0, 0), plan);
    assertFalse(plan.split());
    assertEquals(1, plan.tileCount());
    SmolVlmImageProcessor.ChainIntermediates chain = noSplit.intermediates(blank(100, 80));
    assertEquals(null, chain.stage2());
    assertEquals(0, chain.rows());
    assertEquals(0, chain.cols());
    assertEquals(plan.tileCount(), chain.frames().size());

    // Resize disabled: stage 1 is the input itself; stage 2 and the grid still follow it.
    SmolVlmImageProcessor noResize =
        new SmolVlmImageProcessor(config("\"do_resize\": true", "\"do_resize\": false"));
    assertEquals(new SmolVlmGeometryPlan(80, 100, 512, 512, 0, 0), noResize.geometry(80, 100));
  }

  @Test
  void geometryAgreesWithThePixelChain() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(pinnedConfig());
    for (int[] size : new int[][] {{100, 80}, {2048, 512}, {512, 2048}, {512, 512}, {3, 777}}) {
      SmolVlmGeometryPlan plan = processor.geometry(size[1], size[0]);
      SmolVlmImageProcessor.ChainIntermediates chain =
          processor.intermediates(blank(size[0], size[1]));
      assertEquals(
          plan.stage1Height(),
          chain.stage1().height(),
          "stage 1 height " + size[0] + "x" + size[1]);
      assertEquals(
          plan.stage1Width(), chain.stage1().width(), "stage 1 width " + size[0] + "x" + size[1]);
      if (plan.split()) {
        assertEquals(plan.stage2Height(), chain.stage2().height());
        assertEquals(plan.stage2Width(), chain.stage2().width());
      } else {
        assertEquals(null, chain.stage2());
      }
      assertEquals(plan.rows(), chain.rows(), "rows " + size[0] + "x" + size[1]);
      assertEquals(plan.cols(), chain.cols(), "cols " + size[0] + "x" + size[1]);
      assertEquals(plan.tileCount(), chain.frames().size());
    }
  }

  @Test
  void geometryRejectsNonPositiveDimensions() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(pinnedConfig());
    assertEquals(
        "image dimensions must be positive: 0x10",
        assertThrows(IllegalArgumentException.class, () -> processor.geometry(10, 0)).getMessage());
    assertThrows(IllegalArgumentException.class, () -> processor.geometry(-1, 10));
  }

  @Test
  void planRecordValidatesItsOwnConsistency() {
    assertThrows(
        IllegalArgumentException.class, () -> new SmolVlmGeometryPlan(10, 10, 10, 10, 2, 0));
    assertThrows(
        IllegalArgumentException.class, () -> new SmolVlmGeometryPlan(10, 10, 10, 10, 0, 2));
    assertThrows(
        IllegalArgumentException.class, () -> new SmolVlmGeometryPlan(0, 10, 10, 10, 0, 0));
    assertThrows(
        IllegalArgumentException.class, () -> new SmolVlmGeometryPlan(10, 10, -1, 10, 0, 0));
    // A split grid requires a positive stage-2 size.
    assertThrows(IllegalArgumentException.class, () -> new SmolVlmGeometryPlan(10, 10, 0, 0, 1, 1));
    // A no-split plan may carry a stage-2 size (splitting enabled but not needed) or 0x0
    // (splitting disabled).
    assertEquals(1, new SmolVlmGeometryPlan(10, 10, 512, 512, 0, 0).tileCount());
    assertEquals(1, new SmolVlmGeometryPlan(10, 10, 0, 0, 0, 0).tileCount());
    assertEquals(17, new SmolVlmGeometryPlan(10, 10, 2048, 2048, 4, 4).tileCount());
  }
}
