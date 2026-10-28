package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Geometry tests for the Phi-3.5-Vision HD preprocessing chain. Every expected plan is
 * venv-verified against the pinned bundled processor (the {@code phi3v} image-oracle family, {@code
 * tools/image-oracle/fixtures/phi3v.expected.json}) for the pinned artifact numbers (CLIP mean/std,
 * {@code num_crops} 4).
 */
class Phi3VGeometryPlanTest {

  /** The effective config the pinned artifact parses to: see the config test. */
  private static Phi3VImageProcessor processor() {
    return new Phi3VImageProcessor(
        Phi3VProcessorConfig.builder()
            .numCrops(4)
            .imageMean(new float[] {0.48145466f, 0.4578275f, 0.40821073f})
            .imageStd(new float[] {0.26862954f, 0.26130258f, 0.27577711f})
            .build());
  }

  private static void assertPlan(
      Phi3VGeometryPlan plan,
      boolean transposed,
      int resizedWidth,
      int resizedHeight,
      int paddedWidth,
      int paddedHeight,
      int gridH,
      int gridW,
      int cropBlocks,
      int numImgTokens) {
    assertEquals(transposed, plan.transposed());
    assertEquals(resizedWidth, plan.resizedWidth());
    assertEquals(resizedHeight, plan.resizedHeight());
    assertEquals(paddedWidth, plan.paddedWidth());
    assertEquals(paddedHeight, plan.paddedHeight());
    assertEquals(gridH, plan.gridH());
    assertEquals(gridW, plan.gridW());
    assertEquals(cropBlocks, plan.cropBlocks());
    assertEquals(4, plan.numCrops());
    assertEquals(5, plan.slotCount());
    assertEquals(numImgTokens, plan.numImgTokens());
    assertArrayEquals(
        new int[] {paddedHeight, paddedWidth}, plan.imageSize(), "image_sizes order [h, w]");
  }

  @Test
  void geometryMatchesTheReferenceForTheVerifiedCases() {
    Phi3VImageProcessor processor = processor();
    // square 336x336: ratio 1, scale 2, resize 672x672, no padding.
    assertPlan(processor.geometry(336, 336), false, 672, 672, 672, 672, 2, 2, 4, 757);
    // wide 672x336: ratio 2, scale 2, resize 672x336, short side already a multiple of 336.
    assertPlan(processor.geometry(672, 336), false, 672, 336, 672, 336, 1, 2, 2, 457);
    // tall 336x672: transposed working space 672x336, transposed back to 336x672.
    assertPlan(processor.geometry(336, 672), true, 336, 672, 336, 672, 2, 1, 2, 469);
    // trunc 300x301: transposed working 301x300, ratio 1.003333, scale 2, long 672,
    // short int(672 / 1.003333) = 669, padded to 672 (top 1, bottom 2) after the transpose back.
    assertPlan(processor.geometry(300, 301), true, 669, 672, 672, 672, 2, 2, 4, 757);
  }

  @Test
  void extremeAspectRatioIsRejectedInBothOrientations() {
    Phi3VImageProcessor processor = processor();
    // num_crops * 336 = 1344: above it the double-truncating short side would truncate to 0.
    IllegalArgumentException portrait =
        assertThrows(IllegalArgumentException.class, () -> processor.geometry(1, 1345));
    assertTrue(portrait.getMessage().contains("1345.0"), portrait.getMessage());
    IllegalArgumentException landscape =
        assertThrows(IllegalArgumentException.class, () -> processor.geometry(1345, 1));
    assertTrue(landscape.getMessage().contains("1345.0"), landscape.getMessage());
    // The bound itself is valid: ratio 1344 gives scale 4, long 1344, short int(1344/1344) = 1,
    // padded to 336 -- the reachable set's 781-token corner (gridH 4, one block per row).
    assertPlan(processor.geometry(1, 1344), true, 1, 1344, 336, 1344, 4, 1, 4, 781);
    assertPlan(processor.geometry(1344, 1), false, 1344, 1, 1344, 336, 1, 4, 4, 745);
  }

  @Test
  void nonPositiveDimensionsAreRejected() {
    Phi3VImageProcessor processor = processor();
    assertThrows(IllegalArgumentException.class, () -> processor.geometry(0, 336));
    assertThrows(IllegalArgumentException.class, () -> processor.geometry(336, 0));
    assertThrows(IllegalArgumentException.class, () -> processor.geometry(-1, 336));
  }

  @Test
  void constructorRejectsInconsistentPlans() {
    // A consistent baseline (the verified square-336x336 plan).
    int[] base = {336, 336, 672, 672, 672, 672, 2, 2, 4, 4, 5, 757};
    assertEquals(
        5,
        new Phi3VGeometryPlan(
                base[0], base[1], false, base[2], base[3], base[4], base[5], base[6], base[7],
                base[8], base[9], base[10], base[11])
            .slotCount());

    // transposed must equal (inputWidth < inputHeight).
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(672, 336, true, 672, 336, 672, 336, 1, 2, 2, 4, 5, 457));
    // Padded sides must be positive multiples of 336.
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(672, 336, false, 672, 336, 670, 336, 0, 2, 0, 4, 5, 457));
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(336, 672, true, 336, 672, 336, 670, 1, 1, 1, 4, 5, 469));
    // Padded sides must be at least the resized sides.
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(672, 336, false, 672, 336, 336, 336, 1, 1, 1, 4, 5, 457));
    // The long side is never padded.
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(672, 336, false, 672, 336, 1008, 336, 1, 3, 3, 4, 5, 469));
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(336, 672, true, 336, 672, 336, 1008, 3, 1, 3, 4, 5, 469));
    // Grid and block-count consistency.
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(336, 336, false, 672, 672, 672, 672, 3, 2, 6, 4, 5, 757));
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(336, 336, false, 672, 672, 672, 672, 2, 2, 4, 4, 4, 757));
    // The scale loop bounds the block count by num_crops.
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(336, 336, false, 1008, 672, 1008, 672, 2, 3, 6, 4, 5, 757));
    // The slot count and token formula are derived, not free.
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(336, 336, false, 672, 672, 672, 672, 2, 2, 4, 4, 4, 757));
    assertThrows(
        IllegalArgumentException.class,
        () -> new Phi3VGeometryPlan(336, 336, false, 672, 672, 672, 672, 2, 2, 4, 4, 5, 700));
    // The imageSize accessor reports the padded [height, width].
    assertArrayEquals(
        new int[] {672, 672},
        new Phi3VGeometryPlan(336, 336, false, 672, 672, 672, 672, 2, 2, 4, 4, 5, 757).imageSize());
  }
}
