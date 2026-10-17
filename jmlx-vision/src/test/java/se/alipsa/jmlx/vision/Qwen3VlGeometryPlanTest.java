package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Geometry tests for the Qwen3-VL preprocessing chain. Every expected resize target is
 * venv-verified against the pinned transformers 4.57.6 slow-processor {@code smart_resize} (patch
 * 16, temporal 2, merge 2, {@code min_pixels} 65,536 and the decision-(b) effective {@code
 * max_pixels} 1,048,576 — the value {@code fromJson} derives for the pinned artifact).
 */
class Qwen3VlGeometryPlanTest {

  /** The effective config the pinned artifact parses to: see the config test. */
  private static Qwen3VlProcessorConfig pinnedConfig() {
    return Qwen3VlProcessorConfig.builder()
        .patchSize(16)
        .temporalPatchSize(2)
        .mergeSize(2)
        .minPixels(65536)
        .maxPixels(1048576)
        .imageMean(new float[] {0.5f, 0.5f, 0.5f})
        .imageStd(new float[] {0.5f, 0.5f, 0.5f})
        .build();
  }

  private static Qwen3VlImageProcessor processor() {
    return new Qwen3VlImageProcessor(pinnedConfig());
  }

  private static Qwen3VlImageProcessor noResizeProcessor() {
    return new Qwen3VlImageProcessor(pinnedConfigBuilder().doResize(false).build());
  }

  private static Qwen3VlProcessorConfig.Builder pinnedConfigBuilder() {
    return Qwen3VlProcessorConfig.builder()
        .patchSize(16)
        .temporalPatchSize(2)
        .mergeSize(2)
        .minPixels(65536)
        .maxPixels(1048576)
        .imageMean(new float[] {0.5f, 0.5f, 0.5f})
        .imageStd(new float[] {0.5f, 0.5f, 0.5f});
  }

  private static void assertPlan(
      Qwen3VlGeometryPlan plan,
      int inputWidth,
      int inputHeight,
      int resizedWidth,
      int resizedHeight,
      int gridH,
      int gridW,
      int rawPatchRows,
      int mergedRowCount) {
    assertEquals(inputWidth, plan.inputWidth());
    assertEquals(inputHeight, plan.inputHeight());
    assertEquals(resizedWidth, plan.resizedWidth());
    assertEquals(resizedHeight, plan.resizedHeight());
    assertEquals(1, plan.gridT());
    assertEquals(gridH, plan.gridH());
    assertEquals(gridW, plan.gridW());
    assertEquals(rawPatchRows, plan.rawPatchRows());
    assertEquals(mergedRowCount, plan.mergedRowCount());
    assertEquals(1536, plan.patchStride());
    assertEquals(resizedHeight + "x" + resizedWidth, plan.resizedHeightWidth());
  }

  @Test
  void resizeGeometryMatchesTheReferenceForTheVerifiedCases() {
    Qwen3VlImageProcessor processor = processor();
    // All (resized H x W, grid, raw, merged) tuples below are the venv-verified outputs of the
    // pinned smart_resize(width-first here) for the pinned artifact numbers.
    assertPlan(processor.geometry(1920, 1080), 1920, 1080, 1344, 768, 48, 84, 4032, 1008);
    assertPlan(processor.geometry(1280, 960), 1280, 960, 1152, 864, 54, 72, 3888, 972);
    assertPlan(processor.geometry(1080, 1920), 1080, 1920, 768, 1344, 84, 48, 4032, 1008);
    // 1024x1024 plans to exactly the hard pixel cap (4096 rows x 256 pixels).
    assertPlan(processor.geometry(1024, 1024), 1024, 1024, 1024, 1024, 64, 64, 4096, 1024);
    // Max-pixels branch: the 3264 raw rows are the result of the port's two sequential divisions
    // (height / beta / factor). A full venv sweep of [2, 4000)^2 (12,087,517 cells reaching the
    // max-pixels branch) plus squares in [1000, 20000) step 16 found zero cases where the folded
    // form height / (beta * factor) differs, so the left-associative order is kept as the
    // reference's, though no regression case for it exists in these ranges.
    assertPlan(processor.geometry(160, 6560), 160, 6560, 128, 6528, 408, 8, 3264, 816);
    // Exactly at min_pixels: no branch fires.
    assertPlan(processor.geometry(256, 256), 256, 256, 256, 256, 16, 16, 256, 64);
    // Min-pixels branch (the 64x32 WP1 probe image).
    assertPlan(processor.geometry(64, 32), 64, 32, 384, 192, 12, 24, 288, 72);
    assertPlan(processor.geometry(8, 8), 8, 8, 256, 256, 16, 16, 256, 64);
    // Banker's-tie inputs. 144/32 = 4.5: Math.rint (Python round) ties to even 4, so the
    // pre-branch plan is 128x128 = 16,384 pixels; 176/32 = 5.5 ties to even 6, pre-branch
    // 192x192 = 36,864 pixels. Both fall below the pinned min_pixels of 65,536, so the
    // min-pixels branch scales both to 256x256, venv-verified. (The spec's 128x128 / 192x192
    // expectations hold only under the reference __init__ bug's 3,136-pixel default min; see
    // findings §4.2.)
    assertPlan(processor.geometry(144, 144), 144, 144, 256, 256, 16, 16, 256, 64);
    assertPlan(processor.geometry(176, 176), 176, 176, 256, 256, 16, 16, 256, 64);
    // 1040/32 = 32.5 ties to even 32: the pre-branch plan is 1024x1024, exactly at the cap, so
    // no branch fires even though the raw 1040x1040 area exceeds it (the reference compares the
    // rounded h_bar * w_bar, not the raw area).
    assertPlan(processor.geometry(1040, 1040), 1040, 1040, 1024, 1024, 64, 64, 4096, 1024);
    // Aspect ratio exactly 200 is accepted (the guard is strict: greater than 200). The min-pixels
    // branch rescales the short edge up to 32 and the long one to 3648 (venv-verified).
    assertPlan(processor.geometry(2000, 10), 2000, 10, 3648, 32, 2, 228, 456, 114);
  }

  @Test
  void aspectRatioAboveTwoHundredIsRejected() {
    // 2002/10 = 200.2: the message carries the ratio, matching the reference's ValueError text.
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> processor().geometry(2002, 10));
    assertEquals("absolute aspect ratio must be smaller than 200, got 200.2", e.getMessage());
    e = assertThrows(IllegalArgumentException.class, () -> processor().geometry(10, 2002));
    assertEquals("absolute aspect ratio must be smaller than 200, got 200.2", e.getMessage());
  }

  @Test
  void nonPositiveDimensionsAreRejected() {
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> processor().geometry(0, 64));
    assertEquals("image dimensions must be positive: 0x64", e.getMessage());
    e = assertThrows(IllegalArgumentException.class, () -> processor().geometry(64, 0));
    assertEquals("image dimensions must be positive: 64x0", e.getMessage());
    e = assertThrows(IllegalArgumentException.class, () -> processor().geometry(-1, 64));
    assertEquals("image dimensions must be positive: -1x64", e.getMessage());
  }

  @Test
  void noResizeGeometryKeepsTheInputDimensions() {
    Qwen3VlGeometryPlan plan = noResizeProcessor().geometry(96, 64);
    assertPlan(plan, 96, 64, 96, 64, 4, 6, 24, 6);
  }

  @Test
  void noResizeRequiresDivisibilityByTheFactor() {
    // 100 is not divisible by patch_size at all; the ported requirement (patch_size * merge_size
    // = 32) names the offending dimension, the height being checked first.
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> noResizeProcessor().geometry(100, 100));
    assertEquals(
        "height must be divisible by patch_size * merge_size = 32 when resizing is disabled: 100",
        e.getMessage());
    // 176 is divisible by 16 (grid side 11) but not by the factor: the reference's reshape would
    // fail at grid_h // merge_size = 11 / 2, and the factor check rejects it first.
    e = assertThrows(IllegalArgumentException.class, () -> noResizeProcessor().geometry(176, 176));
    assertEquals(
        "height must be divisible by patch_size * merge_size = 32 when resizing is disabled: 176",
        e.getMessage());
    // Width-side offenses when the height passes the check.
    e = assertThrows(IllegalArgumentException.class, () -> noResizeProcessor().geometry(90, 64));
    assertEquals(
        "width must be divisible by patch_size * merge_size = 32 when resizing is disabled: 90",
        e.getMessage());
    e = assertThrows(IllegalArgumentException.class, () -> noResizeProcessor().geometry(96, 60));
    assertEquals(
        "height must be divisible by patch_size * merge_size = 32 when resizing is disabled: 60",
        e.getMessage());
  }

  @Test
  void noResizeAcceptsFactorDivisibleInputs() {
    // 160x160: grid 10x10, merge blocks 5x5, 25 merged rows; 100 raw rows, no budget issue.
    assertPlan(noResizeProcessor().geometry(160, 160), 160, 160, 160, 160, 10, 10, 100, 25);
  }

  @Test
  void noResizeStillEnforcesTheRawPatchRowBudget() {
    // 160x6560 is divisible by the factor, but plans 410 * 10 = 4100 raw rows > 4096.
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> noResizeProcessor().geometry(160, 6560));
    assertEquals(
        "raw patch rows 4100 exceed the raw patch row budget of 4096 for 160x6560", e.getMessage());
  }

  @Test
  void planRecordRejectsInconsistentConstruction() {
    // gridT is always 1 for a still image.
    assertThrows(
        IllegalArgumentException.class,
        () -> new Qwen3VlGeometryPlan(64, 32, 384, 192, 2, 12, 24, 288, 72, 1536));
    assertThrows(
        IllegalArgumentException.class,
        () -> new Qwen3VlGeometryPlan(0, 32, 384, 192, 1, 12, 24, 288, 72, 1536));
    assertThrows(
        IllegalArgumentException.class,
        () -> new Qwen3VlGeometryPlan(64, 32, 0, 192, 1, 12, 24, 288, 72, 1536));
    // rawPatchRows must equal gridH * gridW.
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> new Qwen3VlGeometryPlan(64, 32, 384, 192, 1, 16, 16, 200, 50, 1536));
    assertEquals("rawPatchRows must equal gridH * gridW (16 * 16): 200", e.getMessage());
    // rawPatchRows must be a positive multiple of mergedRowCount.
    e =
        assertThrows(
            IllegalArgumentException.class,
            () -> new Qwen3VlGeometryPlan(64, 32, 384, 192, 1, 12, 24, 288, 70, 1536));
    assertEquals(
        "rawPatchRows must be a positive multiple of mergedRowCount: 288 / 70", e.getMessage());
    // ... and the quotient must be a perfect square (merge_size^2).
    e =
        assertThrows(
            IllegalArgumentException.class,
            () -> new Qwen3VlGeometryPlan(96, 64, 96, 64, 1, 4, 6, 24, 2, 1536));
    assertEquals(
        "rawPatchRows / mergedRowCount must be merge_size^2 for some positive merge_size: 12",
        e.getMessage());
    // patchStride is 3 * temporal_patch_size * patch_size^2, hence a multiple of 3.
    e =
        assertThrows(
            IllegalArgumentException.class,
            () -> new Qwen3VlGeometryPlan(64, 32, 384, 192, 1, 12, 24, 288, 72, 1535));
    assertTrue(
        e.getMessage()
            .startsWith(
                "patchStride must be a positive multiple of 3 (3 * temporal_patch_size * patch_size"
                    + " * patch_size): 1535"),
        "unexpected message: " + e.getMessage());
  }
}
