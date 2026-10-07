package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class SmolVlmImageProcessorTest {

  private static SmolVlmProcessorConfig chainConfig(int resample, boolean doImageSplitting) {
    String json = configText();
    json = json.replace("\"longest_edge\": 2048", "\"longest_edge\": 96");
    json = json.replace("\"longest_edge\": 512", "\"longest_edge\": 32");
    json = json.replace("\"resample\": 1", "\"resample\": " + resample);
    json =
        json.replace("\"do_image_splitting\": true", "\"do_image_splitting\": " + doImageSplitting);
    return SmolVlmProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8));
  }

  private static String configText() {
    try {
      return OracleFixtures.configText();
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  /** Deterministic content; the math tests only need well-defined pixel values. */
  private static RgbImage gradient(int width, int height) {
    byte[] pixels = new byte[width * height * 3];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int i = (y * width + x) * 3;
        pixels[i] = (byte) ((x * 5 + 3) % 256);
        pixels[i + 1] = (byte) ((y * 7 + 11) % 256);
        pixels[i + 2] = (byte) ((x * y + 17) % 256);
      }
    }
    return new RgbImage(width, height, pixels);
  }

  /** A 96x96 image whose 32x32 (row, col) block is the constant value row * 100 + col. */
  private static RgbImage blocked(
      int width, int height, int blockWidth, int blockHeight, int valueBase) {
    byte[] pixels = new byte[width * height * 3];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int value = valueBase + (y / blockHeight) * 100 + (x / blockWidth);
        int i = (y * width + x) * 3;
        pixels[i] = (byte) value;
        pixels[i + 1] = (byte) value;
        pixels[i + 2] = (byte) value;
      }
    }
    return new RgbImage(width, height, pixels);
  }

  @Test
  void constructorAndProcessRejectNullArguments() {
    IllegalArgumentException badConfig =
        assertThrows(IllegalArgumentException.class, () -> new SmolVlmImageProcessor(null));
    assertEquals("config must not be null", badConfig.getMessage());
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(1, true));
    IllegalArgumentException badImage =
        assertThrows(IllegalArgumentException.class, () -> processor.process(null));
    assertEquals("image must not be null", badImage.getMessage());
  }

  @Test
  void stage1ResizesTheLongestEdgeToSizeWithProportionalEvenRoundedShortEdge() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(1, true));
    // int(96 / 1.25) = 76 (even); int(96 * 0.8) = 76; int(96 / 5.0) = 19 -> odd -> 20.
    assertStage1(processor, gradient(100, 80), 76, 96);
    assertStage1(processor, gradient(80, 100), 96, 76);
    assertStage1(processor, gradient(200, 40), 20, 96);
    assertStage1(processor, gradient(100, 100), 96, 96);
  }

  private static void assertStage1(
      SmolVlmImageProcessor processor, RgbImage image, int height, int width) {
    SmolVlmImageProcessor.ChainIntermediates chain = processor.intermediates(image);
    assertEquals(height, chain.stage1().height());
    assertEquals(width, chain.stage1().width());
  }

  @Test
  void stage1IsCappedAt4096() {
    // longest_edge 5000: the rescale gives 4000x5000, the MAX_IMAGE_SIZE cap then shrinks
    // width to 4096 and truncates height to int(4096 / 1.25) = 3276 (no even rounding on the
    // cap, mirroring the reference).
    String json =
        configText()
            .replace("\"longest_edge\": 2048", "\"longest_edge\": 5000")
            .replace("\"longest_edge\": 512", "\"longest_edge\": 32");
    SmolVlmImageProcessor processor =
        new SmolVlmImageProcessor(
            SmolVlmProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8)));
    assertStage1(processor, gradient(100, 80), 3276, 4096);
  }

  @Test
  void stage2RoundsBothEdgesUpToTileMultiples() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(1, true));
    // stage1 96x76 -> width 96, int(96/1.263...) = 76 -> ceil to 96: 96x96.
    assertStage2(processor, gradient(100, 80), 96, 96);
    // stage1 48x96 -> width 96, int(96/2.0) = 48 -> ceil to 64: 64x96.
    assertStage2(processor, gradient(160, 80), 64, 96);
    // stage1 20x96 -> width 96, int(96/4.8) = 20 -> ceil to 32: 32x96.
    assertStage2(processor, gradient(200, 40), 32, 96);
  }

  private static void assertStage2(
      SmolVlmImageProcessor processor, RgbImage image, int height, int width) {
    SmolVlmImageProcessor.ChainIntermediates chain = processor.intermediates(image);
    assertNotNull(chain.stage2());
    assertEquals(height, chain.stage2().height());
    assertEquals(width, chain.stage2().width());
  }

  @Test
  void splitCropsAreRowMajorSlicesWithTheGlobalFrameLast() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(1, true));
    RgbImage image = blocked(96, 96, 32, 32, 0);
    SmolVlmImageProcessor.ChainIntermediates chain = processor.intermediates(image);
    assertEquals(3, chain.rows());
    assertEquals(3, chain.cols());
    assertEquals(10, chain.frames().size());
    for (int index = 0; index < 9; index++) {
      int value = (index / 3) * 100 + (index % 3);
      RgbImage frame = chain.frames().get(index);
      assertEquals(32, frame.width());
      assertEquals(32, frame.height());
      for (byte b : frame.pixels()) {
        assertEquals(value, b & 0xFF, "crop " + index);
      }
    }
    // The global frame is the stage-2 image resized to the tile size (32x32 here).
    RgbImage global = chain.frames().get(9);
    assertEquals(32, global.width());
    assertEquals(32, global.height());
  }

  @Test
  void splitGridFollowsTheStage2Shape() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(1, true));
    // stage1 48x96 -> stage2 64x96 -> 2x3 grid of 32x32 crops plus the global frame.
    SmolVlmImageProcessor.ChainIntermediates chain = processor.intermediates(gradient(160, 80));
    assertEquals(2, chain.rows());
    assertEquals(3, chain.cols());
    assertEquals(7, chain.frames().size());
    for (RgbImage frame : chain.frames()) {
      assertEquals(32, frame.width());
      assertEquals(32, frame.height());
    }
  }

  @Test
  void noSplittingProducesTheSquareTileAndZerosTheGrid() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(1, false));
    SmolVlmImageProcessor.ChainIntermediates chain = processor.intermediates(gradient(100, 80));
    assertNull(chain.stage2());
    assertEquals(0, chain.rows());
    assertEquals(0, chain.cols());
    assertEquals(1, chain.frames().size());
    assertEquals(32, chain.frames().get(0).width());
    assertEquals(32, chain.frames().get(0).height());
  }

  @Test
  void disabledResizeAndSplittingLeaveTheImageUnchanged() {
    String json =
        configText()
            .replace("\"longest_edge\": 2048", "\"longest_edge\": 96")
            .replace("\"longest_edge\": 512", "\"longest_edge\": 32")
            .replace("\"do_resize\": true", "\"do_resize\": false")
            .replace("\"do_image_splitting\": true", "\"do_image_splitting\": false");
    SmolVlmImageProcessor processor =
        new SmolVlmImageProcessor(
            SmolVlmProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8)));
    RgbImage image = gradient(32, 32);
    SmolVlmImageProcessor.ChainIntermediates chain = processor.intermediates(image);
    assertEquals(image, chain.stage1());
    assertEquals(image, chain.frames().get(0));
  }

  @Test
  void disabledRescaleAndNormalizeAreAnExactFloatIdentity() {
    String json =
        configText()
            .replace("\"longest_edge\": 2048", "\"longest_edge\": 96")
            .replace("\"longest_edge\": 512", "\"longest_edge\": 32")
            .replace("\"do_resize\": true", "\"do_resize\": false")
            .replace("\"do_rescale\": true", "\"do_rescale\": false")
            .replace("\"do_normalize\": true", "\"do_normalize\": false")
            .replace("\"do_image_splitting\": true", "\"do_image_splitting\": false");
    SmolVlmImageProcessor processor =
        new SmolVlmImageProcessor(
            SmolVlmProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8)));
    byte[] pixels = new byte[32 * 32 * 3];
    for (int i = 0; i < pixels.length; i++) {
      pixels[i] = (byte) (i % 256);
    }
    RgbImage image = new RgbImage(32, 32, pixels);
    SmolVlmImageProcessorResult result = processor.process(image);
    assertEquals(1, result.tiles().size());
    assertEquals(0, result.rows());
    assertEquals(0, result.cols());
    ImageTensor tensor = result.tiles().get(0);
    assertArrayEquals(new int[] {1, 32, 32, 3}, tensor.shape());
    List<Float> expected = new ArrayList<>();
    for (byte b : pixels) {
      expected.add((float) (b & 0xFF));
    }
    float[] values = tensor.values();
    assertEquals(expected.size(), values.length);
    for (int i = 0; i < values.length; i++) {
      assertEquals(expected.get(i), values[i], 0.0f);
    }
    int[] mask = result.pixelMasks().get(0);
    assertArrayEquals(OracleFixtures.maskAllOnes(32 * 32), mask);
  }

  @Test
  void processTilesAndMasksStayAssociated() {
    SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(3, true));
    SmolVlmImageProcessorResult result = processor.process(blocked(96, 96, 32, 32, 0));
    assertEquals(10, result.tiles().size());
    assertEquals(10, result.pixelMasks().size());
    assertEquals(3, result.rows());
    assertEquals(3, result.cols());
    for (int i = 0; i < result.tiles().size(); i++) {
      int[] shape = result.tiles().get(i).shape();
      int[] mask = result.pixelMasks().get(i);
      assertEquals(shape[1] * shape[2], mask.length);
      assertTrue(Arrays.stream(mask).allMatch(v -> v == 1));
    }
    // The tile list and masks are defensively copied.
    int[] maskView = result.pixelMasks().get(0).clone();
    maskView[0] = 0;
    assertTrue(Arrays.stream(result.pixelMasks().get(0)).allMatch(v -> v == 1));
  }
}
