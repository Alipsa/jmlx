package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class SmolVlmProcessorConfigTest {

  private static String text() throws Exception {
    return OracleFixtures.configText();
  }

  private static String textOrThrow() {
    try {
      return text();
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static SmolVlmProcessorConfig parse(String json) {
    return SmolVlmProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8));
  }

  private static IllegalArgumentException expectRejection(String json, String message) {
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(json));
    assertEquals(message, e.getMessage());
    return e;
  }

  @Test
  void pinnedConfigParsesToTheSmolVlm256MValues() throws Exception {
    SmolVlmProcessorConfig config = parse(text());
    assertTrue(config.doConvertRgb());
    assertTrue(config.doResize());
    assertEquals(2048, config.sizeLongestEdge());
    assertEquals(Resampling.LANCZOS, config.resample());
    assertTrue(config.doImageSplitting());
    assertEquals(512, config.maxImageSizeLongestEdge());
    assertTrue(config.doRescale());
    assertEquals(0.00392156862745098, config.rescaleFactor());
    assertTrue(config.doNormalize());
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageMean());
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageStd());
    assertTrue(config.doPad());
    assertEquals("Idefics3Processor", config.processorClass());
  }

  @Test
  void meanAndStdAccessorsReturnDefensiveCopies() throws Exception {
    SmolVlmProcessorConfig config = parse(text());
    float[] mean = config.imageMean();
    float[] std = config.imageStd();
    mean[0] = 9f;
    std[0] = 9f;
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageMean());
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageStd());
  }

  @Test
  void resampleNumbersMapToKernels() throws Exception {
    String base = text();
    assertEquals(
        Resampling.LANCZOS, parse(base.replace("\"resample\": 1", "\"resample\": 1")).resample());
    assertEquals(
        Resampling.BILINEAR, parse(base.replace("\"resample\": 1", "\"resample\": 2")).resample());
    assertEquals(
        Resampling.BICUBIC, parse(base.replace("\"resample\": 1", "\"resample\": 3")).resample());
  }

  @Test
  void processorClassIsOptional() throws Exception {
    SmolVlmProcessorConfig config =
        parse(text().replace("  \"processor_class\": \"Idefics3Processor\",\n", ""));
    assertNull(config.processorClass());
  }

  @Test
  void unknownKeysAreRejected() {
    String json =
        textOrThrow().replace("\"resample\": 1,", "\"resample\": 1,\n  \"bogus_option\": 1,");
    expectRejection(json, "preprocessor config has unrecognized key 'bogus_option'");
  }

  @Test
  void missingRequiredKeysAreRejected() {
    expectRejection(
        textOrThrow().replace("  \"resample\": 1,\n", ""),
        "preprocessor config is missing required key 'resample'");
    expectRejection(
        textOrThrow().replace("  \"do_convert_rgb\": true,\n", ""),
        "preprocessor config is missing required key 'do_convert_rgb'");
  }

  @Test
  void wrongProcessorTypeIsRejected() {
    expectRejection(
        textOrThrow().replace("\"Idefics3ImageProcessor\"", "\"SmolVLMImageProcessor\""),
        "preprocessor config image_processor_type must be 'Idefics3ImageProcessor'");
  }

  @Test
  void resampleValuesAreStrict() {
    String base = textOrThrow();
    expectRejection(
        base.replace("\"resample\": 1", "\"resample\": 4"),
        "unsupported resample value 4 (expected 1=LANCZOS, 2=BILINEAR or 3=BICUBIC)");
    expectRejection(
        base.replace("\"resample\": 1", "\"resample\": 0"),
        "unsupported resample value 0 (expected 1=LANCZOS, 2=BILINEAR or 3=BICUBIC)");
    expectRejection(
        base.replace("\"resample\": 1", "\"resample\": 2.5"), "resample must be an integer");
  }

  @Test
  void sizeVariantsAreRejected() {
    String base = textOrThrow();
    String size = "\"size\": {\n    \"longest_edge\": 2048\n  }";
    expectRejection(
        base.replace(size, "\"size\": {\n    \"height\": 2048\n  }"),
        "size must have exactly the key 'longest_edge'");
    expectRejection(
        base.replace(size, "\"size\": {}"), "size must have exactly the key 'longest_edge'");
    expectRejection(base.replace(size, "\"size\": 2048"), "size must be a JSON object");
    expectRejection(
        base.replace(size, "\"size\": {\n    \"longest_edge\": 2.5\n  }"),
        "size.longest_edge must be an integer");
    expectRejection(
        base.replace(size, "\"size\": {\n    \"longest_edge\": 0\n  }"),
        "size.longest_edge must be positive: 0");
    String maxSize = "\"max_image_size\": {\n    \"longest_edge\": 512\n  }";
    expectRejection(
        base.replace(maxSize, "\"max_image_size\": 512"), "max_image_size must be a JSON object");
  }

  @Test
  void rescaleFactorIsStrict() {
    String base = textOrThrow();
    String factor = "\"rescale_factor\": 0.00392156862745098";
    expectRejection(
        base.replace(factor, "\"rescale_factor\": -1"),
        "rescale_factor must be finite and positive: -1.0");
    expectRejection(
        base.replace(factor, "\"rescale_factor\": 0"),
        "rescale_factor must be finite and positive: 0.0");
    expectRejection(
        base.replace(factor, "\"rescale_factor\": \"0.5\""),
        "rescale_factor must be a JSON number");
  }

  @Test
  void meanAndStdAreStrict() {
    String base = textOrThrow();
    String mean = "\"image_mean\": [\n    0.5,\n    0.5,\n    0.5\n  ]";
    String std = "\"image_std\": [\n    0.5,\n    0.5,\n    0.5\n  ]";
    expectRejection(
        base.replace(std, "\"image_std\": [\n    0,\n    0.5,\n    0.5\n  ]"),
        "image_std values must be positive: 0.0");
    expectRejection(
        base.replace(mean, "\"image_mean\": [\n    0.5,\n    0.5\n  ]"),
        "image_mean must be an array of exactly three numbers");
    expectRejection(
        base.replace(mean, "\"image_mean\": [\n    \"0.5\",\n    0.5,\n    0.5\n  ]"),
        "image_mean[0] must be a number");
  }

  @Test
  void booleansAreStrict() {
    expectRejection(
        textOrThrow().replace("\"do_pad\": true", "\"do_pad\": \"true\""),
        "do_pad must be a JSON boolean");
  }

  @Test
  void malformedDocumentsAreRejected() {
    IllegalArgumentException notJson =
        assertThrows(IllegalArgumentException.class, () -> parse("not json"));
    assertTrue(notJson.getMessage().startsWith("preprocessor config is not valid JSON: "));
    expectRejection("[]", "preprocessor config root must be a JSON object");
  }
}
