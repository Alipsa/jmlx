package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;
import org.junit.jupiter.api.Test;

class Qwen3VlProcessorConfigTest {

  /**
   * The pinned Qwen3-VL-2B-Instruct {@code preprocessor_config.json}, byte-identical to the
   * artifact file (4-space indent, no trailing newline); the SHA-256 is the one recorded in the WP1
   * probe goldens ({@code tools/hf-reference/goldens/qwen3-vl-2b-wp1-probe.json}).
   */
  private static final String BASE =
      """
      {
          "size": {
              "longest_edge": 16777216,
              "shortest_edge": 65536
          },
          "patch_size": 16,
          "temporal_patch_size": 2,
          "merge_size": 2,
          "image_mean": [
              0.5,
              0.5,
              0.5
          ],
          "image_std": [
              0.5,
              0.5,
              0.5
          ],
          "processor_class": "Qwen3VLProcessor",
          "image_processor_type": "Qwen2VLImageProcessorFast"
      }
      """;

  private static final String SIZE_BLOCK =
      """
          "size": {
              "longest_edge": 16777216,
              "shortest_edge": 65536
          },
      """;

  private static final String MEAN_BLOCK =
      """
          "image_mean": [
              0.5,
              0.5,
              0.5
          ],
      """;

  private static final String STD_BLOCK =
      """
          "image_std": [
              0.5,
              0.5,
              0.5
          ],
      """;

  private static Qwen3VlProcessorConfig parse(String json) {
    return Qwen3VlProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8));
  }

  private static IllegalArgumentException expectRejection(String json, String message) {
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(json));
    assertEquals(message, e.getMessage());
    return e;
  }

  private static String hex(byte[] bytes) {
    StringBuilder sb = new StringBuilder();
    for (byte b : bytes) {
      sb.append(String.format("%02x", b));
    }
    return sb.toString();
  }

  private static String sha256(String text) {
    try {
      return hex(
          MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8)));
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static Qwen3VlProcessorConfig.Builder baseBuilder() {
    return Qwen3VlProcessorConfig.builder()
        .patchSize(16)
        .temporalPatchSize(2)
        .mergeSize(2)
        .minPixels(65536)
        .maxPixels(1048576)
        .imageMean(new float[] {0.5f, 0.5f, 0.5f})
        .imageStd(new float[] {0.5f, 0.5f, 0.5f});
  }

  @Test
  void pinnedJsonIsByteIdenticalToTheTierBArtifact() {
    // The text block carries a trailing newline the 390-byte artifact file lacks; strip it for
    // the byte-identical check (the SHA-256 is the one recorded in the WP1 probe goldens,
    // tools/hf-reference/goldens/qwen3-vl-2b-wp1-probe.json, and §3.4 of the findings).
    assertEquals(
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        sha256(BASE.stripTrailing()));
  }

  @Test
  void pinnedConfigParsesToTheQwen3VlValues() {
    Qwen3VlProcessorConfig config = parse(BASE);
    assertEquals(16, config.patchSize());
    assertEquals(2, config.temporalPatchSize());
    assertEquals(2, config.mergeSize());
    assertEquals(65536, config.minPixels());
    // Decision (b): the declared 16,777,216-pixel ceiling is capped at the hard pixel cap of the
    // default 4,096-row budget, not rejected; the pre-cap value stays available as declared.
    assertEquals(1048576, config.maxPixels());
    assertEquals(16777216, config.declaredMaxPixels());
    assertEquals(4096, config.rawRowBudget());
    assertEquals(1048576, config.hardCapPixels());
    assertTrue(config.doResize());
    assertTrue(config.doConvertRgb());
    // No "resample" key in the artifact: the reference default is BICUBIC.
    assertEquals(Resampling.BICUBIC, config.resample());
    assertEquals(1.0 / 255.0, config.rescaleFactor());
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageMean());
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageStd());
    assertEquals("Qwen3VLProcessor", config.processorClass());
    assertEquals(32, config.factor());
    assertEquals(1536, config.patchStride());
  }

  @Test
  void bothPinnedProcessorTypesAreAccepted() {
    // The artifact declares the Fast type; the slow PIL processor is the port target, so both
    // declarations must parse (a documented deviation, findings §4.3).
    parse(BASE);
    parse(BASE.replace("Qwen2VLImageProcessorFast", "Qwen2VLImageProcessor"));
  }

  @Test
  void wrongProcessorTypeIsRejected() {
    expectRejection(
        BASE.replace("Qwen2VLImageProcessorFast", "Qwen2_5_VLImageProcessor"),
        "qwen3-vl preprocessor config image_processor_type must be 'Qwen2VLImageProcessor' or "
            + "'Qwen2VLImageProcessorFast'");
  }

  @Test
  void meanAndStdAccessorsReturnDefensiveCopies() {
    Qwen3VlProcessorConfig config = parse(BASE);
    float[] mean = config.imageMean();
    float[] std = config.imageStd();
    mean[0] = 9f;
    std[0] = 9f;
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageMean());
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, config.imageStd());
  }

  @Test
  void unknownKeysAreRejected() {
    expectRejection(
        BASE.replace("\"patch_size\": 16,", "\"patch_size\": 16,\n    \"bogus_option\": 1,"),
        "qwen3-vl preprocessor config has unrecognized key 'bogus_option'");
  }

  @Test
  void fullDumpFormWithInertKeysParsesToTheSameConfig() {
    // The pinned 8B 4-bit artifact dumps the image processor's complete field set; every inert
    // key is null and default_to_square is a boolean. The effective config must equal the
    // minimal form's.
    String fullDump =
        BASE.replace(
            "\"patch_size\": 16,",
            """
            "crop_size": null,
            "data_format": "channels_first",
            "default_to_square": true,
            "device": null,
            "disable_grouping": null,
            "do_center_crop": null,
            "do_pad": null,
            "input_data_format": null,
            "max_pixels": null,
            "min_pixels": null,
            "pad_size": null,
            "return_tensors": null,
            "patch_size": 16,\
            """);
    Qwen3VlProcessorConfig full = parse(fullDump);
    Qwen3VlProcessorConfig minimal = parse(BASE);
    assertEquals(minimal.patchSize(), full.patchSize());
    assertEquals(minimal.temporalPatchSize(), full.temporalPatchSize());
    assertEquals(minimal.mergeSize(), full.mergeSize());
    assertEquals(minimal.minPixels(), full.minPixels());
    assertEquals(minimal.maxPixels(), full.maxPixels());
    assertEquals(minimal.declaredMaxPixels(), full.declaredMaxPixels());
    assertEquals(minimal.rawRowBudget(), full.rawRowBudget());
  }

  @Test
  void fullDumpInertKeysMustStayNull() {
    for (String key :
        List.of(
            "crop_size",
            "device",
            "disable_grouping",
            "do_center_crop",
            "do_pad",
            "input_data_format",
            "pad_size",
            "return_tensors")) {
      expectRejection(
          BASE.replace("\"patch_size\": 16,", "\"" + key + "\": 1,\n    \"patch_size\": 16,"),
          "qwen3-vl preprocessor config full-dump key '"
              + key
              + "' must be null; a non-null value is an active processor setting this port"
              + " does not implement");
    }
    expectRejection(
        BASE.replace("\"patch_size\": 16,", "\"default_to_square\": 1,\n    \"patch_size\": 16,"),
        "qwen3-vl preprocessor config default_to_square must be a boolean");
    expectRejection(
        BASE.replace(
            "\"patch_size\": 16,", "\"data_format\": \"channels_last\",\n    \"patch_size\": 16,"),
        "qwen3-vl preprocessor config data_format must be null or 'channels_first'");
  }

  @Test
  void missingRequiredKeysAreRejected() {
    expectRejection(
        BASE.replace(SIZE_BLOCK, ""),
        "qwen3-vl preprocessor config is missing required key 'size'");
    expectRejection(
        BASE.replace(MEAN_BLOCK, ""),
        "qwen3-vl preprocessor config is missing required key 'image_mean'");
    expectRejection(
        BASE.replace("\"patch_size\": 16,\n", ""),
        "qwen3-vl preprocessor config is missing required key 'patch_size'");
  }

  @Test
  void malformedJsonIsRejected() {
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> parse("{ not json"));
    assertTrue(
        e.getMessage().startsWith("qwen3-vl preprocessor config is not valid JSON: "),
        "unexpected message: " + e.getMessage());
    expectRejection("[1, 2, 3]", "qwen3-vl preprocessor config root must be a JSON object");
    assertThrows(IllegalArgumentException.class, () -> Qwen3VlProcessorConfig.fromJson(null));
  }

  @Test
  void resampleNumbersMapToKernels() {
    assertEquals(Resampling.BICUBIC, parse(BASE).resample(), "absent resample defaults to BICUBIC");
    assertEquals(
        Resampling.BILINEAR,
        parse(BASE.replace("\"patch_size\": 16,", "\"resample\": 2,\n    \"patch_size\": 16,"))
            .resample());
    assertEquals(
        Resampling.BICUBIC,
        parse(BASE.replace("\"patch_size\": 16,", "\"resample\": 3,\n    \"patch_size\": 16,"))
            .resample());
    assertEquals(
        Resampling.LANCZOS,
        parse(BASE.replace("\"patch_size\": 16,", "\"resample\": 4,\n    \"patch_size\": 16,"))
            .resample());
  }

  @Test
  void resampleValuesAreStrict() {
    String insert = "\"resample\": %s,\n    \"patch_size\": 16,";
    for (int value : new int[] {1, 0, 5}) {
      expectRejection(
          BASE.replace("\"patch_size\": 16,", String.format(insert, value)),
          "unsupported resample value " + value + " (expected 2=BILINEAR, 3=BICUBIC or 4=LANCZOS)");
    }
    expectRejection(
        BASE.replace("\"patch_size\": 16,", String.format(insert, "2.5")),
        "resample must be an integer");
    expectRejection(
        BASE.replace("\"patch_size\": 16,", String.format(insert, "4294967297")),
        "resample must fit in a signed 32-bit integer: 4294967297");
  }

  @Test
  void sizeVariantsAreRejected() {
    expectRejection(BASE.replace(SIZE_BLOCK, "\"size\": 2048,"), "size must be a JSON object");
    expectRejection(
        BASE.replace(SIZE_BLOCK, "\"size\": {},"),
        "size must have exactly the keys 'shortest_edge' and 'longest_edge'");
    expectRejection(
        BASE.replace(
            SIZE_BLOCK,
            "\"size\": {\n"
                + "        \"longest_edge\": 16777216,\n"
                + "        \"shortest_edge\": 65536,\n"
                + "        \"medium_edge\": 1000\n"
                + "    },"),
        "size must have exactly the keys 'shortest_edge' and 'longest_edge'");
    expectRejection(
        BASE.replace("\"longest_edge\": 16777216", "\"longest_edge\": 2.5"),
        "size.longest_edge must be an integer");
    expectRejection(
        BASE.replace("\"shortest_edge\": 65536", "\"shortest_edge\": 0"),
        "size.shortest_edge must be positive: 0");
    expectRejection(
        BASE.replace("\"longest_edge\": 16777216", "\"longest_edge\": 4294967297"),
        "size.longest_edge must fit in a signed 32-bit integer: 4294967297");
  }

  @Test
  void integerOptionsAreStrict() {
    expectRejection(
        BASE.replace("\"patch_size\": 16,", "\"patch_size\": 0,"),
        "patch_size must be positive: 0");
    expectRejection(
        BASE.replace("\"patch_size\": 16,", "\"patch_size\": 16.5,"),
        "patch_size must be an integer");
    expectRejection(
        BASE.replace("\"patch_size\": 16,", "\"patch_size\": 4294967297,"),
        "patch_size must fit in a signed 32-bit integer: 4294967297");
    expectRejection(
        BASE.replace("\"temporal_patch_size\": 2,", "\"temporal_patch_size\": 0,"),
        "temporal_patch_size must be positive: 0");
    expectRejection(
        BASE.replace("\"merge_size\": 2,", "\"merge_size\": 0,"), "merge_size must be positive: 0");
  }

  @Test
  void oversizedPatchSizeThatOverflowsTheHardCapIsRejected() {
    // 4096 x 46341^2 > Integer.MAX_VALUE: the cap computation must stay in long and report it.
    expectRejection(
        BASE.replace("\"patch_size\": 16,", "\"patch_size\": 46341,"),
        "the default raw row budget (4096) times patch_size^2 (2147488281) overflows a signed "
            + "32-bit integer");
    // 4096 x (10^9)^2 also overflows a LONG: the check must catch it by division, where a
    // multiply-first form would wrap and slip past the range check.
    expectRejection(
        BASE.replace("\"patch_size\": 16,", "\"patch_size\": 1000000000,"),
        "the default raw row budget (4096) times patch_size^2 (1000000000000000000) overflows a "
            + "signed 32-bit integer");
  }

  @Test
  void builderOverflowChecksUseDivisionOrderToo() {
    // 3 x 10^9 x (10^9)^2 overflows a long: the builder's stride guard must reject it by the
    // division-ordered check, not by a wrapped product.
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Qwen3VlProcessorConfig.builder()
                    .patchSize(1000000000)
                    .temporalPatchSize(1000000000)
                    .mergeSize(1)
                    .minPixels(1)
                    .maxPixels(1)
                    .imageMean(new float[] {0.5f, 0.5f, 0.5f})
                    .imageStd(new float[] {0.5f, 0.5f, 0.5f})
                    .build());
    assertEquals(
        "3 * temporalPatchSize * patchSize * patchSize overflows a signed 32-bit integer: "
            + "3000000000 x 1000000000000000000",
        e.getMessage());
  }

  @Test
  void legacyPixelKeysOverrideTheSizeDict() {
    // Reference __init__ precedence: explicit min_pixels/max_pixels win over the size dict.
    Qwen3VlProcessorConfig min =
        parse(
            BASE.replace(
                "\"patch_size\": 16,", "\"min_pixels\": 100000,\n    \"patch_size\": 16,"));
    assertEquals(100000, min.minPixels());
    assertEquals(65536, parse(BASE).minPixels());

    Qwen3VlProcessorConfig max =
        parse(
            BASE.replace(
                "\"patch_size\": 16,", "\"max_pixels\": 500000,\n    \"patch_size\": 16,"));
    assertEquals(500000, max.declaredMaxPixels());
    assertEquals(500000, max.maxPixels(), "a declared max under the cap is kept");
  }

  @Test
  void declaredMaxIsCappedNotRejected() {
    // Decision (b) cases.
    assertEquals(1048576, parse(BASE).maxPixels());
    assertEquals(16777216, parse(BASE).declaredMaxPixels());

    Qwen3VlProcessorConfig underCap =
        parse(BASE.replace("\"longest_edge\": 16777216", "\"longest_edge\": 500000"));
    assertEquals(500000, underCap.maxPixels());
    assertEquals(500000, underCap.declaredMaxPixels());

    Qwen3VlProcessorConfig legacyOverCap =
        parse(
            BASE.replace(
                "\"patch_size\": 16,", "\"max_pixels\": 20000000,\n    \"patch_size\": 16,"));
    assertEquals(1048576, legacyOverCap.maxPixels());
    assertEquals(20000000, legacyOverCap.declaredMaxPixels());
  }

  @Test
  void minGreaterThanMaxIsRejected() {
    expectRejection(
        BASE.replace("\"longest_edge\": 16777216", "\"longest_edge\": 100000")
            .replace("\"shortest_edge\": 65536", "\"shortest_edge\": 200000"),
        "minPixels must be at most maxPixels: 200000 > 100000");
  }

  @Test
  void minExceedingTheCappedMaxNamesTheEffectiveCap() {
    // Declared 16,777,216 is capped at the raw-row hard cap 1,048,576 (decision (b)); the
    // rejection must name the effective cap and the declared value, not present the cap as
    // if it were the declared max_pixels.
    expectRejection(
        BASE.replace("\"shortest_edge\": 65536", "\"shortest_edge\": 2000000"),
        "min_pixels 2000000 exceeds the effective max_pixels 1048576 (declared 16777216, "
            + "capped at the raw-row hard cap 1048576)");
  }

  @Test
  void skippedPipelineStagesAreRejected() {
    String anchor = "{\n    \"size\": {";
    expectRejection(
        BASE.replace(anchor, "{\n    \"do_convert_rgb\": false,\n    \"size\": {"),
        "do_convert_rgb must be true: the ported decoder always converts decoded input to RGB, "
            + "so a config that skips the conversion would be silently ignored");
    expectRejection(
        BASE.replace(anchor, "{\n    \"do_rescale\": false,\n    \"size\": {"),
        "do_rescale must be true: the ported output contract is rescaled and normalized patches, "
            + "and a config that skips the rescale cannot be honored");
    expectRejection(
        BASE.replace(anchor, "{\n    \"do_normalize\": false,\n    \"size\": {"),
        "do_normalize must be true: the ported output contract is rescaled and normalized patches, "
            + "and a config that skips the normalize cannot be honored");
  }

  @Test
  void imageMeanAndStdVectorsAreStrict() {
    expectRejection(
        BASE.replace(
            MEAN_BLOCK, "    \"image_mean\": [\n" + "        0.5,\n" + "        0.5\n" + "    ],"),
        "image_mean must be an array of exactly three numbers");
    expectRejection(
        BASE.replace(
            MEAN_BLOCK,
            "    \"image_mean\": [\n"
                + "        \"0.5\",\n"
                + "        0.5,\n"
                + "        0.5\n"
                + "    ],"),
        "image_mean[0] must be a number");
    expectRejection(
        BASE.replace(
            STD_BLOCK,
            "    \"image_std\": [\n"
                + "        0.5,\n"
                + "        0.0,\n"
                + "        0.5\n"
                + "    ],"),
        "image_std values must be positive: 0.0");
    expectRejection(
        BASE.replace(
            STD_BLOCK,
            "    \"image_std\": [\n"
                + "        0.5,\n"
                + "        -1.0,\n"
                + "        0.5\n"
                + "    ],"),
        "image_std values must be positive: -1.0");
  }

  @Test
  void rescaleFactorIsStrict() {
    assertEquals(
        1.0 / 255.0, parse(BASE).rescaleFactor(), "absent rescale_factor defaults to 1/255");
    expectRejection(
        BASE.replace(anchor(), "\"rescale_factor\": 0.0,\n" + anchor()),
        "rescale_factor must be finite and positive: 0.0");
    expectRejection(
        BASE.replace(anchor(), "\"rescale_factor\": -2.0,\n" + anchor()),
        "rescale_factor must be finite and positive: -2.0");
    expectRejection(
        BASE.replace(anchor(), "\"rescale_factor\": \"a\",\n" + anchor()),
        "rescale_factor must be a JSON number");
  }

  private static String anchor() {
    return "\"patch_size\": 16,";
  }

  @Test
  void processorClassIsOptional() {
    assertNull(
        parse(BASE.replace("    \"processor_class\": \"Qwen3VLProcessor\",\n", ""))
            .processorClass());
    expectRejection(
        BASE.replace("\"processor_class\": \"Qwen3VLProcessor\"", "\"processor_class\": 123"),
        "processor_class must be a JSON string");
  }

  @Test
  void builderRequiresMeanAndStd() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            Qwen3VlProcessorConfig.builder()
                .patchSize(16)
                .temporalPatchSize(2)
                .mergeSize(2)
                .minPixels(65536)
                .maxPixels(1048576)
                .build(),
        "imageMean must be set");
    assertThrows(
        IllegalArgumentException.class,
        () ->
            Qwen3VlProcessorConfig.builder()
                .patchSize(16)
                .temporalPatchSize(2)
                .mergeSize(2)
                .minPixels(65536)
                .maxPixels(1048576)
                .imageMean(new float[] {0.5f, 0.5f, 0.5f})
                .build(),
        "imageStd must be set");
  }

  @Test
  void builderMaxPixelsIsRejectedAboveTheHardCap() {
    // 1,049,600 = the cap + half a raw row budget row (512 pixels of one 16x16 row).
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class, () -> baseBuilder().maxPixels(1049600).build());
    assertEquals(
        "maxPixels 1049600 exceeds the hard pixel cap of 1048576 (rawRowBudget 4096 x "
            + "patchSize^2 256)",
        e.getMessage());
    // The cap itself is accepted: for a built config the effective value is the declared value.
    Qwen3VlProcessorConfig atCap = baseBuilder().build();
    assertEquals(1048576, atCap.maxPixels());
    assertEquals(1048576, atCap.declaredMaxPixels());
    assertEquals(32, atCap.factor());
    assertEquals(1536, atCap.patchStride());
  }

  @Test
  void builderRawRowBudgetChangesTheHardCap() {
    // The budget-1024 cap (262144) is below the base builder's maxPixels, so a valid build at
    // that budget must lower maxPixels; the cap value is what is asserted here.
    assertEquals(
        262144, baseBuilder().rawRowBudget(1024).maxPixels(262144).build().hardCapPixels());
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> baseBuilder().rawRowBudget(1024).maxPixels(300000).build());
    assertEquals(
        "maxPixels 300000 exceeds the hard pixel cap of 262144 (rawRowBudget 1024 x "
            + "patchSize^2 256)",
        e.getMessage());
  }

  @Test
  void builderSharedGuardsApply() {
    assertThrows(IllegalArgumentException.class, () -> baseBuilder().resample(null).build());
    assertThrows(IllegalArgumentException.class, () -> baseBuilder().rescaleFactor(0.0).build());
    assertThrows(IllegalArgumentException.class, () -> baseBuilder().patchSize(0).build());
    assertThrows(IllegalArgumentException.class, () -> baseBuilder().mergeSize(0).build());
    assertThrows(IllegalArgumentException.class, () -> baseBuilder().minPixels(0).build());
    assertThrows(IllegalArgumentException.class, () -> baseBuilder().maxPixels(0).build());
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> baseBuilder().minPixels(900000).maxPixels(100000).build());
    assertEquals("minPixels must be at most maxPixels: 900000 > 100000", e.getMessage());
  }
}
