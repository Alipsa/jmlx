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

class Phi3VProcessorConfigTest {

  /**
   * The pinned Phi-3.5-vision-instruct {@code preprocessor_config.json}, byte-identical to the
   * artifact file (2-space indent, trailing newline); the SHA-256 of the no-trailing-newline form
   * is the one recorded here and in {@code tools/image-oracle/provenance.json} fixtureSources.
   */
  private static final String BASE =
      """
      {
        "auto_map": {
          "AutoImageProcessor": "processing_phi3_v.Phi3VImageProcessor",
          "AutoProcessor": "processing_phi3_v.Phi3VProcessor"
        },
        "do_convert_rgb": true,
        "image_mean": [
          0.48145466,
          0.4578275,
          0.40821073
        ],
        "image_processor_type": "Phi3VImageProcessor",
        "image_std": [
          0.26862954,
          0.26130258,
          0.27577711
        ],
        "num_crops": 4,
        "num_img_tokens": 144,
        "processor_class": "Phi3VProcessor"
      }
      """;

  /** The exact two-space-indented block of {@link #BASE} carrying {@code image_mean}. */
  private static final String MEAN_BLOCK =
      "  \"image_mean\": [\n"
          + "    0.48145466,\n"
          + "    0.4578275,\n"
          + "    0.40821073\n"
          + "  ],\n";

  /** The exact two-space-indented block of {@link #BASE} carrying {@code image_std}. */
  private static final String STD_BLOCK =
      "  \"image_std\": [\n"
          + "    0.26862954,\n"
          + "    0.26130258,\n"
          + "    0.27577711\n"
          + "  ],\n";

  private static Phi3VProcessorConfig parse(String json) {
    return Phi3VProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8));
  }

  private static IllegalArgumentException expectRejection(String json, String message) {
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(json));
    assertEquals(message, e.getMessage());
    return e;
  }

  private static String sha256(String text) {
    try {
      byte[] digest =
          MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
      StringBuilder sb = new StringBuilder(digest.length * 2);
      for (byte b : digest) {
        sb.append(String.format("%02x", b & 0xFF));
      }
      return sb.toString();
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static Phi3VProcessorConfig.Builder baseBuilder() {
    return Phi3VProcessorConfig.builder()
        .numCrops(4)
        .imageMean(new float[] {0.48145466f, 0.4578275f, 0.40821073f})
        .imageStd(new float[] {0.26862954f, 0.26130258f, 0.27577711f});
  }

  @Test
  void pinnedJsonIsByteIdenticalToTheTierBArtifact() {
    // The text block carries a trailing newline the 442-byte artifact file also carries, so the
    // block minus its single trailing newline is the exact artifact content.
    assertEquals(
        "5b7aedbc52905162c9a7e7b0b9cadcf620a6b6a8374a1015a3453efe4ffcc156",
        sha256(BASE.stripTrailing()));
  }

  @Test
  void pinnedConfigParsesToThePhi3VValues() {
    Phi3VProcessorConfig config = parse(BASE);
    assertEquals(4, config.numCrops());
    assertArrayEquals(new float[] {0.48145466f, 0.4578275f, 0.40821073f}, config.imageMean());
    assertArrayEquals(new float[] {0.26862954f, 0.26130258f, 0.27577711f}, config.imageStd());
    assertEquals("Phi3VProcessor", config.processorClass());
  }

  @Test
  void meanAndStdAccessorsReturnDefensiveCopies() {
    Phi3VProcessorConfig config = parse(BASE);
    float[] mean = config.imageMean();
    float[] std = config.imageStd();
    mean[0] = 9f;
    std[0] = 9f;
    assertArrayEquals(new float[] {0.48145466f, 0.4578275f, 0.40821073f}, config.imageMean());
    assertArrayEquals(new float[] {0.26862954f, 0.26130258f, 0.27577711f}, config.imageStd());
  }

  @Test
  void unknownKeysAreRejected() {
    expectRejection(
        BASE.replace("\"num_crops\": 4,", "\"num_crops\": 4,\n    \"bogus_option\": 1,"),
        "phi3v preprocessor config has unrecognized key 'bogus_option'");
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
          BASE.replace("\"num_crops\": 4,", "\"" + key + "\": 1,\n    \"num_crops\": 4,"),
          "phi3v preprocessor config full-dump key '"
              + key
              + "' must be null; a non-null value is an active processor setting this port"
              + " does not implement");
    }
    // The null form of the full dump parses to the same effective config as the minimal form.
    String fullDump =
        BASE.replace(
            "\"num_crops\": 4,",
            "\"crop_size\": null,\n    \"device\": null,\n    \"do_center_crop\": null,\n"
                + "    \"do_pad\": null,\n    \"num_crops\": 4,");
    assertEquals(4, parse(fullDump).numCrops());
  }

  @Test
  void wrongProcessorTypeIsRejected() {
    expectRejection(
        BASE.replace(
            "\"image_processor_type\": \"Phi3VImageProcessor\"",
            "\"image_processor_type\": \"CLIPImageProcessor\""),
        "phi3v preprocessor config image_processor_type must be 'Phi3VImageProcessor'");
  }

  @Test
  void missingRequiredKeysAreRejected() {
    expectRejection(
        BASE.replace(MEAN_BLOCK, ""),
        "phi3v preprocessor config is missing required key 'image_mean'");
    expectRejection(
        BASE.replace(STD_BLOCK, ""),
        "phi3v preprocessor config is missing required key 'image_std'");
    expectRejection(
        BASE.replace("  \"num_crops\": 4,\n", ""),
        "phi3v preprocessor config is missing required key 'num_crops'");
  }

  @Test
  void numCropsIsStrict() {
    expectRejection(
        BASE.replace("\"num_crops\": 4,", "\"num_crops\": 0,"), "num_crops must be positive: 0");
    expectRejection(
        BASE.replace("\"num_crops\": 4,", "\"num_crops\": -2,"), "num_crops must be positive: -2");
    expectRejection(
        BASE.replace("\"num_crops\": 4,", "\"num_crops\": 4.5,"), "num_crops must be an integer");
    expectRejection(
        BASE.replace("\"num_crops\": 4,", "\"num_crops\": \"4\","), "num_crops must be an integer");
    expectRejection(
        BASE.replace("\"num_crops\": 4,", "\"num_crops\": 4294967297,"),
        "num_crops must fit in a signed 32-bit integer: 4294967297");
  }

  @Test
  void legacyNumImgTokensIsAcceptedAndIgnored() {
    // The reference processor never reads num_img_tokens from the image-processor config.
    assertEquals(4, parse(BASE).numCrops());
    // null is accepted (the full-dump form); a positive int is accepted too.
    assertEquals(
        4, parse(BASE.replace("\"num_img_tokens\": 144,", "\"num_img_tokens\": null,")).numCrops());
    assertEquals(
        4, parse(BASE.replace("\"num_img_tokens\": 144,", "\"num_img_tokens\": 757,")).numCrops());
    expectRejection(
        BASE.replace("\"num_img_tokens\": 144,", "\"num_img_tokens\": 0,"),
        "num_img_tokens must be positive: 0");
  }

  @Test
  void autoMapMustBeAnObject() {
    String autoMapBlock =
        """
        "auto_map": {
            "AutoImageProcessor": "processing_phi3_v.Phi3VImageProcessor",
            "AutoProcessor": "processing_phi3_v.Phi3VProcessor"
          },
        """;
    expectRejection(
        BASE.replace(autoMapBlock, "\"auto_map\": \"processing_phi3_v\","),
        "auto_map must be a JSON object");
  }

  @Test
  void doConvertRgbFalseIsRejected() {
    expectRejection(
        BASE.replace("\"do_convert_rgb\": true,", "\"do_convert_rgb\": false,"),
        "do_convert_rgb must be true: the ported decoder always converts decoded input to RGB, "
            + "so a config that skips the conversion would be silently ignored");
  }

  @Test
  void imageMeanAndStdVectorsAreStrict() {
    expectRejection(
        BASE.replace(
            MEAN_BLOCK, "    \"image_mean\": [\n        0.48145466,\n        0.4578275\n    ],"),
        "image_mean must be an array of exactly three numbers");
    expectRejection(
        BASE.replace(
            MEAN_BLOCK,
            "    \"image_mean\": [\n        \"0.48145466\",\n        0.4578275,\n"
                + "        0.40821073\n    ],"),
        "image_mean[0] must be a number");
    expectRejection(
        BASE.replace(STD_BLOCK, "    \"image_std\": [0.26862954, 0.0, 0.27577711],"),
        "imageStd values must be finite and positive: 0.0");
    expectRejection(
        BASE.replace(STD_BLOCK, "    \"image_std\": [0.26862954, -1.0, 0.27577711],"),
        "imageStd values must be finite and positive: -1.0");
  }

  @Test
  void malformedJsonIsRejected() {
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> parse("{ not json"));
    assertTrue(
        e.getMessage().startsWith("phi3v preprocessor config is not valid JSON: "),
        "unexpected message: " + e.getMessage());
    expectRejection("[1, 2, 3]", "phi3v preprocessor config root must be a JSON object");
    assertThrows(IllegalArgumentException.class, () -> Phi3VProcessorConfig.fromJson(null));
  }

  @Test
  void processorClassIsOptional() {
    assertNull(
        parse(BASE.replace(",\n  \"processor_class\": \"Phi3VProcessor\"\n", "\n"))
            .processorClass());
    expectRejection(
        BASE.replace("\"processor_class\": \"Phi3VProcessor\"", "\"processor_class\": 123"),
        "processor_class must be a JSON string");
  }

  @Test
  void builderSharedGuardsApply() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            Phi3VProcessorConfig.builder()
                .imageMean(new float[] {0.5f, 0.5f, 0.5f})
                .imageStd(new float[] {0.5f, 0.5f, 0.5f})
                .build(),
        "numCrops must be positive: 0");
    assertThrows(
        IllegalArgumentException.class,
        () ->
            Phi3VProcessorConfig.builder()
                .numCrops(4)
                .imageStd(new float[] {0.5f, 0.5f, 0.5f})
                .build(),
        "imageMean must be set");
    assertThrows(
        IllegalArgumentException.class,
        () ->
            Phi3VProcessorConfig.builder()
                .numCrops(4)
                .imageMean(new float[] {0.5f, 0.5f, 0.5f})
                .build(),
        "imageStd must be set");
    assertThrows(
        IllegalArgumentException.class,
        () -> baseBuilder().numCrops(0).build(),
        "numCrops must be positive: 0");
    assertThrows(
        IllegalArgumentException.class,
        () -> baseBuilder().imageStd(new float[] {0.5f, 0.0f, 0.5f}).build(),
        "imageStd values must be finite and positive: 0.0");
    assertThrows(
        IllegalArgumentException.class,
        () -> baseBuilder().imageMean(new float[2]).build(),
        "imageMean must be an array of exactly three values");
    assertEquals(
        "Phi3VProcessor", baseBuilder().processorClass("Phi3VProcessor").build().processorClass());
    assertEquals(4, baseBuilder().build().numCrops());
  }
}
