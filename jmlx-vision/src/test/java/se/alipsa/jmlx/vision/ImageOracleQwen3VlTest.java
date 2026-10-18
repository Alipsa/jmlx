package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

/**
 * Byte-exact comparison of the Qwen3-VL preprocessing chain against the committed {@code qwen3vl}
 * image-oracle family. The fixtures pin the pinned transformers 4.57.6 slow {@code
 * Qwen2VLImageProcessor} file-to-processor path: decode with drop-alpha + EXIF-apply, {@code
 * smart_resize}, Pillow-exact BICUBIC resize, rescale/normalize, and the merge-block patch
 * permutation (see findings §4.4–§4.6).
 */
class ImageOracleQwen3VlTest {

  /** Cases whose processed patch bin is committed under {@code qwen3vl/patches/}. */
  private static final Set<String> PATCH_CASES = Set.of("model-rgb-256x256", "model-rgb-64x32");

  /** Cases whose {@code processFile} entry point must be exercised for parity. */
  private static final Set<String> PROCESS_FILE_CASES =
      Set.of("model-rgb-256x256", "rotated-orient6-64x32");

  @Test
  void qwen3vlOracleFamilyMatchesForEveryCase() throws Exception {
    JsonNode inputCases = OracleFixtures.read("qwen3vl.input.json").required("cases");
    JsonNode expectedCases = OracleFixtures.read("qwen3vl.expected.json").required("cases");
    assertEquals(expectedCases.size(), inputCases.size());
    assertTrue(inputCases.size() > 0);
    for (int i = 0; i < inputCases.size(); i++) {
      assertEquals(
          expectedCases.get(i).required("name").asString(),
          inputCases.get(i).required("name").asString(),
          "case order at " + i);
    }

    byte[] configBytes =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("qwen3vl/preprocessor_config.json"));
    Qwen3VlProcessorConfig config = Qwen3VlProcessorConfig.fromJson(configBytes);
    assertEquals(65536, config.minPixels());
    assertEquals(1048576, config.maxPixels());
    assertTrue(config.doResize());
    assertEquals(16, config.patchSize());
    assertEquals(2, config.mergeSize());
    assertEquals(4096, config.rawRowBudget());
    assertEquals(Resampling.BICUBIC, config.resample());

    // The no-resize processor mirrors every other effective option of the parsed config, with
    // do_resize switched off (the builder has no toBuilder, so the values come from the getters).
    Qwen3VlProcessorConfig noResizeConfig =
        Qwen3VlProcessorConfig.builder()
            .patchSize(config.patchSize())
            .temporalPatchSize(config.temporalPatchSize())
            .mergeSize(config.mergeSize())
            .minPixels(config.minPixels())
            .maxPixels(config.maxPixels())
            .rawRowBudget(config.rawRowBudget())
            .doResize(false)
            .resample(config.resample())
            .rescaleFactor(config.rescaleFactor())
            .imageMean(config.imageMean())
            .imageStd(config.imageStd())
            .processorClass(config.processorClass())
            .build();
    Qwen3VlImageProcessor processor = new Qwen3VlImageProcessor(config);
    Qwen3VlImageProcessor noResizeProcessor = new Qwen3VlImageProcessor(noResizeConfig);

    for (int i = 0; i < inputCases.size(); i++) {
      JsonNode testCase = inputCases.get(i);
      String name = testCase.required("name").asString();
      JsonNode expectedCase = OracleFixtures.findCase(expectedCases, name);
      if (expectedCase.has("expectedRejected")) {
        assertRejection(name, testCase, expectedCase, processor, noResizeProcessor);
      } else {
        assertProcessedCase(name, testCase, expectedCase, processor, noResizeProcessor);
      }
    }
  }

  /** The decode must succeed; the processor must reject with the exact geometry message. */
  private static void assertRejection(
      String name,
      JsonNode testCase,
      JsonNode expectedCase,
      Qwen3VlImageProcessor processor,
      Qwen3VlImageProcessor noResizeProcessor)
      throws Exception {
    assertTrue(expectedCase.required("expectedRejected").asBoolean(), name);
    Path fixture = OracleFixtures.fixtures().resolve(testCase.required("image").asString());
    RgbImage decoded =
        ImageDecoder.decode(
            fixture,
            ImageDecodeLimits.DEFAULTS,
            AlphaPolicy.DROP_ALPHA,
            ExifOrientationPolicy.APPLY);
    int[] source = OracleFixtures.ints(expectedCase.required("source"));
    assertEquals(source[1], decoded.height(), name + " stored height");
    assertEquals(source[0], decoded.width(), name + " stored width");

    String kind = expectedCase.required("rejectKind").asString();
    if ("aspect".equals(kind)) {
      IllegalArgumentException failure =
          assertThrows(IllegalArgumentException.class, () -> processor.process(decoded), name);
      assertEquals(
          "absolute aspect ratio must be smaller than 200, got 200.2", failure.getMessage(), name);
    } else {
      assertEquals("raw-row-budget", kind, name);
      assertEquals(4100, expectedCase.required("rawPatchRows").intValue(), name);
      assertEquals(4096, expectedCase.required("rawRowBudget").intValue(), name);
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class, () -> noResizeProcessor.process(decoded), name);
      assertEquals(
          "raw patch rows 4100 exceed the raw patch row budget of 4096 for 160x6560",
          failure.getMessage(),
          name);
    }
  }

  private static void assertProcessedCase(
      String name,
      JsonNode testCase,
      JsonNode expectedCase,
      Qwen3VlImageProcessor processor,
      Qwen3VlImageProcessor noResizeProcessor)
      throws Exception {
    boolean noResize = testCase.has("doResize") && !testCase.required("doResize").asBoolean();
    Qwen3VlImageProcessor chosen = noResize ? noResizeProcessor : processor;

    List<String> imagePaths = new ArrayList<>();
    if (testCase.has("images")) {
      for (JsonNode path : testCase.required("images")) {
        imagePaths.add(path.asString());
      }
    } else {
      imagePaths.add(testCase.required("image").asString());
    }
    JsonNode expectedImages = expectedCase.required("images");
    assertEquals(expectedImages.size(), imagePaths.size(), name);
    for (int i = 0; i < imagePaths.size(); i++) {
      assertProcessedImage(
          name, name + " image " + i, imagePaths.get(i), expectedImages.get(i), chosen);
    }
  }

  private static void assertProcessedImage(
      String caseName,
      String label,
      String imagePath,
      JsonNode record,
      Qwen3VlImageProcessor processor)
      throws Exception {
    Path fixture = OracleFixtures.fixtures().resolve(imagePath);
    RgbImage decoded =
        ImageDecoder.decode(
            fixture,
            ImageDecodeLimits.DEFAULTS,
            AlphaPolicy.DROP_ALPHA,
            ExifOrientationPolicy.APPLY);
    Qwen3VlImageProcessorResult result = processor.process(decoded);

    assertArrayEquals(OracleFixtures.ints(record.required("gridThw")), result.gridThw(), label);
    assertEquals(record.required("rawPatchRows").intValue(), result.rawPatchRows(), label);
    assertEquals(record.required("mergedRowCount").intValue(), result.mergedRowCount(), label);
    assertEquals(1536, result.patchStride(), label);

    float[] values = result.values();
    int rawRows = record.required("rawPatchRows").intValue();
    assertEquals(rawRows * 1536, values.length, label);
    // The reference's pixel_values serialized as C-order little-endian IEEE-754 float32.
    byte[] littleEndian = littleEndianFloat32(values);
    assertEquals(
        record.required("pixelSha256").asString(), OracleFixtures.sha256Hex(littleEndian), label);

    JsonNode samples = record.required("samples");
    for (JsonNode sample : samples) {
      int row = sample.required("row").intValue();
      float[] expectedRow =
          OracleFixtures.float32(OracleFixtures.base64(sample.required("valuesBase64").asString()));
      assertEquals(1536, expectedRow.length, label + " sample row " + row);
      for (int j = 0; j < expectedRow.length; j++) {
        // The oracle's big-endian float32 bits must be bit-identical, not merely close.
        assertEquals(
            Float.floatToIntBits(expectedRow[j]),
            Float.floatToIntBits(values[row * 1536 + j]),
            label + " sample row " + row + " column " + j);
      }
    }

    // The uint8 pass: resizing the decoded (EXIF-applied) image to the recorded target must
    // reproduce the reference's resize output byte-for-byte. A same-size resize is an exact
    // copy, so this also covers the no-resize and already-at-target cases.
    int[] resized = OracleFixtures.ints(record.required("resized"));
    RgbImage resizedImage =
        ImageTransforms.resize(decoded, resized[0], resized[1], Resampling.BICUBIC);
    assertEquals(
        record.required("uint8Sha256").asString(),
        OracleFixtures.sha256Hex(resizedImage.pixels()),
        label);

    if (record.has("whiteComposite")) {
      // Pins the module-default decode (COMPOSITE_OVER_WHITE + IGNORE) for these images: the
      // unchanged SmolVLM/Idefics3 path.
      JsonNode composite = record.required("whiteComposite");
      RgbImage defaultDecode = ImageDecoder.decode(fixture);
      assertEquals(composite.required("width").intValue(), defaultDecode.width(), label);
      assertEquals(composite.required("height").intValue(), defaultDecode.height(), label);
      assertArrayEquals(
          OracleFixtures.base64(composite.required("pixelsBase64").asString()),
          defaultDecode.pixels(),
          label + " white composite");
    }
    if (record.has("storedDecode")) {
      JsonNode storedRecord = record.required("storedDecode");
      RgbImage stored =
          ImageDecoder.decode(
              fixture,
              ImageDecodeLimits.DEFAULTS,
              AlphaPolicy.DROP_ALPHA,
              ExifOrientationPolicy.IGNORE);
      assertEquals(storedRecord.required("width").intValue(), stored.width(), label);
      assertEquals(storedRecord.required("height").intValue(), stored.height(), label);
      assertArrayEquals(
          OracleFixtures.base64(storedRecord.required("pixelsBase64").asString()),
          stored.pixels(),
          label + " stored decode");
      // Non-matching control: the EXIF-ignoring pipeline must NOT reproduce the record's pixels;
      // an exif_transpose-free port would match them exactly.
      Qwen3VlImageProcessorResult untransposed = processor.process(stored);
      assertNotEquals(
          record.required("pixelSha256").asString(),
          OracleFixtures.sha256Hex(littleEndianFloat32(untransposed.values())),
          label + " EXIF-ignoring pipeline must not match");
    }

    if (PATCH_CASES.contains(caseName)) {
      byte[] patchBytes =
          Files.readAllBytes(
              OracleFixtures.fixtures()
                  .resolve("qwen3vl/patches/" + caseName + ".pixel-values.f32"));
      assertTrue(Arrays.equals(patchBytes, littleEndian), label + " shared patch bin");
    }
    if (PROCESS_FILE_CASES.contains(caseName)) {
      Qwen3VlImageProcessorResult fromFile =
          processor.processFile(fixture, ImageDecodeLimits.DEFAULTS);
      assertTrue(
          Arrays.equals(littleEndianFloat32(fromFile.values()), littleEndian),
          label + " processFile(Path) parity");
      Qwen3VlImageProcessorResult fromBytes =
          processor.processFile(Files.readAllBytes(fixture), ImageDecodeLimits.DEFAULTS);
      assertTrue(
          Arrays.equals(littleEndianFloat32(fromBytes.values()), littleEndian),
          label + " processFile(byte[]) parity");
    }
  }

  /** The little-endian IEEE-754 float32 encoding of {@code values} (the reference layout). */
  private static byte[] littleEndianFloat32(float[] values) {
    ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    buffer.asFloatBuffer().put(values);
    return buffer.array();
  }
}
