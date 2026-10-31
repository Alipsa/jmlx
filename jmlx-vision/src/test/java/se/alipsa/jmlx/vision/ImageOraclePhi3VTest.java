package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
 * Byte-exact comparison of the Phi-3.5-Vision preprocessing chain against the committed {@code
 * phi3v} image-oracle family. The fixtures pin the pinned bundled trust-remote-code {@code
 * Phi3VImageProcessor} of the {@code microsoft/Phi-3.5-vision-instruct} artifact (findings §6):
 * drop-alpha + EXIF-ignore decode, the HD transpose/resize/pad transform, the ToTensor/Normalize
 * float32 arithmetic, the torch bicubic global image, and the h-major 336x336 crop slots with zero
 * padding to {@code num_crops + 1}.
 */
class ImageOraclePhi3VTest {

  /** The single-image cases whose full batch row is committed under {@code phi3v/patches/}. */
  private static final Set<String> PATCH_CASES =
      Set.of("square-336x336", "wide-672x336", "tall-336x672", "trunc-300x301");

  /** The case whose {@code processFile}/{@code processEncoded} entry points get parity checks. */
  private static final Set<String> PROCESS_FILE_CASES = Set.of("trunc-300x301");

  @Test
  void phi3vOracleFamilyMatchesForEveryCase() throws Exception {
    JsonNode inputCases = OracleFixtures.read("phi3v.input.json").required("cases");
    JsonNode expectedCases = OracleFixtures.read("phi3v.expected.json").required("cases");
    assertEquals(expectedCases.size(), inputCases.size());
    assertTrue(inputCases.size() > 0);
    for (int i = 0; i < inputCases.size(); i++) {
      assertEquals(
          expectedCases.get(i).required("name").asString(),
          inputCases.get(i).required("name").asString(),
          "case order at " + i);
    }

    // The parsed config must agree with the oracle's recorded construction options.
    JsonNode config = OracleFixtures.read("phi3v.expected.json").required("config");
    byte[] configBytes =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("phi3v/preprocessor_config.json"));
    Phi3VProcessorConfig parsed = Phi3VProcessorConfig.fromJson(configBytes);
    assertEquals("Phi3VImageProcessor", config.required("class").asString());
    assertEquals(4, config.required("numCrops").intValue());
    assertEquals(4, parsed.numCrops());
    assertEquals("Phi3VProcessor", parsed.processorClass());
    for (int c = 0; c < 3; c++) {
      assertEquals(
          Float.floatToIntBits((float) config.required("imageMean").get(c).asDouble()),
          Float.floatToIntBits(parsed.imageMean()[c]),
          "imageMean channel " + c);
      assertEquals(
          Float.floatToIntBits((float) config.required("imageStd").get(c).asDouble()),
          Float.floatToIntBits(parsed.imageStd()[c]),
          "imageStd channel " + c);
    }
    Phi3VImageProcessor processor = new Phi3VImageProcessor(parsed);

    for (int i = 0; i < inputCases.size(); i++) {
      JsonNode testCase = inputCases.get(i);
      String name = testCase.required("name").asString();
      JsonNode expectedCase = OracleFixtures.findCase(expectedCases, name);

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
      for (int j = 0; j < imagePaths.size(); j++) {
        assertProcessedImage(
            name, name + " image " + j, imagePaths.get(j), expectedImages.get(j), processor);
      }
    }
  }

  private static void assertProcessedImage(
      String caseName,
      String label,
      String imagePath,
      JsonNode record,
      Phi3VImageProcessor processor)
      throws Exception {
    Path fixture = OracleFixtures.fixtures().resolve(imagePath);
    // The Phi-3.5-Vision file policy: the reference convert("RGB") drops alpha without
    // compositing and the processor never calls exif_transpose.
    RgbImage decoded =
        ImageDecoder.decode(
            fixture,
            ImageDecodeLimits.DEFAULTS,
            AlphaPolicy.DROP_ALPHA,
            ExifOrientationPolicy.IGNORE);
    int[] source = OracleFixtures.ints(record.required("source"));
    assertEquals(source[0], decoded.width(), label + " stored width");
    assertEquals(source[1], decoded.height(), label + " stored height");

    Phi3VImageProcessorResult result = processor.process(decoded);
    Phi3VGeometryPlan plan = result.plan();
    assertEquals(record.required("gridH").intValue(), plan.gridH(), label + " gridH");
    assertEquals(record.required("gridW").intValue(), plan.gridW(), label + " gridW");
    assertEquals(
        record.required("cropBlocks").intValue(), plan.cropBlocks(), label + " cropBlocks");
    assertEquals(record.required("slotCount").intValue(), plan.slotCount(), label + " slotCount");
    assertEquals(
        record.required("numImgTokens").intValue(), plan.numImgTokens(), label + " numImgTokens");
    assertArrayEquals(
        OracleFixtures.ints(record.required("imageSize")),
        plan.imageSize(),
        label + " image_sizes [h, w]");

    float[] values = result.values();
    assertEquals((long) plan.slotCount() * 3 * 336 * 336, values.length, label);
    // The reference's batch row serialized as C-order little-endian IEEE-754 float32.
    byte[] littleEndian = littleEndianFloat32(values);
    assertEquals(
        record.required("pixelSha256").asString(), OracleFixtures.sha256Hex(littleEndian), label);

    JsonNode samples = record.required("samples");
    for (JsonNode sample : samples) {
      int row = sample.required("row").intValue();
      float[] expectedRow =
          OracleFixtures.float32(OracleFixtures.base64(sample.required("valuesBase64").asString()));
      assertEquals(3 * 336 * 336, expectedRow.length, label + " sample slot " + row);
      for (int j = 0; j < expectedRow.length; j += 97) {
        // The oracle's big-endian float32 bits must be bit-identical, not merely close.
        assertEquals(
            Float.floatToIntBits(expectedRow[j]),
            Float.floatToIntBits(values[row * 3 * 336 * 336 + j]),
            label + " sample slot " + row + " element " + j);
      }
    }

    if (PATCH_CASES.contains(caseName)) {
      byte[] patchBytes =
          Files.readAllBytes(
              OracleFixtures.fixtures().resolve("phi3v/patches/" + caseName + ".pixel-values.f32"));
      assertTrue(Arrays.equals(patchBytes, littleEndian), label + " shared patch bin");
    }
    if (PROCESS_FILE_CASES.contains(caseName)) {
      Phi3VImageProcessorResult fromFile =
          processor.processFile(fixture, ImageDecodeLimits.DEFAULTS);
      assertTrue(
          Arrays.equals(littleEndianFloat32(fromFile.values()), littleEndian),
          label + " processFile(Path) parity");
      Phi3VImageProcessorResult fromBytes =
          processor.processEncoded(Files.readAllBytes(fixture), ImageDecodeLimits.DEFAULTS);
      assertTrue(
          Arrays.equals(littleEndianFloat32(fromBytes.values()), littleEndian),
          label + " processEncoded(byte[]) parity");
    }
  }

  /** The little-endian IEEE-754 float32 encoding of {@code values} (the reference layout). */
  private static byte[] littleEndianFloat32(float[] values) {
    ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    buffer.asFloatBuffer().put(values);
    return buffer.array();
  }
}
