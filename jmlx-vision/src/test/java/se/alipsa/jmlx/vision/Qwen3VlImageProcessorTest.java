package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * Pixel-level tests for {@link Qwen3VlImageProcessor}. The two golden SHA-256 tests compare the
 * ported chain (resize + rescale + normalize + merge-block permutation + temporal duplication)
 * against the pinned transformers 4.57.6 slow-processor {@code _preprocess} output for the WP1
 * probe images, byte-for-byte over the little-endian float32 buffer.
 */
class Qwen3VlImageProcessorTest {

  /** The pinned Qwen3-VL-2B-Instruct {@code preprocessor_config.json} (see the config test). */
  private static final String PINNED_JSON =
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

  private static final float[] MEAN = {0.5f, 0.5f, 0.5f};
  private static final float[] STD = {0.5f, 0.5f, 0.5f};
  private static final double SCALE = 1.0 / 255.0;

  // WP1 probe goldens: SHA-256 of the reference slow-processor pixel_values as little-endian
  // float32 bytes (see the venv verification recorded in the WP2 plan and findings §4.5).
  private static final String SHA_256X256 =
      "df316e1ea42d36286d68368e17cbbcb74dfb0bb332839059f48be6ac93f5b07a";
  private static final String SHA_64X32 =
      "1b98a584a002f17b3a62eac192dd9a1a473d52a60c4571a3acc10a40ffa72754";

  private static Qwen3VlImageProcessor pinnedProcessor() {
    return new Qwen3VlImageProcessor(
        Qwen3VlProcessorConfig.fromJson(PINNED_JSON.getBytes(StandardCharsets.UTF_8)));
  }

  private static Qwen3VlImageProcessor noResizeProcessor() {
    return new Qwen3VlImageProcessor(
        Qwen3VlProcessorConfig.builder()
            .patchSize(16)
            .temporalPatchSize(2)
            .mergeSize(2)
            .minPixels(65536)
            .maxPixels(1048576)
            .doResize(false)
            .imageMean(MEAN)
            .imageStd(STD)
            .build());
  }

  /** WP1 probe image: r=x, g=y, b=(x+y)/2. */
  private static RgbImage probeImage256x256() {
    byte[] pixels = new byte[256 * 256 * 3];
    int i = 0;
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        pixels[i++] = (byte) x;
        pixels[i++] = (byte) y;
        pixels[i++] = (byte) ((x + y) / 2);
      }
    }
    return new RgbImage(256, 256, pixels);
  }

  /** WP1 probe image: r=(x*255)/63, g=(y*255)/31, b=(x*3+y*5)%256. */
  private static RgbImage probeImage64x32() {
    byte[] pixels = new byte[64 * 32 * 3];
    int i = 0;
    for (int y = 0; y < 32; y++) {
      for (int x = 0; x < 64; x++) {
        pixels[i++] = (byte) ((x * 255) / 63);
        pixels[i++] = (byte) ((y * 255) / 31);
        pixels[i++] = (byte) ((x * 3 + y * 5) % 256);
      }
    }
    return new RgbImage(64, 32, pixels);
  }

  /** Index-coded pixels for the no-resize exact-value tests. */
  private static int indexPixel(int x, int y, int channel) {
    return switch (channel) {
      case 0 -> (x * 7 + y * 3) % 256;
      case 1 -> (x + y * 11) % 256;
      default -> (x * 5 + y) % 256;
    };
  }

  private static RgbImage indexImage(int width, int height) {
    byte[] pixels = new byte[width * height * 3];
    int i = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pixels[i++] = (byte) indexPixel(x, y, 0);
        pixels[i++] = (byte) indexPixel(x, y, 1);
        pixels[i++] = (byte) indexPixel(x, y, 2);
      }
    }
    return new RgbImage(width, height, pixels);
  }

  private static String hex(byte[] bytes) {
    StringBuilder sb = new StringBuilder();
    for (byte b : bytes) {
      sb.append(String.format("%02x", b));
    }
    return sb.toString();
  }

  /** SHA-256 over the little-endian float32 encoding of {@code values} (the reference layout). */
  private static String sha256LeFloat32(float[] values) {
    try {
      ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
      buffer.asFloatBuffer().put(values);
      return hex(MessageDigest.getInstance("SHA-256").digest(buffer.array()));
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static void assertGoldenSha(float[] values, String expectedSha, float[] expectedFirst16) {
    float[] first16 = Arrays.copyOf(values, 16);
    assertEquals(
        expectedSha,
        sha256LeFloat32(values),
        "SHA-256 mismatch: expected first16 "
            + Arrays.toString(expectedFirst16)
            + " but got "
            + Arrays.toString(first16));
    assertArrayEquals(expectedFirst16, first16);
  }

  @Test
  void constructorAndProcessRejectNullArguments() {
    assertThrows(IllegalArgumentException.class, () -> new Qwen3VlImageProcessor(null));
    assertThrows(IllegalArgumentException.class, () -> pinnedProcessor().process(null));
    assertThrows(IllegalArgumentException.class, () -> noResizeProcessor().process(null));
  }

  @Test
  void probeImage256x256MatchesTheReferenceGolden() {
    Qwen3VlImageProcessor processor = pinnedProcessor();
    RgbImage image = probeImage256x256();
    Qwen3VlImageProcessorResult result = processor.process(image);

    // Same-size resize (an exact copy in the reference and the ported kernel): the SHA therefore
    // discriminates the rescale/normalize arithmetic and the merge-block permutation.
    final Qwen3VlGeometryPlan plan = processor.geometry(256, 256);
    assertArrayEquals(new int[] {1, 16, 16}, result.gridThw());
    assertEquals(256, result.rawPatchRows());
    assertEquals(64, result.mergedRowCount());
    assertEquals(1536, result.patchStride());
    assertEquals(plan.rawPatchRows(), result.rawPatchRows());
    assertEquals(plan.mergedRowCount(), result.mergedRowCount());
    assertEquals(256 * 1536, result.values().length);

    assertGoldenSha(
        result.values(),
        SHA_256X256,
        new float[] {
          -1.0f,
          -0.9921568632125854f,
          -0.9843137264251709f,
          -0.9764705896377563f,
          -0.9686274528503418f,
          -0.9607843160629272f,
          -0.9529411792755127f,
          -0.9450980424880981f,
          -0.9372549057006836f,
          -0.929411768913269f,
          -0.9215686321258545f,
          -0.9137254953384399f,
          -0.9058823585510254f,
          -0.8980392217636108f,
          -0.8901960849761963f,
          -0.8823529481887817f
        });
  }

  @Test
  void probeImage64x32MatchesTheReferenceGolden() {
    Qwen3VlImageProcessor processor = pinnedProcessor();
    Qwen3VlImageProcessorResult result = processor.process(probeImage64x32());

    // A real BICUBIC upscale (64x32 to 384x192) through the Pillow-exact ported kernel, then the
    // same rescale/normalize and permutation chain.
    final Qwen3VlGeometryPlan plan = processor.geometry(64, 32);
    assertArrayEquals(new int[] {1, 12, 24}, result.gridThw());
    assertEquals(288, result.rawPatchRows());
    assertEquals(72, result.mergedRowCount());
    assertEquals(1536, result.patchStride());
    assertEquals(plan.rawPatchRows(), result.rawPatchRows());
    assertEquals(plan.mergedRowCount(), result.mergedRowCount());
    assertEquals(288 * 1536, result.values().length);

    assertGoldenSha(
        result.values(),
        SHA_64X32,
        new float[] {
          -1.0f,
          -1.0f,
          -1.0f,
          -1.0f,
          -0.9921568632125854f,
          -0.9921568632125854f,
          -0.9843137264251709f,
          -0.9764705896377563f,
          -0.9686274528503418f,
          -0.9686274528503418f,
          -0.9607843160629272f,
          -0.9529411792755127f,
          -0.9529411792755127f,
          -0.9450980424880981f,
          -0.9372549057006836f,
          -0.9372549057006836f
        });
  }

  @Test
  void noResizeNinetySixBySixtyFourMatchesAnIndependentPermutationAndArithmetic() {
    Qwen3VlImageProcessor processor = noResizeProcessor();
    RgbImage image = indexImage(96, 64);
    Qwen3VlImageProcessorResult result = processor.process(image);

    // The reference succeeds at 96x64 without resizing: 24 raw rows x 1536 floats (the WU2 spec's
    // "9216 floats" shorthand was the post-merge token count x stride; the reference pixel_values
    // is grid_h * grid_w rows, and both the idx formula and the golden SHAs use that row count).
    assertArrayEquals(new int[] {1, 4, 6}, result.gridThw());
    assertEquals(24, result.rawPatchRows());
    assertEquals(6, result.mergedRowCount());
    assertEquals(1536, result.patchStride());

    float[] values = result.values();
    assertEquals(24 * 1536, values.length);

    // Independent check: decode each flat index by successive modulo/division (a different nesting
    // than the implementation's mh/mw/imh/imw/c/t/pr/pc loops) and recompute the exact
    // float32(float64(pixel) * 1/255) -> (x - 0.5f) / 0.5f arithmetic.
    int ps = 16;
    int tps = 2;
    int merge = 2;
    int mergedW = 3;
    for (int i = 0; i < values.length; i++) {
      final int pc = i % ps;
      int r = i / ps;
      final int pr = r % ps;
      r /= ps;
      int t = r % tps;
      r /= tps;
      final int c = r % 3;
      r /= 3;
      int imw = r % merge;
      r /= merge;
      int imh = r % merge;
      r /= merge;
      int mw = r % mergedW;
      int mh = r / mergedW;
      int x = (mw * merge + imw) * ps + pc;
      int y = (mh * merge + imh) * ps + pr;
      float expected = ((float) (indexPixel(x, y, c) * SCALE) - MEAN[c]) / STD[c];
      assertEquals(expected, values[i], "mismatch at flat index " + i);
    }

    // Temporal duplication: within each channel block of 512 floats, the two 256-float temporal
    // slots are identical (both read the same single image frame).
    for (int i = 0; i < values.length / 2; i++) {
      if (i % (tps * ps * ps) < ps * ps) {
        assertEquals(values[i], values[i + ps * ps], "temporal slot mismatch at " + i);
      }
    }
  }

  @Test
  void noResize32x32MatchesDirectNormalizeMappedToTheMergeOrder() {
    Qwen3VlImageProcessor processor = noResizeProcessor();
    RgbImage image = indexImage(32, 32);
    Qwen3VlImageProcessorResult result = processor.process(image);

    assertArrayEquals(new int[] {1, 2, 2}, result.gridThw());
    assertEquals(4, result.rawPatchRows());
    assertEquals(1, result.mergedRowCount());
    assertEquals(1536, result.patchStride());

    // Cross-check the permutation against a direct normalize of the same image (no resize, so the
    // pixel bytes are untouched): every output float must equal the same HWC-normalized float.
    float[] normalized = ImageTransforms.normalize(image, SCALE, MEAN, STD).values();
    float[] expected = new float[4 * 1536];
    for (int imh = 0; imh < 2; imh++) {
      for (int imw = 0; imw < 2; imw++) {
        for (int c = 0; c < 3; c++) {
          for (int t = 0; t < 2; t++) {
            for (int pr = 0; pr < 16; pr++) {
              for (int pc = 0; pc < 16; pc++) {
                int y = imh * 16 + pr;
                int x = imw * 16 + pc;
                int row = imh * 2 + imw;
                int col = (c * 2 + t) * 256 + pr * 16 + pc;
                expected[row * 1536 + col] = normalized[(y * 32 + x) * 3 + c];
              }
            }
          }
        }
      }
    }
    assertArrayEquals(expected, result.values());
  }

  @Test
  void resultAccessorsAreDefensive() {
    Qwen3VlImageProcessorResult result = noResizeProcessor().process(indexImage(32, 32));
    final int length = result.values().length;

    float[] values = result.values();
    float original = values[0];
    values[0] = 99f;
    assertEquals(original, result.values()[0], "mutating a returned copy must not be observable");

    int[] grid = result.gridThw();
    grid[1] = 99;
    assertArrayEquals(new int[] {1, 2, 2}, result.gridThw());

    float[] valuesBefore = result.values();
    float[] destination = new float[length + 7];
    result.copyValuesTo(destination, 7);
    assertArrayEquals(valuesBefore, Arrays.copyOfRange(destination, 7, 7 + length));
    destination[7 + length / 2] = -1f;
    assertArrayEquals(
        valuesBefore, result.values(), "mutating the destination must not be observable");

    assertThrows(IllegalArgumentException.class, () -> result.copyValuesTo(null, 0));
    assertThrows(IndexOutOfBoundsException.class, () -> result.copyValuesTo(destination, -1));
    assertThrows(
        IndexOutOfBoundsException.class, () -> result.copyValuesTo(destination, length + 8));
  }
}
