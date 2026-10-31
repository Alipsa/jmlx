package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Structural unit tests for the Phi-3.5-Vision preprocessing chain: the padding policy, the slot
 * layout and its zero padding, the ToTensor/Normalize float32 arithmetic, and the {@link
 * TorchBicubic} port's identity and constant-image behavior. The byte-exact comparison against the
 * pinned reference is the {@code phi3v} image-oracle family ({@code ImageOraclePhi3VTest}).
 */
class Phi3VImageProcessorTest {

  private static final int CROP = 336;
  private static final int PLANE = CROP * CROP;

  private static final float[] CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
  private static final float[] CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};

  private static Phi3VImageProcessor clipProcessor() {
    return new Phi3VImageProcessor(
        Phi3VProcessorConfig.builder().numCrops(4).imageMean(CLIP_MEAN).imageStd(CLIP_STD).build());
  }

  /** {@code (pixel / 255) - mean} over {@code std} for a white (255) pixel, float32. */
  private static float white(int c) {
    return (1.0f - CLIP_MEAN[c]) / CLIP_STD[c];
  }

  /** The same for a black (0) pixel. */
  private static float black(int c) {
    return (0.0f - CLIP_MEAN[c]) / CLIP_STD[c];
  }

  private static float at(float[] out, int slot, int c, int r, int q) {
    return out[slot * 3 * PLANE + c * PLANE + r * CROP + q];
  }

  private static RgbImage solid(int width, int height, int v) {
    byte[] pixels = new byte[width * height * 3];
    java.util.Arrays.fill(pixels, (byte) v);
    return new RgbImage(width, height, pixels);
  }

  @Test
  void nullArgumentsAreRejected() {
    assertThrows(IllegalArgumentException.class, () -> new Phi3VImageProcessor(null));
    assertThrows(IllegalArgumentException.class, () -> clipProcessor().process(null));
  }

  @Test
  void paddedRegionsCarryTheNormalizedWhiteValue() {
    // The verified trunc geometry: 300x301 (portrait) works in the transposed 301x300 space,
    // resizes to 672x669 and pads the short side to 672 (top 1, bottom 2) before the transpose
    // back, so the white bands are the original left column and the two rightmost columns.
    Phi3VImageProcessor processor = clipProcessor();
    Phi3VImageProcessorResult result = processor.process(solid(300, 301, 0));
    Phi3VGeometryPlan plan = result.plan();
    assertEquals(669, plan.resizedWidth());
    assertEquals(672, plan.resizedHeight());
    assertEquals(672, plan.paddedWidth());
    assertEquals(672, plan.paddedHeight());
    assertEquals(5, plan.slotCount());
    float[] out = result.values();
    assertEquals(5L * 3 * PLANE, out.length);

    // Slot 0 (the global bicubic image) is a resample of the whole padded image; the oracle
    // family byte-verifies it. The four crop slots are pure copies and are checked per pixel:
    // h-major slot order makes slot 1 (0,0), 2 (0,1), 3 (1,0), 4 (1,1); the white columns are
    // original 0, 670 and 671.
    int[][] whiteLocalColumns = {{0}, {334, 335}, {0}, {334, 335}};
    for (int k = 0; k < 4; k++) {
      int slot = 1 + k;
      for (int c = 0; c < 3; c++) {
        for (int r : new int[] {0, 1, 335}) {
          for (int q = 0; q < CROP; q++) {
            boolean isWhiteColumn = false;
            for (int w : whiteLocalColumns[k]) {
              if (q == w) {
                isWhiteColumn = true;
              }
            }
            float expected = isWhiteColumn ? white(c) : black(c);
            assertEquals(
                Float.floatToIntBits(expected),
                Float.floatToIntBits(at(out, slot, c, r, q)),
                "slot " + slot + " channel " + c + " row " + r + " column " + q);
          }
        }
      }
    }
  }

  @Test
  void zeroPaddedSlotsAreAllZero() {
    // Wide 672x336: two crop blocks, so slots 3 and 4 are pad_to_max_num_crops_tensor zeros.
    Phi3VImageProcessor processor = clipProcessor();
    Phi3VImageProcessorResult result = processor.process(solid(672, 336, 0));
    assertEquals(2, result.plan().cropBlocks());
    assertEquals(5, result.plan().slotCount());
    float[] out = result.values();
    for (int slot : new int[] {3, 4}) {
      for (int j = 0; j < 3 * PLANE; j += 997) {
        assertEquals(0.0f, out[slot * 3 * PLANE + j], "slot " + slot + " must be all-zero padding");
      }
    }
    // The two crop slots are pure copies of the normalized (constant) image.
    for (int slot : new int[] {1, 2}) {
      for (int c = 0; c < 3; c++) {
        assertEquals(
            Float.floatToIntBits(black(c)),
            Float.floatToIntBits(at(out, slot, c, 5, 200)),
            "crop slot " + slot + " channel " + c);
      }
    }
    // The global slot is the bicubic resize of the constant normalized image: the cubic weights
    // sum to 1, so every output pixel is the constant within a few ulps (the exact bits are the
    // oracle family's job).
    for (int i = 0; i < 3 * PLANE; i += 517) {
      float v = out[i];
      float reference = black(i / PLANE);
      assertTrue(
          Math.abs(v - reference) <= 16 * Math.ulp(reference),
          "global slot pixel " + i + ": " + v + " vs " + reference);
    }
  }

  @Test
  void toTensorNormalizeUsesFloat32Division() {
    // Mean 0, std 1 isolates the ToTensor division: the crop slots are pure copies, so their
    // bits pin the float32(pixel / 255) rounding (the 0x3b808081/0x3f008081 constants are the
    // float32 nearest values of 1/255 and 128/255).
    Phi3VImageProcessor processor =
        new Phi3VImageProcessor(
            Phi3VProcessorConfig.builder()
                .numCrops(1)
                .imageMean(new float[] {0.0f, 0.0f, 0.0f})
                .imageStd(new float[] {1.0f, 1.0f, 1.0f})
                .build());
    assertEquals(1, processor.geometry(1, 1).cropBlocks());
    for (int[] pixelBits :
        new int[][] {{1, 0x3b808081}, {128, 0x3f008081}, {255, 0x3f800000}, {0, 0x00000000}}) {
      Phi3VImageProcessorResult result = processor.process(solid(1, 1, pixelBits[0]));
      float[] out = result.values();
      assertEquals(2L * 3 * PLANE, out.length, "global + one crop");
      for (int i = 3 * PLANE; i < out.length; i += 331) {
        assertEquals(
            pixelBits[1],
            Float.floatToIntBits(out[i]),
            "ToTensor pixel " + pixelBits[0] + " at offset " + i);
      }
    }
  }

  @Test
  void torchBicubicSameSizeResizeIsBitExact() {
    // A same-size bicubic resize is the identity: real = i, lambda = 0, weights (0, 1, 0, 0), so
    // the accumulation is input[i] * 1.0 plus zero-weight products, returning the input bits
    // unchanged for every non-zero value. At a zero input the IEEE fadd sequence yields +0
    // regardless of the input zero's sign (the same as torch's fmaf/fadd accumulation).
    int height = 8;
    int width = 9;
    float[] pattern = new float[3 * height * width];
    for (int i = 0; i < pattern.length; i++) {
      int n = (i * 31 + 7) % 7 - 3;
      pattern[i] = n == 0 ? 0.0f : n / 4.0f;
    }
    float[] out = TorchBicubic.resizeChw(pattern, height, width, height, width);
    for (int i = 0; i < pattern.length; i++) {
      assertEquals(
          Float.floatToIntBits(pattern[i]),
          Float.floatToIntBits(out[i]),
          "same-size identity bit " + i);
    }
  }

  @Test
  void torchBicubicConstantImageStaysConstant() {
    // The four cubic weights sum to 1, so a constant plane resizes to the same constant: exact
    // for zero (every product is a signed zero and the sums cancel to +0), within a few ulps
    // otherwise.
    for (float constant : new float[] {0.5f, -0.5f, 1.0f}) {
      float[] in = new float[3 * 4 * 5];
      java.util.Arrays.fill(in, constant);
      float[] out = TorchBicubic.resizeChw(in, 4, 5, 6, 7);
      for (float v : out) {
        assertTrue(
            Math.abs(v - constant) <= 16 * Math.ulp(constant),
            "constant " + constant + " drifted to " + v);
      }
    }
    float[] zeros = new float[3 * 4 * 5];
    float[] out = TorchBicubic.resizeChw(zeros, 4, 5, 6, 7);
    for (float v : out) {
      // The IEEE fadd sequence of zero products sums to the positive zero.
      assertEquals(0, Float.compare(v, 0.0f), "a zero plane must resize to +0, got " + v);
    }
  }

  @Test
  void rangedCopyCopiesOnlyTheRequestedRange() {
    Phi3VImageProcessorResult result = clipProcessor().process(solid(CROP, CROP, 123));
    int slots = result.plan().slotCount();
    int slotLength = 3 * PLANE;
    float[] whole = result.values();
    // Per-slot ranges reassemble the whole row, and the real-slot prefix a prefill uploads is a
    // prefix of it: no padded tail, no allocation of one.
    float[] rebuilt = new float[whole.length];
    for (int slot = 0; slot < slots; slot++) {
      result.copyValuesTo(rebuilt, slot * slotLength, slot * slotLength, slotLength);
    }
    assertTrue(java.util.Arrays.equals(whole, rebuilt), "per-slot ranges must reassemble the row");
    float[] prefix = new float[slotLength];
    result.copyValuesTo(prefix, 0, 0, slotLength);
    assertTrue(
        java.util.Arrays.equals(whole, 0, slotLength, prefix, 0, slotLength),
        "the real-slot prefix must be a prefix of the row");
    // The two-argument overload still copies the entire row.
    float[] bulk = new float[whole.length];
    result.copyValuesTo(bulk, 0);
    assertTrue(java.util.Arrays.equals(whole, bulk), "the two-argument copy must copy the row");
  }

  @Test
  void rangedCopyRejectsInvalidRanges() {
    Phi3VImageProcessorResult result = clipProcessor().process(solid(CROP, CROP, 123));
    float[] destination = new float[2];
    assertThrows(IllegalArgumentException.class, () -> result.copyValuesTo(null, 0, 0, 1));
    assertThrows(IllegalArgumentException.class, () -> result.copyValuesTo(destination, 0, -1, 1));
    assertThrows(IllegalArgumentException.class, () -> result.copyValuesTo(destination, 0, 0, -1));
    assertThrows(IndexOutOfBoundsException.class, () -> result.copyValuesTo(destination, 2, 0, 1));
    assertThrows(IndexOutOfBoundsException.class, () -> result.copyValuesTo(destination, 1, 0, 2));
    int rowLength = result.plan().slotCount() * 3 * PLANE;
    assertThrows(
        IndexOutOfBoundsException.class, () -> result.copyValuesTo(destination, 0, rowLength, 1));
    assertThrows(
        IndexOutOfBoundsException.class,
        () -> result.copyValuesTo(destination, 0, rowLength - 1, 2));
  }

  @Test
  void torchBicubicRejectsInvalidArguments() {
    assertThrows(IllegalArgumentException.class, () -> TorchBicubic.resizeChw(null, 4, 5, 6, 7));
    assertThrows(
        IllegalArgumentException.class,
        () -> TorchBicubic.resizeChw(new float[3 * 4 * 5], 4, 4, 6, 7));
    assertThrows(
        IllegalArgumentException.class, () -> TorchBicubic.resizeChw(new float[60], 0, 5, 6, 7));
    assertThrows(
        IllegalArgumentException.class, () -> TorchBicubic.resizeChw(new float[60], 4, 5, 6, 0));
  }
}
