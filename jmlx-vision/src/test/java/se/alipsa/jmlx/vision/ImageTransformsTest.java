package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

class ImageTransformsTest {

  @Test
  void resizeRejectsInvalidArguments() {
    RgbImage image = new RgbImage(2, 2, new byte[12]);
    IllegalArgumentException missingImage =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageTransforms.resize(null, 2, 2, Resampling.BILINEAR));
    assertEquals("image must not be null", missingImage.getMessage());
    IllegalArgumentException badWidth =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageTransforms.resize(image, 0, 2, Resampling.BILINEAR));
    assertEquals("width must be positive: 0", badWidth.getMessage());
    IllegalArgumentException badHeight =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageTransforms.resize(image, 2, -1, Resampling.BILINEAR));
    assertEquals("height must be positive: -1", badHeight.getMessage());
    IllegalArgumentException badMethod =
        assertThrows(
            IllegalArgumentException.class, () -> ImageTransforms.resize(image, 2, 2, null));
    assertEquals("method must not be null", badMethod.getMessage());
  }

  @Test
  void resizeRejectsOversizedTargets() {
    // width*height must stay within Integer.MAX_VALUE / 3, so the 3-bytes-per-pixel output
    // buffer allocation cannot wrap to a negative size; 50000x50000 also overflows the int
    // product itself, which must be computed in long.
    RgbImage image = new RgbImage(2, 2, new byte[12]);
    IllegalArgumentException beyondIntProduct =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageTransforms.resize(image, 50000, 50000, Resampling.BILINEAR));
    assertEquals(
        "resize target 50000x50000 overflows an int pixel buffer: width*height is 2500000000 "
            + "but must be at most 715827882",
        beyondIntProduct.getMessage());
    IllegalArgumentException withinIntProduct =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageTransforms.resize(image, 40000, 40000, Resampling.BILINEAR));
    assertEquals(
        "resize target 40000x40000 overflows an int pixel buffer: width*height is 1600000000 "
            + "but must be at most 715827882",
        withinIntProduct.getMessage());
  }

  @Test
  void identityResizeIsExactForEveryKernel() {
    byte[] pixels = new byte[3 * 3 * 3];
    for (int i = 0; i < pixels.length; i++) {
      pixels[i] = (byte) ((i * 7) % 256);
    }
    RgbImage image = new RgbImage(3, 3, pixels);
    for (Resampling method : Resampling.values()) {
      assertEquals(image, ImageTransforms.resize(image, 3, 3, method), method.toString());
    }
  }

  @Test
  void solidImageResizeStaysSolidForEveryKernel() {
    byte[] pixels = new byte[4 * 3 * 3];
    Arrays.fill(pixels, (byte) 200);
    RgbImage image = new RgbImage(4, 3, pixels);
    for (Resampling method : Resampling.values()) {
      RgbImage resized = ImageTransforms.resize(image, 10, 7, method);
      assertEquals(10, resized.width());
      assertEquals(7, resized.height());
      for (byte b : resized.pixels()) {
        assertEquals(200, b & 0xFF, method.toString());
      }
    }
  }

  @Test
  void normalizeIdentityIsExact() {
    byte[] pixels = new byte[] {10, 20, 30, 40, 50, 60};
    RgbImage image = new RgbImage(2, 1, pixels);
    ImageTensor tensor =
        ImageTransforms.normalize(image, 1.0f, new float[] {0, 0, 0}, new float[] {1, 1, 1});
    assertArrayEquals(new float[] {10, 20, 30, 40, 50, 60}, tensor.values());
    assertArrayEquals(new int[] {1, 1, 2, 3}, tensor.shape());
    assertEquals(Layout.NHWC, tensor.layout());
  }

  @Test
  void normalizeMatchesHandComputedFloatMath() {
    // float32(float64(pixel) * scale), then (x - mean[c]) / std[c] in float32.
    byte[] pixels = new byte[] {10, 20, 30};
    RgbImage image = new RgbImage(1, 1, pixels);
    ImageTensor exact =
        ImageTransforms.normalize(image, 0.5f, new float[] {1, 2, 3}, new float[] {2, 4, 8});
    assertArrayEquals(new float[] {2.0f, 2.0f, 1.5f}, exact.values());
    // Rounding case: (float)(255 * 0.2f-as-double) is exactly 51.0f, then (51.0 - 0.125) / 0.25
    // is exactly 203.5f.
    RgbImage full = new RgbImage(1, 1, new byte[] {(byte) 255, (byte) 255, (byte) 255});
    ImageTensor rounded =
        ImageTransforms.normalize(
            full, 0.2f, new float[] {0.125f, 0.125f, 0.125f}, new float[] {0.25f, 0.25f, 0.25f});
    assertArrayEquals(new float[] {203.5f, 203.5f, 203.5f}, rounded.values());
  }

  @Test
  void normalizeRescaleFactorCaseMatchesPinnedConfigMath() {
    // (float)(127 * 0.00392156862745098f-as-double) is the exact float32 0.49803925f
    // (bits 0x3efeff00); the pinned config's zero-mean/unit-std identity normalizes it away.
    byte[] pixels = new byte[] {127, 127, 127};
    RgbImage image = new RgbImage(1, 1, pixels);
    ImageTensor tensor =
        ImageTransforms.normalize(
            image, 0.00392156862745098f, new float[] {0, 0, 0}, new float[] {1, 1, 1});
    assertArrayEquals(new float[] {0.49803925f, 0.49803925f, 0.49803925f}, tensor.values());
  }

  @Test
  void normalizeRejectsInvalidArguments() {
    RgbImage image = new RgbImage(1, 1, new byte[] {0, 0, 0});
    IllegalArgumentException missingImage =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(
                    null, 1.0f, new float[] {0, 0, 0}, new float[] {1, 1, 1}));
    assertEquals("image must not be null", missingImage.getMessage());
    IllegalArgumentException nanScale =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(
                    image, Float.NaN, new float[] {0, 0, 0}, new float[] {1, 1, 1}));
    assertEquals("scale must be finite: NaN", nanScale.getMessage());
    IllegalArgumentException infScale =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(
                    image, Float.POSITIVE_INFINITY, new float[] {0, 0, 0}, new float[] {1, 1, 1}));
    assertEquals("scale must be finite: Infinity", infScale.getMessage());
    IllegalArgumentException nullMean =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageTransforms.normalize(image, 1.0f, null, new float[] {1, 1, 1}));
    assertEquals("mean must be an array of exactly three finite values", nullMean.getMessage());
    IllegalArgumentException shortMean =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(image, 1.0f, new float[] {0, 0}, new float[] {1, 1, 1}));
    assertEquals("mean must be an array of exactly three finite values", shortMean.getMessage());
    IllegalArgumentException nanMean =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(
                    image, 1.0f, new float[] {0, 0, Float.NaN}, new float[] {1, 1, 1}));
    assertEquals("mean must be finite: NaN", nanMean.getMessage());
    IllegalArgumentException nullStd =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageTransforms.normalize(image, 1.0f, new float[] {0, 0, 0}, null));
    assertEquals("std must be an array of exactly three finite values", nullStd.getMessage());
    IllegalArgumentException shortStd =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(image, 1.0f, new float[] {0, 0, 0}, new float[] {1, 1}));
    assertEquals("std must be an array of exactly three finite values", shortStd.getMessage());
    IllegalArgumentException zeroStd =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(
                    image, 1.0f, new float[] {0, 0, 0}, new float[] {0, 1, 1}));
    assertEquals("std values must be finite and positive: 0.0", zeroStd.getMessage());
    IllegalArgumentException negativeStd =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageTransforms.normalize(
                    image, 1.0f, new float[] {0, 0, 0}, new float[] {-1, 1, 1}));
    assertEquals("std values must be finite and positive: -1.0", negativeStd.getMessage());
  }
}
