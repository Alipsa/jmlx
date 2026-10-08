package se.alipsa.jmlx.vision;

/**
 * Stateless image transforms shared by the {@link SmolVlmImageProcessor} chain: Pillow-exact {@link
 * #resize} and float32 {@link #normalize} (rescale + mean/std, in that order, matching the pinned
 * transformers reference arithmetic).
 */
public final class ImageTransforms {
  private ImageTransforms() {}

  /**
   * Resizes {@code image} to {@code width x height} with the pinned Pillow kernel (bit-exact uint8
   * target; see {@code NOTICE} and the reference findings for the measured bounds).
   *
   * @param image source image
   * @param width positive target width
   * @param height positive target height
   * @param method resampling kernel
   * @throws IllegalArgumentException if an argument is null, a target dimension is not positive, or
   *     {@code width * height * 3} (or the two-pass intermediate buffer, for wide-short targets
   *     from tall sources) overflows an int
   */
  public static RgbImage resize(RgbImage image, int width, int height, Resampling method) {
    if (image == null) {
      throw new IllegalArgumentException("image must not be null");
    }
    if (width <= 0) {
      throw new IllegalArgumentException("width must be positive: " + width);
    }
    if (height <= 0) {
      throw new IllegalArgumentException("height must be positive: " + height);
    }
    if (method == null) {
      throw new IllegalArgumentException("method must not be null");
    }
    if ((long) width * height > Integer.MAX_VALUE / 3) {
      throw new IllegalArgumentException(
          "resize target "
              + width
              + "x"
              + height
              + " overflows an int pixel buffer: width*height is "
              + (long) width * height
              + " but must be at most "
              + (Integer.MAX_VALUE / 3));
    }
    // pixelsRaw() and ofUnchecked() keep the no-copy internal path: the resampler allocates its
    // own output (a copy only in the identity case, mirroring the reference's ImagingCopy), and
    // the result is adopted without the public constructor's clone.
    byte[] source = image.pixelsRaw();
    byte[] resized =
        PillowResample.resize(source, image.width(), image.height(), width, height, method);
    return RgbImage.ofUnchecked(width, height, resized);
  }

  /**
   * Rescales (uint8 to float32 via {@code float32(float64(pixel) * scale)}) and normalizes ({@code
   * (rescaled - mean[c]) / std[c]} in float32), matching the pinned transformers {@code rescale} +
   * {@code normalize} arithmetic. The result is a single-batch NHWC tensor.
   *
   * @param image source image
   * @param scale finite rescale factor, the double parsed from {@code rescale_factor} in {@code
   *     preprocessor_config.json}, passed through without float32 narrowing so the rescale matches
   *     the reference's {@code float32(float64(pixel) * factor)} arithmetic exactly
   * @param mean exactly three finite per-channel means
   * @param std exactly three finite positive per-channel standard deviations
   * @throws IllegalArgumentException on null arguments, non-finite scale/mean, non-positive std, or
   *     wrong array lengths
   */
  public static ImageTensor normalize(RgbImage image, double scale, float[] mean, float[] std) {
    if (image == null) {
      throw new IllegalArgumentException("image must not be null");
    }
    if (!Double.isFinite(scale)) {
      throw new IllegalArgumentException("scale must be finite: " + scale);
    }
    if (mean == null || mean.length != 3) {
      throw new IllegalArgumentException("mean must be an array of exactly three finite values");
    }
    for (float m : mean) {
      if (!Float.isFinite(m)) {
        throw new IllegalArgumentException("mean must be finite: " + m);
      }
    }
    if (std == null || std.length != 3) {
      throw new IllegalArgumentException("std must be an array of exactly three finite values");
    }
    for (float s : std) {
      if (!Float.isFinite(s) || s <= 0.0f) {
        throw new IllegalArgumentException("std values must be finite and positive: " + s);
      }
    }

    int width = image.width();
    int height = image.height();
    float[] values = new float[width * height * 3];
    // pixelsRaw(): the source buffer is only read here; the values buffer is fresh and is copied
    // exactly once, by the ImageTensor constructor.
    byte[] pixels = image.pixelsRaw();
    int i = 0;
    for (int p = 0; p < pixels.length; p += 3) {
      // float32(float64(pixel) * scale), then (x - mean[c]) / std[c] in float32.
      values[i] = ((float) ((pixels[p] & 0xFF) * scale) - mean[0]) / std[0];
      values[i + 1] = ((float) ((pixels[p + 1] & 0xFF) * scale) - mean[1]) / std[1];
      values[i + 2] = ((float) ((pixels[p + 2] & 0xFF) * scale) - mean[2]) / std[2];
      i += 3;
    }
    return new ImageTensor(values, new int[] {1, height, width, 3}, Layout.NHWC);
  }
}
