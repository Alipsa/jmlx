package se.alipsa.jmlx.vision;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * The Phi-3.5-Vision (LLaVA-style) image preprocessing chain, ported from the bundled trust-remote-
 * code {@code Phi3VImageProcessor.preprocess} of the pinned {@code
 * microsoft/Phi-3.5-vision-instruct} artifact (findings §6, {@code
 * req/plans/phase7-3c-reference-findings.md}):
 *
 * <ol>
 *   <li>{@code HD_transform}: when the input is portrait ({@code width < height}) the working space
 *       is the transposed image; the scale-selection loop {@code while scale * ceil(scale / ratio)
 *       <= num_crops} (double, {@code ratio = long side / short side}) picks the resize factor; the
 *       image is resized to {@code [int(new_w / ratio), int(scale * 336)]} (the short side is a
 *       double-truncating division) with the reference's default PIL BILINEAR; then the short side
 *       is padded to the next multiple of 336 (top {@code (tar - h) / 2} rows, remainder bottom,
 *       white 255, never left/right); the result is transposed back;
 *   <li>{@code ToTensor()} ({@code float32(pixel / 255)}) and {@code Normalize(mean, std)} ({@code
 *       (x - mean[c]) / std[c]} in float32) on the padded image — the reference's torchvision
 *       compose, transcribed per operation, not the module's {@code ImageTransforms.normalize} (its
 *       rescale is {@code float32(float64(pixel) * factor)}, a different rounding);
 *   <li>the reference's global-image step: {@code F.interpolate(..., size=(336, 336),
 *       mode='bicubic')} run on the normalized float32 tensor, ported by {@link TorchBicubic}
 *       (align_corners=false, no antialias, A = -0.75, the pinned torch v2.6.0 CPU float32 path);
 *   <li>the sub-crop split: the normalized padded image reshaped to {@code (gridH, gridW, 3, 336,
 *       336)} in h-major block order (no resampling, pure copies);
 *   <li>the slot assembly: {@code [global] + crops}, zero-padded to {@code num_crops + 1} slots.
 * </ol>
 *
 * <p>The output of {@link #process(RgbImage)} is one reference batch row: {@code slotCount() * 3 *
 * 336 * 336} float32 values, slot-major then CHW ({@code [slot][channel][row][column]}), where slot
 * 0 is the global bicubic image, slots 1..cropBlocks the sub-crops in h-major order, and any
 * remaining slots all-zero padding ({@code pad_to_max_num_crops_tensor} zero-pads only, never
 * truncates; with the reference's scale loop {@code cropBlocks <= num_crops}, so with the pinned
 * artifact every row has exactly 5 slots). The reference's per-image {@code image_sizes} entry and
 * {@code num_img_tokens} are carried by the result's {@link Phi3VGeometryPlan}.
 *
 * <p>The input must already be an {@link RgbImage} (decoded and converted to RGB; the reference's
 * {@code convert("RGB")} step is therefore implicit in the decode, and a config with {@code
 * do_convert_rgb: false} is rejected at parse time because the ported decode path always delivers
 * RGB). The file/bytes entry points decode with the Phi-3.5-Vision policy (drop-alpha + EXIF-
 * ignore, see {@link #processFile(Path, ImageDecodeLimits)}).
 */
public final class Phi3VImageProcessor {
  /** The crop and global-image side, in pixels (336 for the reference). */
  private static final int CROP = 336;

  private final Phi3VProcessorConfig config;

  /**
   * The geometry of the working (possibly transposed) space: the long side is always horizontal.
   *
   * @param transposed whether the working space is the transposed image ({@code width < height})
   * @param resizedLong HD-resize target of the long side ({@code scale * 336})
   * @param resizedShort HD-resize target of the short side ({@code int(resizedLong / ratio)},
   *     double truncation)
   * @param paddedShort the short side after 336-multiple padding (a multiple of 336, at least
   *     {@code resizedShort})
   */
  private record WorkingGeometry(
      boolean transposed, int resizedLong, int resizedShort, int paddedShort) {}

  /**
   * Creates a processor with the given validated configuration.
   *
   * @param config validated processor configuration; null is rejected
   * @throws IllegalArgumentException if {@code config} is null
   */
  public Phi3VImageProcessor(Phi3VProcessorConfig config) {
    if (config == null) {
      throw new IllegalArgumentException("config must not be null");
    }
    this.config = config;
  }

  /**
   * The pure geometry of the preprocessing chain for an input of the given dimensions, without
   * touching any pixels: the resize and padding targets in the original orientation, the 336-block
   * crop grid, the slot count and the per-image image-token count, so it can be shared with callers
   * that plan from dimensions alone (for example the zero-token generation path of the model
   * layer).
   *
   * @param width input image width in pixels
   * @param height input image height in pixels
   * @return the geometry plan
   * @throws IllegalArgumentException if the dimensions are not positive, or the aspect ratio (long
   *     side over short side, both orientations) exceeds {@code num_crops * 336}, where the
   *     reference's double-truncating short-side resize would truncate to 0
   */
  public Phi3VGeometryPlan geometry(int width, int height) {
    if (width <= 0 || height <= 0) {
      throw new IllegalArgumentException(
          "image dimensions must be positive: " + width + "x" + height);
    }
    WorkingGeometry work = workingGeometry(width, height);
    int resizedWidth;
    int resizedHeight;
    int paddedWidth;
    int paddedHeight;
    if (work.transposed()) {
      // The working space is the transposed image (long side horizontal); its padded short side
      // (the working height) becomes the original width on the transpose back, and the unpadded
      // long side becomes the original height.
      resizedWidth = work.resizedShort();
      resizedHeight = work.resizedLong();
      paddedWidth = work.paddedShort();
      paddedHeight = work.resizedLong();
    } else {
      resizedWidth = work.resizedLong();
      resizedHeight = work.resizedShort();
      paddedWidth = work.resizedLong();
      paddedHeight = work.paddedShort();
    }
    int gridH = paddedHeight / CROP;
    int gridW = paddedWidth / CROP;
    int cropBlocks = gridH * gridW;
    int numCrops = config.numCrops();
    int slotCount = Math.max(cropBlocks + 1, numCrops + 1);
    // The reference's per-image token formula, evaluated on the padded sizes (h-major grid):
    // (h//336 * w//336 + 1) * 144 + 1 + (h//336 + 1) * 12.
    int numImgTokens = (cropBlocks + 1) * 144 + 1 + (gridH + 1) * 12;
    return new Phi3VGeometryPlan(
        width,
        height,
        work.transposed(),
        resizedWidth,
        resizedHeight,
        paddedWidth,
        paddedHeight,
        gridH,
        gridW,
        cropBlocks,
        numCrops,
        slotCount,
        numImgTokens);
  }

  /**
   * Runs the full preprocessing chain on one image and returns the reference's one batch row (see
   * the class Javadoc for the slot layout). The pixel chain consumes the working geometry of {@link
   * #geometry(int, int)} for the same dimensions, so geometry and pixels cannot drift apart.
   *
   * @param image the decoded RGB input; null is rejected
   * @return the slot-major {@code [slotCount, 3, 336, 336]} float32 tensor with its geometry plan
   * @throws IllegalArgumentException if {@code image} is null, or its dimensions fail the geometry
   *     checks (positivity, extreme aspect ratio)
   */
  public Phi3VImageProcessorResult process(RgbImage image) {
    if (image == null) {
      throw new IllegalArgumentException("image must not be null");
    }
    Phi3VGeometryPlan plan = geometry(image.width(), image.height());
    WorkingGeometry work = workingGeometry(image.width(), image.height());
    RgbImage working = image;
    if (work.transposed()) {
      working = transpose(working);
    }
    // The reference resizes unconditionally (torchvision F.resize, default PIL BILINEAR on 8-bit,
    // byte-identical to the module's Pillow kernel per the WP1 probe p5).
    RgbImage resized =
        ImageTransforms.resize(
            working, work.resizedLong(), work.resizedShort(), Resampling.BILINEAR);
    RgbImage padded = padTopBottom(resized, work.paddedShort());
    if (work.transposed()) {
      padded = transpose(padded);
    }
    int width = padded.width();
    int height = padded.height();
    // ToTensor (float32(pixel / 255)) + Normalize ((x - mean[c]) / std[c], float32), HWC.
    float[] norm = toTensorNormalize(padded, config.imageMean(), config.imageStd());
    // Global image: the reference interpolates the CHW normalized tensor to 336x336 bicubic.
    float[] globalChw =
        TorchBicubic.resizeChw(hwcToChw(norm, width, height), height, width, CROP, CROP);
    long total = (long) plan.slotCount() * 3L * CROP * CROP;
    if (total > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "slot count "
              + plan.slotCount()
              + " times 3*336*336 overflows a signed 32-bit float count: "
              + total);
    }
    float[] out = new float[(int) total];
    // Slot 0: the global image, already CHW 3x336x336.
    System.arraycopy(globalChw, 0, out, 0, globalChw.length);
    // Slots 1..cropBlocks: h-major 336x336 block copies of the normalized padded image (the
    // reference's reshape/permute, no resampling). The source is HWC-interleaved, so each
    // destination channel row reads the channel element of every third source pixel.
    int plane = CROP * CROP;
    for (int bh = 0; bh < plan.gridH(); bh++) {
      for (int bw = 0; bw < plan.gridW(); bw++) {
        int slot = 1 + bh * plan.gridW() + bw;
        int base = slot * 3 * plane;
        for (int c = 0; c < 3; c++) {
          int dst = base + c * plane;
          for (int r = 0; r < CROP; r++) {
            int src = (r + bh * CROP) * width * 3 + bw * (CROP * 3) + c;
            int o = dst + r * CROP;
            for (int q = 0; q < CROP; q++) {
              out[o + q] = norm[src + q * 3];
            }
          }
        }
      }
    }
    // Slots cropBlocks+1..slotCount-1 stay all-zero: pad_to_max_num_crops_tensor zero-pads.
    return new Phi3VImageProcessorResult(plan, out);
  }

  /**
   * Decodes the PNG or JPEG file at {@code file} with the Phi-3.5-Vision file policy and runs the
   * full preprocessing chain on the result ({@link ImageDecoder#decode} with {@code
   * AlphaPolicy.DROP_ALPHA} and {@link ExifOrientationPolicy#IGNORE}, then {@link
   * #process(RgbImage)}).
   *
   * <p>The Phi-3.5-Vision file policy is drop-alpha + EXIF-ignore because the pinned reference
   * pipeline never calls {@code exif_transpose} and its {@code convert("RGB")} drops alpha without
   * compositing (findings §6). This is a deliberate contrast with both the module-default decode
   * ({@code COMPOSITE_OVER_WHITE} + {@code IGNORE}) and the Qwen3-VL policy ({@code DROP_ALPHA} +
   * {@code APPLY}).
   *
   * @param file existing PNG or JPEG file
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @return the slot-major tensor of the decoded image with its geometry plan
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   * @throws IllegalArgumentException if the decoded image dimensions fail the geometry checks
   */
  public Phi3VImageProcessorResult processFile(Path file, ImageDecodeLimits limits)
      throws IOException {
    return process(
        ImageDecoder.decode(file, limits, AlphaPolicy.DROP_ALPHA, ExifOrientationPolicy.IGNORE));
  }

  /**
   * In-memory counterpart of {@link #processFile(Path, ImageDecodeLimits)}: where that method reads
   * the image bytes from a file on disk, this one decodes the PNG or JPEG content of {@code bytes}
   * directly from memory (no filesystem access) and then runs the full preprocessing chain on the
   * result with the same Phi-3.5-Vision file policy (see {@code processFile}'s Javadoc).
   *
   * @param bytes PNG or JPEG image bytes; null is rejected; the array is consumed and never
   *     retained
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @return the slot-major tensor of the decoded image with its geometry plan
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   * @throws IllegalArgumentException if {@code bytes} is null, or the decoded image dimensions fail
   *     the geometry checks
   */
  public Phi3VImageProcessorResult processEncoded(byte[] bytes, ImageDecodeLimits limits)
      throws IOException {
    if (bytes == null) {
      throw new IllegalArgumentException("bytes must not be null");
    }
    return process(
        ImageDecoder.decode(
            new ByteArrayInputStream(bytes),
            limits,
            AlphaPolicy.DROP_ALPHA,
            ExifOrientationPolicy.IGNORE));
  }

  /**
   * The shared working-geometry computation for {@link #geometry(int, int)} and {@link
   * #process(RgbImage)} (findings §6, the reference {@code HD_transform} transcribed): the scale
   * loop runs in double exactly as the reference ({@code scale * ceil(scale / ratio) <=
   * num_crops}); {@code new_h} is the reference's double {@code int()} truncation. The aspect ratio
   * above {@code num_crops * 336} is rejected before any resize: there the truncated short side
   * would be 0 and the reference's PIL resize would fail (the plan's WP2 extreme-ratio item; the
   * reference's own failure mode is not the port's contract).
   */
  private WorkingGeometry workingGeometry(int width, int height) {
    int w = width;
    int h = height;
    boolean transposed = w < h;
    if (transposed) {
      int t = w;
      w = h;
      h = t;
    }
    double ratio = (double) w / (double) h;
    int numCrops = config.numCrops();
    if (ratio > (double) numCrops * CROP) {
      throw new IllegalArgumentException(
          "aspect ratio "
              + ratio
              + " exceeds num_crops * 336 = "
              + (long) numCrops * CROP
              + ": the reference's double-truncating short-side resize would truncate to 0 for "
              + width
              + "x"
              + height);
    }
    int scale = 1;
    while ((double) scale * Math.ceil((double) scale / ratio) <= (double) numCrops) {
      scale++;
    }
    scale--;
    long newLong = (long) scale * CROP;
    if (newLong > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "the HD resize long edge " + newLong + " overflows a signed 32-bit dimension");
    }
    // int(new_w / ratio): the double division truncated, the reference's exact arithmetic.
    int newShort = (int) (newLong / ratio);
    if (newShort <= 0) {
      throw new IllegalArgumentException(
          "the HD resize short edge truncated to 0 for " + width + "x" + height);
    }
    int tar = (int) (Math.ceil((double) newShort / CROP) * CROP);
    return new WorkingGeometry(transposed, (int) newLong, newShort, tar);
  }

  /**
   * {@code ToTensor()} + {@code Normalize(mean, std)} in one HWC pass: per channel {@code
   * (float32(pixel / 255) - mean[c]) / std[c]}, each step a single IEEE 754 binary32 operation —
   * the reference's torchvision {@code ToTensor} divides in float32 (no double reciprocal
   * multiply), unlike the module's {@code ImageTransforms.normalize}.
   */
  private static float[] toTensorNormalize(RgbImage image, float[] mean, float[] std) {
    byte[] pixels = image.pixelsRaw();
    float[] out = new float[pixels.length];
    for (int i = 0; i < pixels.length; i += 3) {
      out[i] = ((pixels[i] & 0xFF) / 255.0f - mean[0]) / std[0];
      out[i + 1] = ((pixels[i + 1] & 0xFF) / 255.0f - mean[1]) / std[1];
      out[i + 2] = ((pixels[i + 2] & 0xFF) / 255.0f - mean[2]) / std[2];
    }
    return out;
  }

  /** Copies an interleaved HWC float32 buffer to contiguous CHW (the bicubic kernel's input). */
  private static float[] hwcToChw(float[] hwc, int width, int height) {
    int plane = width * height;
    float[] chw = new float[3 * plane];
    for (int c = 0; c < 3; c++) {
      int src = c;
      int dst = c * plane;
      for (int p = 0; p < plane; p++, src += 3) {
        chw[dst + p] = hwc[src];
      }
    }
    return chw;
  }

  /** The matrix transpose (PIL {@code Image.TRANSPOSE}: swap x and y, no flip). */
  private static RgbImage transpose(RgbImage image) {
    int w = image.width();
    int h = image.height();
    byte[] src = image.pixelsRaw();
    byte[] dst = new byte[w * h * 3];
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        System.arraycopy(src, (y * w + x) * 3, dst, (x * h + y) * 3, 3);
      }
    }
    return new RgbImage(h, w, dst);
  }

  /**
   * The reference {@code padding_336}: top {@code (tar - h) / 2} rows (integer division), the
   * remainder bottom, white 255, never left/right.
   */
  private static RgbImage padTopBottom(RgbImage image, int targetHeight) {
    int w = image.width();
    int h = image.height();
    int top = (targetHeight - h) / 2;
    byte[] src = image.pixelsRaw();
    byte[] dst = new byte[w * targetHeight * 3];
    Arrays.fill(dst, (byte) 255);
    for (int y = 0; y < h; y++) {
      System.arraycopy(src, y * w * 3, dst, (y + top) * w * 3, w * 3);
    }
    return new RgbImage(w, targetHeight, dst);
  }
}
