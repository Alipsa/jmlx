package se.alipsa.jmlx.vision;

/**
 * The Qwen3-VL image preprocessing chain, ported from the pinned transformers 4.57.6 slow {@code
 * Qwen2VLImageProcessor._preprocess} image path (the video path is out of scope):
 *
 * <ol>
 *   <li>optional {@code smart_resize} of the image to a multiple of {@code patch_size * merge_size}
 *       whose pixel count lies in {@code [min_pixels, max_pixels]};
 *   <li>optional Pillow-exact BICUBIC (or configured-kernel) resize through {@code
 *       ImageTransforms.resize};
 *   <li>rescale ({@code float32(float64(pixel) * rescale_factor)}) and normalize ({@code (x -
 *       mean[c]) / std[c]} in float32);
 *   <li>temporal duplication of the one image frame to {@code temporal_patch_size} identical
 *       frames, then the reference's merge-block patch permutation and C-order flatten.
 * </ol>
 *
 * <p>The output has {@code grid_h * grid_w} rows (the raw 16x16 patch count, which is the first
 * dimension of the reference's {@code pixel_values}: 256 for a 256x256 image, 288 for the 64x32
 * probe image resized to 384x192) of {@code 3 * temporal_patch_size * patch_size * patch_size}
 * floats each (1,536 for the pinned artifacts). The 2x2 merge happens later, inside the vision
 * tower: the token count the tower produces is {@code grid_h * grid_w / (merge_size * merge_size)}
 * ({@link Qwen3VlGeometryPlan#mergedRowCount()}), not the row count here.
 *
 * <p>The input must already be an {@link RgbImage} (decoded and converted to RGB; the reference's
 * {@code convert_to_rgb} step is therefore implicit in the decode, and a config with {@code
 * do_convert_rgb: false} is rejected at parse time because the ported decode path always delivers
 * RGB). The output of {@link #process(RgbImage)} is the reference's flattened merge-block patch
 * tensor — <b>not</b> a regular NCHW/NHWC image tensor; do not interpret its layout as an image.
 */
public final class Qwen3VlImageProcessor {
  private final Qwen3VlProcessorConfig config;

  /**
   * Creates a processor with the given validated configuration.
   *
   * @param config validated processor configuration; null is rejected
   * @throws IllegalArgumentException if {@code config} is null
   */
  public Qwen3VlImageProcessor(Qwen3VlProcessorConfig config) {
    if (config == null) {
      throw new IllegalArgumentException("config must not be null");
    }
    this.config = config;
  }

  /**
   * The pure geometry of the preprocessing chain for an input of the given dimensions, without
   * touching any pixels: the resize target, the patch grid and the merge-block row count, plus the
   * raw-patch-row budget check, so it can be shared with callers that plan from dimensions alone
   * (for example the zero-token generation path of the model layer).
   *
   * @param width input image width in pixels
   * @param height input image height in pixels
   * @return the geometry plan
   * @throws IllegalArgumentException if the dimensions are not positive, the aspect ratio exceeds
   *     the reference's guard, resizing is disabled and a dimension is not divisible by {@code
   *     patch_size * merge_size}, or the raw patch rows exceed the configured budget
   */
  public Qwen3VlGeometryPlan geometry(int width, int height) {
    if (width <= 0 || height <= 0) {
      throw new IllegalArgumentException(
          "image dimensions must be positive: " + width + "x" + height);
    }
    int resizedWidth = width;
    int resizedHeight = height;
    if (config.doResize()) {
      int[] resized =
          smartResize(height, width, config.factor(), config.minPixels(), config.maxPixels());
      resizedHeight = resized[0];
      resizedWidth = resized[1];
    } else {
      // smart_resize guarantees divisibility by the factor; without it the reference's reshape
      // would fail, so reject early with the requirement named.
      int factor = config.factor();
      if (height % factor != 0) {
        throw new IllegalArgumentException(
            "height must be divisible by patch_size * merge_size = "
                + factor
                + " when resizing is disabled: "
                + height);
      }
      if (width % factor != 0) {
        throw new IllegalArgumentException(
            "width must be divisible by patch_size * merge_size = "
                + factor
                + " when resizing is disabled: "
                + width);
      }
    }
    long rawPatchRows =
        (long) (resizedHeight / config.patchSize()) * (resizedWidth / config.patchSize());
    if (rawPatchRows > config.rawRowBudget()) {
      throw new IllegalArgumentException(
          "raw patch rows "
              + rawPatchRows
              + " exceed the raw patch row budget of "
              + config.rawRowBudget()
              + " for "
              + width
              + "x"
              + height);
    }
    int gridH = resizedHeight / config.patchSize();
    int gridW = resizedWidth / config.patchSize();
    int mergedRowCount = (int) (rawPatchRows / ((long) config.mergeSize() * config.mergeSize()));
    return new Qwen3VlGeometryPlan(
        width,
        height,
        resizedWidth,
        resizedHeight,
        1,
        gridH,
        gridW,
        (int) rawPatchRows,
        mergedRowCount,
        config.patchStride());
  }

  /**
   * Runs the full preprocessing chain on one image and returns the reference's flattened
   * merge-block patch tensor (see the class Javadoc: {@code grid_h * grid_w} rows of {@code 3 *
   * temporal_patch_size * patch_size * patch_size} floats, not an image layout).
   *
   * <p>The pixel chain consumes the plan of {@link #geometry(int, int)} for the same dimensions, so
   * geometry and pixels cannot drift apart. The resize always goes through {@code
   * ImageTransforms.resize} even when the target equals the source size: the reference resizes
   * unconditionally under {@code do_resize}, and Pillow's same-size resize is an exact {@code
   * ImagingCopy}, which the ported kernel mirrors.
   *
   * <p>The flatten is the reference's {@code reshape(1, temporal, 3, grid_h, merge, patch, grid_w,
   * merge, patch)} followed by {@code transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)} (findings §4.6):
   * row-major over {@code (grid_t, merged_h, merged_w, merge_h, merge_w, channel, temporal,
   * patch_row, patch_col)} with {@code grid_t = 1}; the one image frame is duplicated into both
   * temporal slots, so the two slots of every patch hold identical pixels.
   *
   * @param image the decoded RGB input; null is rejected
   * @return the flattened patch tensor with its grid/order metadata
   * @throws IllegalArgumentException if {@code image} is null, or its dimensions fail the geometry
   *     checks (aspect guard, no-resize divisibility, raw-patch-row budget)
   */
  public Qwen3VlImageProcessorResult process(RgbImage image) {
    if (image == null) {
      throw new IllegalArgumentException("image must not be null");
    }
    Qwen3VlGeometryPlan plan = geometry(image.width(), image.height());
    RgbImage working =
        config.doResize()
            ? ImageTransforms.resize(
                image, plan.resizedWidth(), plan.resizedHeight(), config.resample())
            : image;
    ImageTensor norm =
        ImageTransforms.normalize(
            working, config.rescaleFactor(), config.imageMean(), config.imageStd());
    // valuesRaw(): the normalized buffer is only read here; the flattened output buffer is
    // allocated once and handed to the result constructor by ownership, with no copy.
    float[] normRaw = norm.valuesRaw();
    int ps = config.patchSize();
    int tps = config.temporalPatchSize();
    int merge = config.mergeSize();
    int mergedH = plan.gridH() / merge;
    int mergedW = plan.gridW() / merge;
    int resizedWidth = plan.resizedWidth();
    // The reference's flatten_patches has grid_t * grid_h * grid_w rows (grid_t = 1 for a still
    // image): the post-merge token count is smaller, and the merge happens in the vision tower.
    // A builder budget far above the default can make the product exceed an int even though each
    // factor is individually valid, so the allocation size is checked in long.
    long total = (long) plan.rawPatchRows() * plan.patchStride();
    if (total > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "raw patch rows "
              + plan.rawPatchRows()
              + " times patch stride "
              + plan.patchStride()
              + " overflows a signed 32-bit float count: "
              + total);
    }
    float[] out = new float[(int) total];
    for (int mh = 0; mh < mergedH; mh++) {
      for (int mw = 0; mw < mergedW; mw++) {
        for (int imh = 0; imh < merge; imh++) {
          for (int imw = 0; imw < merge; imw++) {
            for (int c = 0; c < 3; c++) {
              for (int t = 0; t < tps; t++) {
                for (int pr = 0; pr < ps; pr++) {
                  int y = (mh * merge + imh) * ps + pr;
                  for (int pc = 0; pc < ps; pc++) {
                    int x = (mw * merge + imw) * ps + pc;
                    // C-order index of the post-transpose axis order
                    // (grid_t, mergedH, mergedW, merge_h, merge_w, c, tps, pr, pc) with grid_t = 1;
                    // both temporal slots t read the same (identically duplicated) frame.
                    int idx =
                        (((((mh * mergedW + mw) * merge + imh) * merge + imw) * 3 + c) * tps + t)
                                * ps
                            + pr;
                    idx = idx * ps + pc;
                    out[idx] = normRaw[(y * resizedWidth + x) * 3 + c];
                  }
                }
              }
            }
          }
        }
      }
    }
    return new Qwen3VlImageProcessorResult(plan, out);
  }

  /**
   * Exact port of the pinned 4.57.6 slow-processor {@code smart_resize(height, width, factor,
   * min_pixels, max_pixels)} (findings §4.5, transcribed verbatim there). All area products stay in
   * exact {@code long} until the binary64 conversion; {@link Math#rint} is Python's {@code round}
   * (banker's rounding), and the max-pixels branch keeps the reference's two sequential divisions
   * {@code height / beta / factor} — the left-associative order is load-bearing and must not be
   * folded into {@code height / (beta * factor)}, which can differ by a factor step.
   *
   * @param height input height in pixels
   * @param width input width in pixels
   * @param factor the resize multiple ({@code patch_size * merge_size})
   * @param minPixels lower bound on the resized pixel count
   * @param maxPixels upper bound on the resized pixel count
   * @return {@code [resizedHeight, resizedWidth]}, both multiples of {@code factor}
   * @throws IllegalArgumentException if the absolute aspect ratio is not at most 200
   */
  static int[] smartResize(int height, int width, int factor, int minPixels, int maxPixels) {
    double aspect = (double) Math.max(height, width) / Math.min(height, width);
    if (aspect > 200.0) {
      throw new IllegalArgumentException(
          "absolute aspect ratio must be smaller than 200, got " + aspect);
    }
    long hbar = (long) (Math.rint(height / (double) factor) * factor);
    long wbar = (long) (Math.rint(width / (double) factor) * factor);
    long area = (long) height * width;
    if (hbar * wbar > maxPixels) {
      double beta = Math.sqrt((double) area / (double) maxPixels);
      // Two sequential divisions, exactly as the reference (see the method Javadoc); the
      // floor/ceil results are integral doubles, so the long casts are lossless.
      hbar = Math.max(factor, (long) (Math.floor(height / (double) beta / factor) * factor));
      wbar = Math.max(factor, (long) (Math.floor(width / (double) beta / factor) * factor));
    } else if (hbar * wbar < minPixels) {
      double beta = Math.sqrt((double) minPixels / (double) area);
      hbar = (long) (Math.ceil(height * (double) beta / factor) * factor);
      wbar = (long) (Math.ceil(width * (double) beta / factor) * factor);
    }
    return new int[] {(int) hbar, (int) wbar};
  }
}
