package se.alipsa.jmlx.vision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The SmolVLM/Idefics3 image preprocessing chain, ported from the pinned transformers 4.57.6 {@code
 * Idefics3ImageProcessor.preprocess} for a single decoded image:
 *
 * <ol>
 *   <li>optional first resize of the longest edge to {@code size.longest_edge} (2048), the short
 *       edge kept proportional and rounded to even, capped at 4096;
 *   <li>resize of both edges to multiples of {@code max_image_size.longest_edge} (512), then
 *       splitting into {@code rows x cols} crops plus a global thumbnail resized to {@code 512 x
 *       512} (or a single square resize to {@code 512 x 512} when splitting is disabled);
 *   <li>per-tile rescale ({@code float32(float64(pixel) * rescale_factor)}) and normalize ({@code
 *       (x - mean[c]) / std[c]} in float32);
 *   <li>optional padding of every tile to the largest tile size with constant {@code 0.0f} plus a
 *       per-tile valid-pixel mask.
 * </ol>
 *
 * <p>The input must already be an {@link RgbImage} (i.e. decoded and converted, with alpha
 * composited over white); the reference's {@code convert_to_rgb} step is therefore implicit in the
 * decode. All resizing uses the {@link SmolVlmProcessorConfig#resample() resampling kernel} from
 * the validated config through the bit-exact Pillow kernel in {@code ImageTransforms.resize}.
 */
public final class SmolVlmImageProcessor {
  // Absolute maximum side of the first resize stage, mirroring the reference's
  // MAX_IMAGE_SIZE = 4096.
  private static final int MAX_IMAGE_SIZE = 4096;

  private final SmolVlmProcessorConfig config;

  /**
   * Creates a processor with the given validated configuration.
   *
   * @param config validated processor configuration; null is rejected
   * @throws IllegalArgumentException if {@code config} is null
   */
  public SmolVlmImageProcessor(SmolVlmProcessorConfig config) {
    if (config == null) {
      throw new IllegalArgumentException("config must not be null");
    }
    this.config = config;
  }

  /**
   * Runs the full preprocessing chain on one image.
   *
   * @param image the decoded RGB input; null is rejected
   * @return the normalized, padded tiles with masks and grid/order metadata
   * @throws IllegalArgumentException if {@code image} is null
   */
  public SmolVlmImageProcessorResult process(RgbImage image) {
    if (image == null) {
      throw new IllegalArgumentException("image must not be null");
    }
    ChainIntermediates chain = intermediates(image);

    double scale = config.doRescale() ? config.rescaleFactor() : 1.0;
    float[] mean = config.doNormalize() ? config.imageMean() : new float[] {0.0f, 0.0f, 0.0f};
    float[] std = config.doNormalize() ? config.imageStd() : new float[] {1.0f, 1.0f, 1.0f};

    List<ImageTensor> tensors = new ArrayList<>(chain.frames().size());
    for (RgbImage frame : chain.frames()) {
      tensors.add(ImageTransforms.normalize(frame, scale, mean, std));
    }

    if (config.doPad()) {
      int maxH = 0;
      int maxW = 0;
      for (ImageTensor tensor : tensors) {
        int[] shape = tensor.shape();
        maxH = Math.max(maxH, shape[1]);
        maxW = Math.max(maxW, shape[2]);
      }
      List<ImageTensor> padded = new ArrayList<>(tensors.size());
      List<int[]> masks = new ArrayList<>(tensors.size());
      for (ImageTensor tensor : tensors) {
        int[] shape = tensor.shape();
        int h = shape[1];
        int w = shape[2];
        if (h == maxH && w == maxW) {
          padded.add(tensor);
        } else {
          padded.add(padTensor(tensor, maxH, maxW));
        }
        masks.add(mask(h, w, maxH, maxW));
      }
      return new SmolVlmImageProcessorResult(padded, masks, chain.rows(), chain.cols());
    }

    List<int[]> masks = new ArrayList<>(tensors.size());
    for (ImageTensor tensor : tensors) {
      int[] shape = tensor.shape();
      masks.add(fullMask(shape[1], shape[2]));
    }
    return new SmolVlmImageProcessorResult(tensors, masks, chain.rows(), chain.cols());
  }

  /**
   * Visible for testing: the uint8 intermediates of the preprocessing chain — the stage-1 and
   * stage-2 resizes (stage 2 is absent, i.e. null, when splitting is disabled, mirroring the
   * reference's no-split path) and the pre-rescale frames in tile order (crops, global last; the
   * single square-resized frame when splitting is disabled).
   */
  record ChainIntermediates(
      RgbImage stage1, RgbImage stage2, List<RgbImage> frames, int rows, int cols) {}

  /**
   * The pure geometry of the preprocessing chain for an input of the given dimensions: the stage-1
   * and stage-2 resize targets and the split grid, without touching any pixels. The pixel chain
   * ({@code intermediates}) uses this plan for its resize targets and split grid, so a caller that
   * plans geometry from dimensions alone (for example to lay out the prompt-side tile markers)
   * cannot drift from what the processor actually produces.
   *
   * @param height input image height in pixels
   * @param width input image width in pixels
   * @return the geometry plan
   * @throws IllegalArgumentException if the dimensions are not positive
   */
  public SmolVlmGeometryPlan geometry(int height, int width) {
    if (height <= 0 || width <= 0) {
      throw new IllegalArgumentException(
          "image dimensions must be positive: " + width + "x" + height);
    }
    int s1h = height;
    int s1w = width;
    if (config.doResize()) {
      int[] size = stage1Size(height, width, config.sizeLongestEdge());
      s1h = size[0];
      s1w = size[1];
    }
    int s2h = 0;
    int s2w = 0;
    int rows = 0;
    int cols = 0;
    if (config.doImageSplitting()) {
      int m = config.maxImageSizeLongestEdge();
      int[] size = stage2Size(s1h, s1w, m);
      s2h = size[0];
      s2w = size[1];
      if (s2h > m || s2w > m) {
        rows = (int) Math.ceil((double) s2h / m);
        cols = (int) Math.ceil((double) s2w / m);
      }
    }
    return new SmolVlmGeometryPlan(s1h, s1w, s2h, s2w, rows, cols);
  }

  ChainIntermediates intermediates(RgbImage image) {
    SmolVlmGeometryPlan plan = geometry(image.height(), image.width());
    RgbImage current = image;
    if (config.doResize()) {
      current =
          ImageTransforms.resize(
              current, plan.stage1Width(), plan.stage1Height(), config.resample());
    }
    RgbImage stage1 = current;
    if (config.doImageSplitting()) {
      RgbImage stage2 =
          ImageTransforms.resize(
              stage1, plan.stage2Width(), plan.stage2Height(), config.resample());
      SplitResult split = split(stage2, config.maxImageSizeLongestEdge());
      if (split.rows != plan.rows() || split.cols != plan.cols()) {
        // Unreachable: both compute the same ceil(size / tile) from the same stage-2 size. The
        // guard makes any future divergence between the pure plan and the pixel chain a crash,
        // not a prompt/pixel mismatch.
        throw new IllegalStateException(
            "split grid drifted from the geometry plan: planned "
                + plan.rows()
                + "x"
                + plan.cols()
                + ", produced "
                + split.rows
                + "x"
                + split.cols);
      }
      return new ChainIntermediates(stage1, stage2, split.frames, plan.rows(), plan.cols());
    }
    int m = config.maxImageSizeLongestEdge();
    return new ChainIntermediates(
        stage1, null, List.of(ImageTransforms.resize(stage1, m, m, config.resample())), 0, 0);
  }

  /**
   * Stage-1 target size: longest edge to {@code longestEdge}, short edge proportional and
   * even-rounded, then capped at {@code MAX_IMAGE_SIZE}. Mirrors the reference's {@code
   * _resize_output_size_rescale_to_max_len} + {@code _resize_output_size_scale_below_ upper_bound}.
   */
  private static int[] stage1Size(int height, int width, int longestEdge) {
    double aspect = (double) width / height;
    int h = height;
    int w = width;
    if (w >= h) {
      w = longestEdge;
      h = (int) (w / aspect);
      if (h % 2 != 0) {
        h += 1;
      }
    } else {
      h = longestEdge;
      w = (int) (h * aspect);
      if (w % 2 != 0) {
        w += 1;
      }
    }
    h = Math.max(h, 1);
    w = Math.max(w, 1);

    double cappedAspect = (double) w / h;
    if (w >= h && w > MAX_IMAGE_SIZE) {
      w = MAX_IMAGE_SIZE;
      h = (int) (w / cappedAspect);
    } else if (h > w && h > MAX_IMAGE_SIZE) {
      h = MAX_IMAGE_SIZE;
      w = (int) (h * cappedAspect);
    }
    h = Math.max(h, 1);
    w = Math.max(w, 1);
    return new int[] {h, w};
  }

  /**
   * Stage-2 target size: both edges rounded up to multiples of {@code max}, mirroring the
   * reference's {@code resize_for_vision_encoder}.
   */
  private static int[] stage2Size(int height, int width, int max) {
    double aspect = (double) width / height;
    int h = height;
    int w = width;
    if (w >= h) {
      w = (int) (Math.ceil((double) w / max) * max);
      h = (int) ((double) w / aspect);
      h = (int) (Math.ceil((double) h / max) * max);
    } else {
      h = (int) (Math.ceil((double) h / max) * max);
      w = (int) ((double) h * aspect);
      w = (int) (Math.ceil((double) w / max) * max);
    }
    return new int[] {h, w};
  }

  /** Mirrors the reference's {@code split_image}: row-major crops, global thumbnail last. */
  private SplitResult split(RgbImage image, int max) {
    int h = image.height();
    int w = image.width();
    List<RgbImage> frames = new ArrayList<>();
    int rows;
    int cols;
    if (h > max || w > max) {
      rows = (int) Math.ceil((double) h / max);
      cols = (int) Math.ceil((double) w / max);
      int optimalH = (int) Math.ceil((double) h / rows);
      int optimalW = (int) Math.ceil((double) w / cols);
      // One exposure of the stage-2 buffer for the whole crop set: pixelsRaw() does not copy,
      // so the cost is one buffer reference, not one full-image clone per tile.
      byte[] src = image.pixelsRaw();
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          int startX = c * optimalW;
          int startY = r * optimalH;
          int endX = Math.min(startX + optimalW, w);
          int endY = Math.min(startY + optimalH, h);
          frames.add(crop(src, w, startX, startY, endX, endY));
        }
      }
      if (h != max || w != max) {
        frames.add(ImageTransforms.resize(image, max, max, config.resample()));
      } else {
        frames.add(image);
      }
    } else {
      rows = 0;
      cols = 0;
      frames.add(image);
    }
    return new SplitResult(frames, rows, cols);
  }

  private static final class SplitResult {
    final List<RgbImage> frames;
    final int rows;
    final int cols;

    SplitResult(List<RgbImage> frames, int rows, int cols) {
      this.frames = frames;
      this.rows = rows;
      this.cols = cols;
    }
  }

  /**
   * Copies the {@code (x0, y0) x (x1, y1)} rectangle out of the source buffer into a fresh tile.
   * The source buffer is only read; the freshly allocated tile is adopted into the result without a
   * further copy — {@link RgbImage#ofUnchecked} does not clone, and the caller exposes each tile
   * once for the whole crop set (see {@link #split}).
   */
  private static RgbImage crop(byte[] src, int srcWidth, int x0, int y0, int x1, int y1) {
    int w = x1 - x0;
    int h = y1 - y0;
    byte[] out = new byte[w * h * 3];
    for (int y = 0; y < h; y++) {
      System.arraycopy(src, (y0 + y) * srcWidth * 3 + x0 * 3, out, y * w * 3, w * 3);
    }
    return RgbImage.ofUnchecked(w, h, out);
  }

  /** Pads an NHWC tensor to (maxH, maxW) at bottom/right with constant 0.0f, as the reference. */
  private static ImageTensor padTensor(ImageTensor tensor, int maxH, int maxW) {
    int[] shape = tensor.shape();
    int h = shape[1];
    int w = shape[2];
    // valuesRaw(): the (multi-megabyte) source buffer is only read here; the padded buffer is
    // fresh and is copied exactly once, by the ImageTensor constructor.
    float[] source = tensor.valuesRaw();
    float[] padded = new float[maxH * maxW * 3];
    for (int y = 0; y < h; y++) {
      System.arraycopy(source, y * w * 3, padded, y * maxW * 3, w * 3);
    }
    return new ImageTensor(padded, new int[] {1, maxH, maxW, 3}, Layout.NHWC);
  }

  private static int[] mask(int h, int w, int maxH, int maxW) {
    int[] mask = new int[maxH * maxW];
    for (int y = 0; y < h; y++) {
      Arrays.fill(mask, y * maxW, y * maxW + w, 1);
    }
    return mask;
  }

  private static int[] fullMask(int h, int w) {
    int[] mask = new int[h * w];
    Arrays.fill(mask, 1);
    return mask;
  }
}
