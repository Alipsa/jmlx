package se.alipsa.jmlx.vision;

/**
 * The pure geometry of the SmolVLM/Idefics3 preprocessing chain for one input image: the stage-1
 * resize target, the stage-2 resize target and the row/column split grid, computed without touching
 * any pixels.
 *
 * <p>Grid semantics mirror the reference: {@code (0, 0)} means the image is not split (a single
 * tile — the stage-2 image itself when splitting is enabled, or one square resize when it is
 * disabled, in which case the stage-2 size is {@code 0x0}); otherwise both grid dimensions are
 * positive and the image splits into {@code rows x cols} crops plus a global thumbnail.
 *
 * <p>{@code SmolVlmImageProcessor.geometry} computes this plan, and the processor's pixel chain
 * ({@code intermediates}) uses that same plan for its resize targets and split grid, so a caller
 * that plans geometry from an image's dimensions (for example to lay out the prompt-side tile
 * markers) cannot drift from what the processor actually produces.
 *
 * @param stage1Height stage-1 target height in pixels
 * @param stage1Width stage-1 target width in pixels
 * @param stage2Height stage-2 target height in pixels, or 0 when splitting is disabled
 * @param stage2Width stage-2 target width in pixels, or 0 when splitting is disabled
 * @param rows number of split rows, or 0 when the image is not split
 * @param cols number of split columns, or 0 when the image is not split
 */
public record SmolVlmGeometryPlan(
    int stage1Height, int stage1Width, int stage2Height, int stage2Width, int rows, int cols) {

  public SmolVlmGeometryPlan {
    if (stage1Height <= 0 || stage1Width <= 0) {
      throw new IllegalArgumentException(
          "stage 1 size must be positive: " + stage1Height + "x" + stage1Width);
    }
    if (stage2Height < 0 || stage2Width < 0) {
      throw new IllegalArgumentException(
          "stage 2 size must not be negative: " + stage2Height + "x" + stage2Width);
    }
    if (rows < 0 || cols < 0) {
      throw new IllegalArgumentException("split grid must not be negative: " + rows + "x" + cols);
    }
    if ((rows == 0) != (cols == 0)) {
      throw new IllegalArgumentException(
          "split grid must be (0, 0) or have both rows and cols positive: " + rows + "x" + cols);
    }
    if (rows > 0 && (stage2Height <= 0 || stage2Width <= 0)) {
      throw new IllegalArgumentException(
          "a split grid requires a positive stage 2 size: " + stage2Height + "x" + stage2Width);
    }
  }

  /** True when the image splits into row/col crops plus a global thumbnail. */
  public boolean split() {
    return rows > 0;
  }

  /**
   * Number of tiles the processor produces: {@code rows x cols} crops plus the global thumbnail; 1
   * when the image is not split.
   *
   * @throws ArithmeticException if the tile count overflows a signed 32-bit integer
   */
  public int tileCount() {
    return Math.addExact(Math.multiplyExact(rows, cols), 1);
  }
}
