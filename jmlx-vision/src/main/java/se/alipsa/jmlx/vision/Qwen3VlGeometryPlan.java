package se.alipsa.jmlx.vision;

/**
 * The pure geometry of the Qwen3-VL preprocessing chain for one input image: the original
 * dimensions, the {@code smart_resize} target (or the input dimensions unchanged when resizing is
 * disabled), the patch grid and the merge-block row count, computed without touching any pixels.
 *
 * <p>Grid semantics mirror the reference {@code image_grid_thw} of the pinned slow processor:
 * {@code gridT} is always 1 for a still image (the one image frame is duplicated to {@code
 * temporal_patch_size} identical frames inside the pixel path), and {@code gridH}/{@code gridW} are
 * the resized dimensions divided by {@code patch_size}. The flattened patch tensor holds {@code
 * rawPatchRows} rows of {@code patchStride} ({@code 3 * temporal_patch_size * patch_size *
 * patch_size}) floats each; {@code mergedRowCount} is the post-merge token count the vision tower's
 * merge produces for the image ({@code rawPatchRows / (merge_size * merge_size)}).
 *
 * <p>{@link Qwen3VlImageProcessor#geometry(int, int)} computes this plan, and the processor's pixel
 * path ({@code process}) uses that same plan for its resize target and patch grid, so a caller that
 * plans geometry from an image's dimensions alone (for example the zero-token planning path of the
 * model layer) cannot drift from what the processor actually produces.
 *
 * @param inputWidth original image width in pixels
 * @param inputHeight original image height in pixels
 * @param resizedWidth resize target width in pixels (equals {@code inputWidth} when resizing is
 *     disabled)
 * @param resizedHeight resize target height in pixels (equals {@code inputHeight} when resizing is
 *     disabled)
 * @param gridT temporal grid dimension; always 1 for a still image
 * @param gridH resized height divided by {@code patch_size}
 * @param gridW resized width divided by {@code patch_size}
 * @param rawPatchRows {@code gridH * gridW}, the raw 16x16 patch rows of the image
 * @param mergedRowCount {@code rawPatchRows / (merge_size * merge_size)}, the post-merge token
 *     count the vision tower produces
 * @param patchStride {@code 3 * temporal_patch_size * patch_size * patch_size}, the width of each
 *     flattened-patch row
 */
public record Qwen3VlGeometryPlan(
    int inputWidth,
    int inputHeight,
    int resizedWidth,
    int resizedHeight,
    int gridT,
    int gridH,
    int gridW,
    int rawPatchRows,
    int mergedRowCount,
    int patchStride) {

  /** Validates that the component values are mutually consistent (see the record Javadoc). */
  public Qwen3VlGeometryPlan {
    if (inputWidth <= 0 || inputHeight <= 0) {
      throw new IllegalArgumentException(
          "input dimensions must be positive: " + inputWidth + "x" + inputHeight);
    }
    if (resizedWidth <= 0 || resizedHeight <= 0) {
      throw new IllegalArgumentException(
          "resized dimensions must be positive: " + resizedWidth + "x" + resizedHeight);
    }
    if (gridT != 1) {
      throw new IllegalArgumentException("gridT must be 1 for a still image: " + gridT);
    }
    if (gridH <= 0 || gridW <= 0) {
      throw new IllegalArgumentException("grid must be positive: " + gridH + "x" + gridW);
    }
    if ((long) gridH * gridW != rawPatchRows) {
      throw new IllegalArgumentException(
          "rawPatchRows must equal gridH * gridW (" + gridH + " * " + gridW + "): " + rawPatchRows);
    }
    if (mergedRowCount <= 0) {
      throw new IllegalArgumentException("mergedRowCount must be positive: " + mergedRowCount);
    }
    long perMergeBlock = rawPatchRows / (long) mergedRowCount;
    if (rawPatchRows % (long) mergedRowCount != 0 || perMergeBlock < 1) {
      throw new IllegalArgumentException(
          "rawPatchRows must be a positive multiple of mergedRowCount: "
              + rawPatchRows
              + " / "
              + mergedRowCount);
    }
    long mergeSide = (long) Math.sqrt(perMergeBlock);
    if (mergeSide * mergeSide != perMergeBlock) {
      throw new IllegalArgumentException(
          "rawPatchRows / mergedRowCount must be merge_size^2 for some positive merge_size: "
              + perMergeBlock);
    }
    if (patchStride <= 0 || patchStride % 3 != 0) {
      throw new IllegalArgumentException(
          "patchStride must be a positive multiple of 3 (3 * temporal_patch_size * patch_size * "
              + "patch_size): "
              + patchStride);
    }
  }

  /** Diagnostic {@code "height x width"} string of the resize target. */
  public String resizedHeightWidth() {
    return resizedHeight + "x" + resizedWidth;
  }
}
