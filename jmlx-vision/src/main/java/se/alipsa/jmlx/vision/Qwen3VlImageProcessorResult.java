package se.alipsa.jmlx.vision;

/**
 * Immutable result of {@link Qwen3VlImageProcessor#process}: the reference's flattened merge-block
 * patch tensor with its grid/order metadata.
 *
 * <p>{@link #values()} is the reference's {@code pixel_values} flat tensor: {@link #rawPatchRows()}
 * rows (the raw 16x16 patch count, {@code grid_h * grid_w}) of {@link #patchStride()} floats each,
 * in the merge-block axis order {@code mergedH, mergedW, merge_h, merge_w, channel, temporal,
 * patch_row, patch_col}. The post-merge token count the vision tower will produce for this image is
 * {@link #mergedRowCount()}; the merge itself happens in the model, not here.
 *
 * <p>The grid and values are defensive: mutation by a caller is never observable, and the values
 * array is copied both on construction and access.
 */
public final class Qwen3VlImageProcessorResult {
  private final int[] gridThw;
  private final int rawPatchRows;
  private final int mergedRowCount;
  private final int patchStride;
  private final float[] values;

  Qwen3VlImageProcessorResult(Qwen3VlGeometryPlan plan, float[] values) {
    if (plan == null) {
      throw new IllegalArgumentException("plan must not be null");
    }
    if (values == null) {
      throw new IllegalArgumentException("values must not be null");
    }
    long expected = (long) plan.rawPatchRows() * plan.patchStride();
    if (expected > Integer.MAX_VALUE || expected != values.length) {
      throw new IllegalArgumentException(
          "values length must equal rawPatchRows * patchStride="
              + expected
              + " but was "
              + values.length);
    }
    this.gridThw = new int[] {1, plan.gridH(), plan.gridW()};
    this.rawPatchRows = plan.rawPatchRows();
    this.mergedRowCount = plan.mergedRowCount();
    this.patchStride = plan.patchStride();
    this.values = values.clone();
  }

  /**
   * A copy of the image grid {@code [grid_t, grid_h, grid_w]} (grid_t is always 1 for a still
   * image).
   */
  public int[] gridThw() {
    return gridThw.clone();
  }

  /** The raw 16x16 patch count, {@code grid_h * grid_w}: the row count of {@link #values()}. */
  public int rawPatchRows() {
    return rawPatchRows;
  }

  /**
   * The post-merge token count, {@code rawPatchRows / (merge_size * merge_size)} — what the vision
   * tower's merge produces for this image.
   */
  public int mergedRowCount() {
    return mergedRowCount;
  }

  /**
   * The width of each flattened-patch row, {@code 3 * temporal_patch_size * patch_size *
   * patch_size}.
   */
  public int patchStride() {
    return patchStride;
  }

  /** Returns a copy of the flattened patch tensor values. */
  public float[] values() {
    return values.clone();
  }

  /**
   * Bulk-copies the flattened patch tensor values into {@code destination} without allocating a new
   * buffer.
   *
   * @param destination destination array; must have room for all the values at {@code offset}
   * @param offset starting index in {@code destination}
   * @throws IllegalArgumentException if {@code destination} is null
   * @throws IndexOutOfBoundsException if {@code offset} is negative or exceeds the remaining
   *     destination capacity
   */
  public void copyValuesTo(float[] destination, int offset) {
    if (destination == null) {
      throw new IllegalArgumentException("destination must not be null");
    }
    if (offset < 0 || offset > destination.length - values.length) {
      throw new IndexOutOfBoundsException(
          "offset "
              + offset
              + " cannot copy "
              + values.length
              + " floats into a destination of length "
              + destination.length);
    }
    System.arraycopy(values, 0, destination, offset, values.length);
  }
}
