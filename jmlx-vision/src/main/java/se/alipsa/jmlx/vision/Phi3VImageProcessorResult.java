package se.alipsa.jmlx.vision;

/**
 * Immutable result of {@link Phi3VImageProcessor#process}: the reference's one {@code pixel_values}
 * batch row with its geometry plan.
 *
 * <p>{@link #values()} is the reference's batch row flat tensor: {@link #plan()} {@code
 * slotCount()} slots of {@code 3 * 336 * 336} floats each, slot-major then CHW ({@code
 * [slot][channel][row][ column]}): slot 0 is the global bicubic 336x336 image, slots 1..cropBlocks
 * the h-major 336x336 sub-crops, and any remaining slots all-zero padding ({@code
 * pad_to_max_num_crops_tensor}, which zero-pads only, never truncates). With the reference's scale
 * loop {@code cropBlocks <= num_crops}, so with the pinned artifact every row has exactly 5 slots.
 * The reference's per-image {@code image_sizes} entry (padded {@code [height, width]}) and {@code
 * num_img_tokens} are on the plan.
 *
 * <p>The values are defensive on read: mutation by a caller is never observable, and the values
 * array is copied on every access. The constructor takes its array by ownership, without a
 * defensive copy; its sole caller, {@link Phi3VImageProcessor#process(RgbImage)}, allocates it
 * fresh and must not retain or mutate it after handing it over.
 */
public final class Phi3VImageProcessorResult {
  private final Phi3VGeometryPlan plan;
  private final float[] values;

  /**
   * Takes ownership of {@code values}, which is retained as-is without a defensive copy: the caller
   * must allocate it fresh for this result and must not retain or mutate it afterwards.
   *
   * @param plan the validated geometry plan; null is rejected
   * @param values the finished slot-major batch row, exactly {@code slotCount() * 3 * 336 * 336}
   *     floats; null is rejected
   * @throws IllegalArgumentException if either argument is null or the values length does not equal
   *     {@code slotCount() * 3 * 336 * 336}
   */
  Phi3VImageProcessorResult(Phi3VGeometryPlan plan, float[] values) {
    if (plan == null) {
      throw new IllegalArgumentException("plan must not be null");
    }
    if (values == null) {
      throw new IllegalArgumentException("values must not be null");
    }
    long expected = (long) plan.slotCount() * 3 * 336 * 336;
    if (expected > Integer.MAX_VALUE || expected != values.length) {
      throw new IllegalArgumentException(
          "values length must equal slotCount * 3 * 336 * 336="
              + expected
              + " but was "
              + values.length);
    }
    this.plan = plan;
    this.values = values;
  }

  /** The validated geometry plan (padded sizes, crop grid, slot count, per-image token count). */
  public Phi3VGeometryPlan plan() {
    return plan;
  }

  /** Returns a copy of the slot-major batch row values. */
  public float[] values() {
    return values.clone();
  }

  /**
   * Bulk-copies the batch row values into {@code destination} without allocating a new buffer.
   *
   * @param destination destination array; must have room for all the values at {@code offset}
   * @param offset starting index in {@code destination}
   * @throws IllegalArgumentException if {@code destination} is null
   * @throws IndexOutOfBoundsException if {@code offset} is negative or exceeds the remaining
   *     destination capacity
   */
  public void copyValuesTo(float[] destination, int offset) {
    copyValuesTo(destination, offset, 0, values.length);
  }

  /**
   * Copies a range of the slot-major row into {@code destination} without touching the rest of the
   * row: the caller that only consumes the image's real slots ({@code geometry.cropBlocks() + 1}
   * leading slots — the reference computes the zero-padded slots and slices their rows away) copies
   * only those, not the padded tail.
   *
   * @param destination destination array; must have room for {@code length} values at {@code
   *     destOffset}
   * @param destOffset starting index in {@code destination}
   * @param srcOffset starting slot-major float index in this row, in [0, slotCount * 3 * 336 * 336)
   * @param length number of floats to copy; the range must not run past the end of the row
   * @throws IllegalArgumentException if {@code destination} is null, or {@code srcOffset} or {@code
   *     length} is negative
   * @throws IndexOutOfBoundsException if either range exceeds its array
   */
  public void copyValuesTo(float[] destination, int destOffset, int srcOffset, int length) {
    if (destination == null) {
      throw new IllegalArgumentException("destination must not be null");
    }
    if (srcOffset < 0 || length < 0) {
      throw new IllegalArgumentException(
          "srcOffset " + srcOffset + " and length " + length + " must be non-negative");
    }
    if (destOffset < 0 || destOffset > destination.length - length) {
      throw new IndexOutOfBoundsException(
          "destOffset "
              + destOffset
              + " cannot copy "
              + length
              + " floats into a destination of length "
              + destination.length);
    }
    if (srcOffset > values.length - length) {
      throw new IndexOutOfBoundsException(
          "srcOffset "
              + srcOffset
              + " cannot copy "
              + length
              + " floats from a row of "
              + values.length
              + " floats");
    }
    System.arraycopy(values, srcOffset, destination, destOffset, length);
  }
}
