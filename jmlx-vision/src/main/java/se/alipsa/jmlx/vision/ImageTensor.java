package se.alipsa.jmlx.vision;

import java.util.Arrays;

/**
 * Immutable floating-point image tensor with an explicit shape and {@link Layout}.
 *
 * <p>The record surface ({@code values}, {@code shape}, {@code layout}) is retained, but the
 * accessors return defensive copies and the component arrays are copied on construction, so no
 * backing array is ever leaked. {@code equals}/{@code hashCode} compare full content; {@code
 * toString} prints only a bounded shape/layout/element-count summary, never the data.
 *
 * <p>{@link #copyValuesTo(float[], int)} provides cross-module bulk access without allocating a
 * fresh multi-megabyte buffer on every accessor call.
 */
public record ImageTensor(float[] values, int[] shape, Layout layout) {
  /**
   * Validates layout, shape and element count, then copies both arrays.
   *
   * @throws IllegalArgumentException if any argument is null, a shape dimension is not positive,
   *     the shape does not match the layout's contract (HWC: {@code [h, w, 3]}; NHWC: {@code [1, h,
   *     w, 3]}), the element count overflows, or the values length differs from it
   */
  public ImageTensor {
    if (values == null) {
      throw new IllegalArgumentException("values must not be null");
    }
    if (shape == null) {
      throw new IllegalArgumentException("shape must not be null");
    }
    if (layout == null) {
      throw new IllegalArgumentException("layout must not be null");
    }
    if (layout == Layout.HWC && (shape.length != 3 || shape[2] != 3)) {
      throw new IllegalArgumentException(
          "HWC shape must be [height, width, 3] but was " + Arrays.toString(shape));
    }
    if (layout == Layout.NHWC && (shape.length != 4 || shape[0] != 1 || shape[3] != 3)) {
      throw new IllegalArgumentException(
          "NHWC shape must be [1, height, width, 3] but was " + Arrays.toString(shape));
    }
    long count = 1;
    for (int dim : shape) {
      if (dim <= 0) {
        throw new IllegalArgumentException(
            "shape dimensions must be positive: " + Arrays.toString(shape));
      }
      count = Math.multiplyExact(count, dim);
    }
    if (count != values.length) {
      throw new IllegalArgumentException(
          "values length must equal the product of shape=" + count + " but was " + values.length);
    }
    values = values.clone();
    shape = shape.clone();
  }

  /** Returns a copy of the tensor values in layout order. */
  @Override
  public float[] values() {
    return values.clone();
  }

  /** Returns a copy of the shape vector. */
  @Override
  public int[] shape() {
    return shape.clone();
  }

  /**
   * Bulk-copies the tensor values into {@code destination} without allocating a new buffer.
   *
   * @param destination destination array; must have room for {@code values.length} floats at {@code
   *     offset}
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

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof ImageTensor that)) {
      return false;
    }
    return layout == that.layout
        && Arrays.equals(shape, that.shape)
        && Arrays.equals(values, that.values);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(shape);
    result = 31 * result + layout.hashCode();
    return 31 * result + Arrays.hashCode(values);
  }

  @Override
  public String toString() {
    return "ImageTensor{layout="
        + layout
        + ", shape="
        + Arrays.toString(shape)
        + ", elements="
        + values.length
        + '}';
  }
}
