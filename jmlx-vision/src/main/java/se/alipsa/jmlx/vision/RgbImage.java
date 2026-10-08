package se.alipsa.jmlx.vision;

import java.util.Arrays;

/**
 * Immutable interleaved unsigned-RGB image, row-major, {@code width * height * 3} bytes.
 *
 * <p>Construction and access copy defensively, so a caller can neither observe the backing buffer
 * nor mutate the image through it. Content equality and hashing compare all pixel bytes.
 */
public final class RgbImage {
  private final int width;
  private final int height;
  private final byte[] pixels;

  /**
   * Creates a new image from the given pixel buffer.
   *
   * @param width positive image width in pixels
   * @param height positive image height in pixels
   * @param pixels exactly {@code width * height * 3} bytes of interleaved unsigned RGB, row-major
   * @throws IllegalArgumentException if a dimension is not positive, the pixel count overflows, or
   *     {@code pixels} is null or the wrong length
   */
  public RgbImage(int width, int height, byte[] pixels) {
    this(width, height, pixels, false);
  }

  private RgbImage(int width, int height, byte[] pixels, boolean owned) {
    if (width <= 0) {
      throw new IllegalArgumentException("width must be positive: " + width);
    }
    if (height <= 0) {
      throw new IllegalArgumentException("height must be positive: " + height);
    }
    if (pixels == null) {
      throw new IllegalArgumentException("pixels must not be null");
    }
    long required = Math.multiplyExact(Math.multiplyExact((long) width, height), 3L);
    if (required > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "pixel buffer length " + required + " overflows an int for " + width + "x" + height);
    }
    if (pixels.length != required) {
      throw new IllegalArgumentException(
          "pixels length must be exactly width*height*3=" + required + " but was " + pixels.length);
    }
    this.width = width;
    this.height = height;
    this.pixels = owned ? pixels : pixels.clone();
  }

  /**
   * Creates a new image from a buffer the caller owns and will no longer touch, adopting it without
   * the defensive copy. Package-private: for the module's own call chain, which passes freshly
   * allocated buffers it never retains or mutates.
   *
   * @param width positive image width in pixels
   * @param height positive image height in pixels
   * @param pixels exactly {@code width * height * 3} bytes, not shared with the caller
   * @throws IllegalArgumentException on invalid dimensions or a wrong buffer length
   */
  static RgbImage ofUnchecked(int width, int height, byte[] pixels) {
    return new RgbImage(width, height, pixels, true);
  }

  /** The image width in pixels. */
  public int width() {
    return width;
  }

  /** The image height in pixels. */
  public int height() {
    return height;
  }

  /** A copy of the interleaved RGB pixel buffer ({@code width * height * 3} bytes). */
  public byte[] pixels() {
    return pixels.clone();
  }

  /**
   * The backing pixel buffer without copying. Package-private escape hatch for the module's own
   * call chain, which hands the buffer to transforms that read it and never mutate it; public code
   * must use {@link #pixels()} or {@link #copyPixelsTo(byte[], int)}.
   */
  byte[] pixelsRaw() {
    return pixels;
  }

  /**
   * Bulk-copies the full pixel buffer into {@code destination} without allocating a new buffer.
   *
   * @param destination destination array; must have room for {@code width * height * 3} bytes at
   *     {@code offset}
   * @param offset starting index in {@code destination}
   * @throws IllegalArgumentException if {@code destination} is null
   * @throws IndexOutOfBoundsException if {@code offset} is negative or exceeds the remaining
   *     destination capacity
   */
  public void copyPixelsTo(byte[] destination, int offset) {
    if (destination == null) {
      throw new IllegalArgumentException("destination must not be null");
    }
    if (offset < 0 || offset > destination.length - pixels.length) {
      throw new IndexOutOfBoundsException(
          "offset "
              + offset
              + " cannot copy "
              + pixels.length
              + " bytes into a destination of length "
              + destination.length);
    }
    System.arraycopy(pixels, 0, destination, offset, pixels.length);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof RgbImage that)) {
      return false;
    }
    return width == that.width && height == that.height && Arrays.equals(pixels, that.pixels);
  }

  @Override
  public int hashCode() {
    int result = 31 * width + height;
    return 31 * result + Arrays.hashCode(pixels);
  }

  @Override
  public String toString() {
    return "RgbImage{" + width + "x" + height + '}';
  }
}
