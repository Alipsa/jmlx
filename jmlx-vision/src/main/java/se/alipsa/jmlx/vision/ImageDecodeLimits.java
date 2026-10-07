package se.alipsa.jmlx.vision;

/**
 * Decode-size bounds checked from decoded metadata before any pixel allocation.
 *
 * <p>Both fields must be positive. The module default ({@link
 * ImageDecoder#decode(java.nio.file.Path)}) uses {@code ImageDecodeLimits(16_777_216L, 16_384)}: a
 * conservative ceiling that rejects typical 24 MP and 48 MP phone photos. Callers that need to
 * accept such images with adequate memory use the explicit {@code ImageDecoder.decode(..., limits)}
 * overloads.
 */
public record ImageDecodeLimits(long maxPixels, int maxDimension) {
  /**
   * Validates that both bounds are positive.
   *
   * @throws IllegalArgumentException if {@code maxPixels} or {@code maxDimension} is not positive
   */
  public ImageDecodeLimits {
    if (maxPixels <= 0) {
      throw new IllegalArgumentException("maxPixels must be positive: " + maxPixels);
    }
    if (maxDimension <= 0) {
      throw new IllegalArgumentException("maxDimension must be positive: " + maxDimension);
    }
  }
}
