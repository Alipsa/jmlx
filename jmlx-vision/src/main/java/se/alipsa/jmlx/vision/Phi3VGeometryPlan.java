package se.alipsa.jmlx.vision;

/**
 * The pure geometry of the Phi-3.5-Vision high-definition (HD) preprocessing chain for one input
 * image: the resize and 336-multiple padding targets, the 336-block crop grid, the slot count and
 * the per-image image-token count, computed without touching any pixels.
 *
 * <p>The geometry mirrors the reference {@code HD_transform} of the pinned bundled processor
 * ({@code processing_phi3_v.py}, findings §6): when the input is portrait ({@code inputWidth <
 * inputHeight}) the working space is the transposed image (long side first); the scale-selection
 * loop {@code while scale * ceil(scale / ratio) <= num_crops} (in double, {@code ratio = long side
 * / short side}) picks the resize factor; {@code new_w = scale * 336} and {@code new_h = int(new_w
 * / ratio)} (double division, truncating); then the short side is padded to the next multiple of
 * 336 (the first {@code (tar - new_h) / 2} rows, white 255, remainder at the end). The record's
 * {@code resized*} and {@code padded*} dimensions are reported in the <b>original</b> orientation
 * (transposed back), which is what the pixel path's {@code image_sizes} and the token formula
 * consume: the resized long side is always {@code scale * 336}, and the unpadded side keeps its
 * resized size while the short side (the height for landscape, the width for portrait) grows to the
 * padded value.
 *
 * <p>With the pinned reference the scale loop's maximum scale is {@code num_crops} (reached only
 * when the aspect ratio is at least {@code num_crops}), so the resized long edge never exceeds
 * {@code num_crops * 336} and the padded grid never exceeds {@code num_crops} blocks — hence {@code
 * cropBlocks <= numCrops} and the slot count is always exactly {@code numCrops + 1} (global image
 * first, then the crops). The processor's {@code geometry} method rejects an aspect ratio above
 * {@code num_crops * 336}, where the double-truncating {@code new_h} would truncate to 0 and the
 * reference's PIL resize would fail (see the plan's WP2 extreme-ratio item).
 *
 * @param inputWidth original image width in pixels
 * @param inputHeight original image height in pixels
 * @param transposed whether the working space is the transposed image ({@code inputWidth <
 *     inputHeight})
 * @param resizedWidth HD-resize target width in pixels, original orientation
 * @param resizedHeight HD-resize target height in pixels, original orientation
 * @param paddedWidth width after 336-multiple padding, original orientation (a multiple of 336, at
 *     least {@code resizedWidth}; equals {@code resizedWidth} for landscape, since the reference
 *     never pads the long side)
 * @param paddedHeight height after 336-multiple padding, original orientation (a multiple of 336,
 *     at least {@code resizedHeight}; equals {@code resizedHeight} for portrait)
 * @param gridH {@code paddedHeight / 336}, the crop grid rows
 * @param gridW {@code paddedWidth / 336}, the crop grid columns
 * @param cropBlocks {@code gridH * gridW}, the number of 336x336 sub-crops (at most {@code
 *     numCrops})
 * @param numCrops the configuration's {@code num_crops} the scale loop planned with
 * @param slotCount {@code max(cropBlocks + 1, numCrops + 1)}: the zero-padded slot count of the
 *     output tensor (the global image plus the crops)
 * @param numImgTokens {@code (cropBlocks + 1) * 144 + 1 + (gridH + 1) * 12}, the reference's
 *     per-image image-token count (from the padded sizes)
 */
public record Phi3VGeometryPlan(
    int inputWidth,
    int inputHeight,
    boolean transposed,
    int resizedWidth,
    int resizedHeight,
    int paddedWidth,
    int paddedHeight,
    int gridH,
    int gridW,
    int cropBlocks,
    int numCrops,
    int slotCount,
    int numImgTokens) {

  /** Validates that the component values are mutually consistent (see the record Javadoc). */
  public Phi3VGeometryPlan {
    if (inputWidth <= 0 || inputHeight <= 0) {
      throw new IllegalArgumentException(
          "input dimensions must be positive: " + inputWidth + "x" + inputHeight);
    }
    if (transposed != (inputWidth < inputHeight)) {
      throw new IllegalArgumentException(
          "transposed must equal (inputWidth < inputHeight) for "
              + inputWidth
              + "x"
              + inputHeight
              + ": "
              + transposed);
    }
    if (resizedWidth <= 0 || resizedHeight <= 0) {
      throw new IllegalArgumentException(
          "resized dimensions must be positive: " + resizedWidth + "x" + resizedHeight);
    }
    if (paddedWidth <= 0 || paddedWidth % 336 != 0) {
      throw new IllegalArgumentException(
          "paddedWidth must be a positive multiple of 336: " + paddedWidth);
    }
    if (paddedHeight <= 0 || paddedHeight % 336 != 0) {
      throw new IllegalArgumentException(
          "paddedHeight must be a positive multiple of 336: " + paddedHeight);
    }
    if (paddedWidth < resizedWidth || paddedHeight < resizedHeight) {
      throw new IllegalArgumentException(
          "padded dimensions must be at least the resized dimensions: "
              + paddedWidth
              + "x"
              + paddedHeight
              + " < "
              + resizedWidth
              + "x"
              + resizedHeight);
    }
    // The reference pads the short side only, so the long side (scale * 336) keeps its resized
    // size: for landscape that is the width, for portrait the height.
    if (transposed) {
      if (paddedHeight != resizedHeight) {
        throw new IllegalArgumentException(
            "the portrait long side is never padded, so paddedHeight must equal resizedHeight: "
                + paddedHeight
                + " != "
                + resizedHeight);
      }
    } else if (paddedWidth != resizedWidth) {
      throw new IllegalArgumentException(
          "the landscape long side is never padded, so paddedWidth must equal resizedWidth: "
              + paddedWidth
              + " != "
              + resizedWidth);
    }
    if (gridH <= 0 || gridH != paddedHeight / 336) {
      throw new IllegalArgumentException(
          "gridH must be paddedHeight / 336 = " + paddedHeight / 336 + ": " + gridH);
    }
    if (gridW <= 0 || gridW != paddedWidth / 336) {
      throw new IllegalArgumentException(
          "gridW must be paddedWidth / 336 = " + paddedWidth / 336 + ": " + gridW);
    }
    if (cropBlocks <= 0 || cropBlocks != (long) gridH * gridW) {
      throw new IllegalArgumentException(
          "cropBlocks must be gridH * gridW = " + (long) gridH * gridW + ": " + cropBlocks);
    }
    if (numCrops <= 0) {
      throw new IllegalArgumentException("numCrops must be positive: " + numCrops);
    }
    if (cropBlocks > numCrops) {
      throw new IllegalArgumentException(
          "the reference scale loop bounds the block count by num_crops, so cropBlocks must be at"
              + " most numCrops: "
              + cropBlocks
              + " > "
              + numCrops);
    }
    if (slotCount != Math.max(cropBlocks + 1, numCrops + 1)) {
      throw new IllegalArgumentException(
          "slotCount must be max(cropBlocks + 1, numCrops + 1) = "
              + Math.max(cropBlocks + 1, numCrops + 1)
              + ": "
              + slotCount);
    }
    int expectedTokens = (cropBlocks + 1) * 144 + 1 + (gridH + 1) * 12;
    if (numImgTokens != expectedTokens) {
      throw new IllegalArgumentException(
          "numImgTokens must be (cropBlocks + 1) * 144 + 1 + (gridH + 1) * 12 = "
              + expectedTokens
              + ": "
              + numImgTokens);
    }
  }

  /**
   * The reference's {@code image_sizes} entry for this image, the padded dimensions as {@code
   * [height, width]}.
   */
  public int[] imageSize() {
    return new int[] {paddedHeight, paddedWidth};
  }
}
