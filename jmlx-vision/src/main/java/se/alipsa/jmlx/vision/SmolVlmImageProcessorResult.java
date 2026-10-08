package se.alipsa.jmlx.vision;

import java.util.List;

/**
 * Immutable result of {@link SmolVlmImageProcessor#process}: the per-tile normalized float32
 * tensors, their valid-pixel masks, and the split grid/order metadata.
 *
 * <p>Tile order matches the reference processor: row-major crops over the split grid, with the
 * final tile being the whole (second-stage) image resized to the global tile size. When no
 * splitting occurred there is exactly one tile, the square-resized image, and {@link #rows()} and
 * {@link #cols()} are both zero.
 *
 * <p>Each tile is a single-batch NHWC tensor ({@code [1, height, width, 3]}); its mask is a
 * row-major {@code height * width} array of 0/1 values where 1 marks a real image pixel and 0 marks
 * padding. Tensors and masks preserve their association: index {@code i} of {@link #tiles()} and
 * index {@code i} of {@link #pixelMasks()} describe the same tile.
 *
 * <p>The list and mask arrays are defensive: mutation by a caller is never observable, and the mask
 * arrays are copied both on construction and access.
 */
public final class SmolVlmImageProcessorResult {
  private final List<ImageTensor> tiles;
  private final List<int[]> pixelMasks;
  private final int rows;
  private final int cols;

  SmolVlmImageProcessorResult(List<ImageTensor> tiles, List<int[]> pixelMasks, int rows, int cols) {
    if (tiles.size() != pixelMasks.size()) {
      throw new IllegalArgumentException("tiles and pixelMasks must have the same length");
    }
    this.tiles = List.copyOf(tiles);
    this.pixelMasks = pixelMasks.stream().map(m -> m.clone()).toList();
    this.rows = rows;
    this.cols = cols;
  }

  /**
   * The normalized tiles in reference order (row-major crops, global thumbnail last); the list is
   * unmodifiable and the tensors are immutable.
   */
  public List<ImageTensor> tiles() {
    return tiles;
  }

  /**
   * The valid-pixel masks, one per tile in the same order as {@link #tiles()}; the list is
   * unmodifiable and every mask array is a fresh copy.
   */
  public List<int[]> pixelMasks() {
    return pixelMasks.stream().map(m -> m.clone()).toList();
  }

  /** The number of vertical crop splits (0 when no splitting occurred). */
  public int rows() {
    return rows;
  }

  /** The number of horizontal crop splits (0 when no splitting occurred). */
  public int cols() {
    return cols;
  }
}
