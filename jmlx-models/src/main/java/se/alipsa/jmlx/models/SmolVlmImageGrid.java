package se.alipsa.jmlx.models;

/**
 * One supplied image's processor split grid: {@code (0, 0)} for an unsplit image (a single tile),
 * otherwise positive rows and columns of crops plus a global thumbnail. Comes from the processor
 * (its result's rows/cols) or from the shared pure geometry plan — never re-derived here.
 *
 * @param rows split rows, or 0 when unsplit
 * @param cols split columns, or 0 when unsplit
 */
record SmolVlmImageGrid(int rows, int cols) {

  public SmolVlmImageGrid {
    if (rows < 0 || cols < 0) {
      throw new IllegalArgumentException("image grid must not be negative: " + rows + "x" + cols);
    }
    if ((rows == 0) != (cols == 0)) {
      throw new IllegalArgumentException(
          "image grid must be (0, 0) or have both rows and cols positive: " + rows + "x" + cols);
    }
  }

  /** Tile count: {@code rows x cols} crops plus the global thumbnail; 1 when unsplit. */
  public int tileCount() {
    return Math.addExact(Math.multiplyExact(rows, cols), 1);
  }
}
