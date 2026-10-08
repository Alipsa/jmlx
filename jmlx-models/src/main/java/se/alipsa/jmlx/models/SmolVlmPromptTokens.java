package se.alipsa.jmlx.models;

import java.util.Objects;

/**
 * Validated SmolVLM prompt-side token IDs, resolved from the tokenizer's own metadata at model load
 * and never copied from a checkpoint's example artifact.
 *
 * <p>The two newline encodings are not special tokens: the reference expands the prompt at the
 * string level and re-tokenizes the result (reference findings §6), and the only non-special
 * characters it inserts are the row newlines. {@code rowNewlineTokenIds} is the tokenizer's
 * encoding of the lone {@code \n} after every row except the last; {@code
 * globalBlockNewlineTokenIds} is the encoding of the doubled {@code \n\n} run after the last row
 * (the row's newline plus the global block's leading newline re-tokenize together — to the single
 * piece 1116 in the pinned SmolLM2 tokenizer, to the lone-newline encoding twice in a tokenizer
 * without that merge).
 *
 * <p>{@code rowColTokenBase} anchors the contiguous {@code <row_r_col_c>} marker block: the marker
 * for 1-based row {@code r} and column {@code c} is {@code rowColTokenBase + (r - 1) * 6 + c}, so
 * the used IDs are {@code rowColTokenBase + 1 .. rowColTokenBase + 36} (the base itself is not a
 * row/column marker).
 *
 * @param imageTokenId the {@code <image>} expansion marker
 * @param videoTokenId the tokenizer's {@code <video>} token, or -1 when the tokenizer declares no
 *     video token (the pinned SmolVLM-256M tokenizer has none)
 * @param fakeTokenId the {@code <fake_token_around_image>} marker
 * @param globalTokenId the {@code <global-img>} marker
 * @param rowColTokenIdBase anchor of the 36-marker {@code <row_r_col_c>} block
 * @param rowNewlineTokenIds tokenizer encoding of the lone newline between tile rows
 * @param globalBlockNewlineTokenIds tokenizer encoding of the doubled newline before the global
 *     block
 */
record SmolVlmPromptTokens(
    int imageTokenId,
    int videoTokenId,
    int fakeTokenId,
    int globalTokenId,
    int rowColTokenIdBase,
    int[] rowNewlineTokenIds,
    int[] globalBlockNewlineTokenIds) {

  /** Number of {@code <row_r_col_c>} markers (rows, cols in 1..6). */
  static final int ROW_COL_MARKER_COUNT = 36;

  /** Side of the square marker grid the processor supports. */
  static final int MARKER_GRID_SIDE = 6;

  public SmolVlmPromptTokens {
    if (imageTokenId < 0 || fakeTokenId < 0 || globalTokenId < 0) {
      throw new IllegalArgumentException("image, fake and global token IDs must be non-negative");
    }
    if (videoTokenId < -1) {
      throw new IllegalArgumentException(
          "videoTokenId must be -1 (no video token) or a token ID: " + videoTokenId);
    }
    if (rowColTokenIdBase < 0 || rowColTokenIdBase > Integer.MAX_VALUE - ROW_COL_MARKER_COUNT) {
      throw new IllegalArgumentException(
          "rowColTokenIdBase must fit the 36-marker block: " + rowColTokenIdBase);
    }
    if (imageTokenId == fakeTokenId
        || imageTokenId == globalTokenId
        || fakeTokenId == globalTokenId) {
      throw new IllegalArgumentException("image, fake and global token IDs must be distinct");
    }
    if (videoTokenId >= 0
        && (videoTokenId == imageTokenId
            || videoTokenId == fakeTokenId
            || videoTokenId == globalTokenId)) {
      throw new IllegalArgumentException("videoTokenId must be distinct from the image markers");
    }
    int blockHigh = rowColTokenIdBase + ROW_COL_MARKER_COUNT;
    for (int id = rowColTokenIdBase + 1; id <= blockHigh; id++) {
      if (id == imageTokenId
          || id == fakeTokenId
          || id == globalTokenId
          || (videoTokenId >= 0 && id == videoTokenId)) {
        throw new IllegalArgumentException(
            "row/column marker block "
                + (rowColTokenIdBase + 1)
                + ".."
                + blockHigh
                + " overlaps special token "
                + id);
      }
    }
    rowNewlineTokenIds = nonEmptyIds(rowNewlineTokenIds, "rowNewlineTokenIds");
    globalBlockNewlineTokenIds =
        nonEmptyIds(globalBlockNewlineTokenIds, "globalBlockNewlineTokenIds");
  }

  /** The {@code <row_row_col_col>} marker ID for 1-based row and column. */
  int rowColTokenId(int row, int col) {
    if (row < 1 || row > MARKER_GRID_SIDE || col < 1 || col > MARKER_GRID_SIDE) {
      throw new IllegalArgumentException(
          "row/column markers are 1.." + MARKER_GRID_SIDE + ": " + row + "x" + col);
    }
    return rowColTokenIdBase + (row - 1) * MARKER_GRID_SIDE + col;
  }

  private static int[] nonEmptyIds(int[] ids, String name) {
    int[] copy = Objects.requireNonNull(ids, name).clone();
    if (copy.length == 0) {
      throw new IllegalArgumentException(name + " must not be empty");
    }
    for (int id : copy) {
      if (id < 0) {
        throw new IllegalArgumentException(name + " IDs must be non-negative: " + id);
      }
    }
    return copy;
  }
}
