package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.vision.Qwen3VlGeometryPlan;

/**
 * Pure Qwen3-VL three-axis RoPE position planning: a faithful port of the pinned reference's {@code
 * Qwen3VLModel.get_rope_index} image path (transformers 4.57.6, {@code
 * modular_qwen3_vl.py:858-955}).
 *
 * <p>The reference loops once per image, in prompt order. For image {@code k}: {@code ed} is the
 * first image token at or after the cursor {@code st}; the text run {@code [st, ed)} takes all
 * three axes equal to {@code arange(text_len) + st_idx}; the image run of {@code llm_t * llm_h *
 * llm_w} tokens takes T constant, H row-major by {@code llm_w} and W fastest, each offset by {@code
 * st_idx + text_len}, where the LLM grid is the vision grid divided by the merge side; the cursor
 * advances to {@code ed + llm_t * llm_h * llm_w}. {@code st_idx} is the previous block's maximum
 * plus one (zero for the first block), so every block starts immediately after the last coordinate
 * written. Trailing text resumes the same way. The delta is the largest coordinate plus one minus
 * the sequence length: negative for image prompts, zero for text-only ones.
 *
 * <p>Three independent counters stay separate: the physical expanded order (one entry per token),
 * the logical cache position (advances by one per expanded token) and the RoPE coordinates (the
 * planned axes). Decode continues from the logical cache position plus the delta on all three axes,
 * never from a retained-KV length.
 *
 * <p>All arithmetic is checked: an image run, cursor advance or coordinate that would overflow a
 * signed 32-bit integer is rejected before any array is written, and a grid that does not fit the
 * expanded prompt is rejected as a geometry/prompt mismatch. This class never receives native
 * tensors or pixels.
 */
final class Qwen3VlPositionPlanner {

  private Qwen3VlPositionPlanner() {}

  /**
   * The LLM-space grid {@code (t, h / merge, w / merge)} of one image's geometry, where the merge
   * side is the square root of the geometry record's raw-to-merged patch ratio. The geometry record
   * guarantees that ratio is a perfect square but not that the grid is divisible by the merge side,
   * so that divisibility and the resulting merged token count are verified here.
   *
   * @param geometry the image's validated geometry; null is rejected
   * @return a positive {@code (t, h, w)} triple in LLM space, whose product equals the geometry's
   *     {@code mergedRowCount}
   * @throws IllegalArgumentException when the grid is not divisible by the merge side or the merged
   *     grid disagrees with the geometry's {@code mergedRowCount}
   */
  static int[] llmGrid(Qwen3VlGeometryPlan geometry) {
    Objects.requireNonNull(geometry, "geometry");
    int t = geometry.gridT();
    int h = geometry.gridH();
    int w = geometry.gridW();
    long perMergeBlock = (long) geometry.rawPatchRows() / geometry.mergedRowCount();
    long mergeSide = (long) Math.sqrt((double) perMergeBlock);
    if (mergeSide * mergeSide != perMergeBlock) {
      throw new IllegalArgumentException(
          "rawPatchRows / mergedRowCount must be merge_size^2: " + perMergeBlock);
    }
    if (h % mergeSide != 0 || w % mergeSide != 0) {
      throw new IllegalArgumentException(
          "image grid " + h + "x" + w + " is not divisible by its merge size " + mergeSide);
    }
    int llmH = (int) (h / mergeSide);
    int llmW = (int) (w / mergeSide);
    long merged = (long) t * llmH * llmW;
    if (merged != geometry.mergedRowCount()) {
      throw new IllegalArgumentException(
          "merged grid "
              + t
              + "x"
              + llmH
              + "x"
              + llmW
              + " ("
              + merged
              + " tokens) disagrees with mergedRowCount "
              + geometry.mergedRowCount());
    }
    return new int[] {t, llmH, llmW};
  }

  /**
   * Plans the three axes and the delta for one expanded prompt (a single batch row), matching the
   * reference's image path: {@code llmGrids} holds one positive LLM-space {@code (t, h, w)} grid
   * per image, in prompt order.
   *
   * @param expandedIds the expanded prompt's token IDs; null is rejected
   * @param tokens the prompt-side token IDs (the image token is searched for)
   * @param llmGrids one LLM-space grid per image, in prompt order; null entries are rejected
   * @return the planned axes and delta, always equally long with the expanded prompt
   * @throws IllegalArgumentException when a grid is malformed, an image token is missing at or
   *     after a cursor, or a grid or coordinate overflows
   */
  static Qwen3VlRopePlan plan(int[] expandedIds, Qwen3VlPromptTokens tokens, List<int[]> llmGrids) {
    Objects.requireNonNull(expandedIds, "expandedIds");
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(llmGrids, "llmGrids");
    for (int[] grid : llmGrids) {
      requireLlmGrid(grid);
    }
    int length = expandedIds.length;
    int[] t = new int[length];
    int[] h = new int[length];
    int[] w = new int[length];

    int st = 0;
    int previousBlockMax = -1;
    int imageToken = tokens.imageTokenId();
    for (int[] grid : llmGrids) {
      int llmT = grid[0];
      int llmH = grid[1];
      int llmW = grid[2];
      long imageLen = (long) llmT * llmH * llmW;
      if (imageLen > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "image run ("
                + llmT
                + ", "
                + llmH
                + ", "
                + llmW
                + ") of "
                + imageLen
                + " tokens overflows a signed 32-bit integer");
      }
      int stIdx = previousBlockMax < 0 ? 0 : Math.addExact(previousBlockMax, 1);
      int ed = imageTokenAtOrAfter(expandedIds, imageToken, st);
      int textLen = ed - st;
      long next = (long) ed + imageLen;
      if (next > length) {
        throw new IllegalArgumentException(
            "image grid ("
                + llmT
                + ", "
                + llmH
                + ", "
                + llmW
                + ") at token "
                + ed
                + " extends past the expanded prompt ("
                + length
                + " tokens)");
      }
      long blockMax = (long) stIdx + textLen + blockMaxDimension(llmT, llmH, llmW) - 1;
      if (blockMax > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "image coordinates overflow a signed 32-bit integer: block maximum " + blockMax);
      }
      writeBlock(t, h, w, st, textLen, stIdx);
      writeImageBlock(t, h, w, ed, (int) imageLen, llmT, llmH, llmW, stIdx + textLen);
      st = (int) next;
      previousBlockMax = (int) blockMax;
    }
    if (st < length) {
      int stIdx = previousBlockMax < 0 ? 0 : Math.addExact(previousBlockMax, 1);
      writeBlock(t, h, w, st, length - st, stIdx);
      previousBlockMax = Math.addExact(stIdx, length - st - 1);
    }
    int delta = length == 0 ? 0 : Math.addExact(Math.addExact(previousBlockMax, 1), -length);
    return new Qwen3VlRopePlan(t, h, w, delta);
  }

  private static void requireLlmGrid(int[] grid) {
    Objects.requireNonNull(grid, "llmGrids entries");
    if (grid.length != 3 || grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0) {
      throw new IllegalArgumentException(
          "llm grids are positive (t, h, w) triples: " + Arrays.toString(grid));
    }
  }

  private static int blockMaxDimension(int llmT, int llmH, int llmW) {
    return Math.max(llmT, Math.max(llmH, llmW));
  }

  /**
   * Writes one text run to all three axes: the run's {@code textLen} coordinates take equal values
   * {@code stIdx + i} on each axis. The block's maximum is checked for overflow first, so the
   * per-element arithmetic below cannot overflow.
   */
  private static void writeBlock(int[] t, int[] h, int[] w, int st, int textLen, int stIdx) {
    long blockMax = (long) stIdx + textLen - 1;
    if (blockMax > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "text block coordinates overflow a signed 32-bit integer: maximum " + blockMax);
    }
    for (int i = 0; i < textLen; i++) {
      int position = st + i;
      int coordinate = stIdx + i;
      t[position] = coordinate;
      h[position] = coordinate;
      w[position] = coordinate;
    }
  }

  /**
   * Writes one image block of {@code imageLen} tokens starting at expanded position {@code ed}: T
   * takes the shared coordinate, H cycles by rows of {@code llmW} and W fastest, matching the
   * reference's {@code arange}-and-{@code expand}-and-{@code flatten} construction.
   */
  private static void writeImageBlock(
      int[] t,
      int[] h,
      int[] w,
      int ed,
      int imageLen,
      int llmT,
      int llmH,
      int llmW,
      int blockBase) {
    long blockMax = (long) blockBase + blockMaxDimension(llmT, llmH, llmW) - 1;
    if (blockMax > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "image coordinates overflow a signed 32-bit integer: block maximum " + blockMax);
    }
    int strideHW = Math.multiplyExact(llmH, llmW);
    for (int r = 0; r < imageLen; r++) {
      int position = ed + r;
      t[position] = blockBase + r / strideHW;
      h[position] = blockBase + (r / llmW) % llmH;
      w[position] = blockBase + r % llmW;
    }
  }

  private static int imageTokenAtOrAfter(int[] ids, int imageToken, int from) {
    for (int i = from; i < ids.length; i++) {
      if (ids[i] == imageToken) {
        return i;
      }
    }
    throw new IllegalArgumentException(
        "expanded prompt holds no image token (ID "
            + imageToken
            + ") at or after position "
            + from
            + "; the image grids disagree with the expanded prompt");
  }
}
