package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Pure SmolVLM prompt expansion: replaces each {@code <image>} placeholder in an unexpanded prompt
 * with the processor's expanded marker sequence.
 *
 * <p>The pinned reference (transformers 4.57.6 {@code SmolVLMProcessor}) expands at the string
 * level and re-tokenizes the result ({@code expand_text_with_image_tokens} followed by the
 * tokenizer on the expanded text). Every inserted character other than the newlines is an added
 * token that re-encodes to its single ID, and the tokens outside the substitution regions re-encode
 * unchanged, so the ID-level result is:
 *
 * <ul>
 *   <li>unsplit image ({@code (0, 0)} grid): {@code fake, global, image x tokensPerTile, fake};
 *   <li>split image ({@code rows x cols}): for each row {@code (fake, <row_r_col_c>, image x
 *       tokensPerTile) x cols} followed by a newline run, then the global block {@code fake,
 *       global, image x tokensPerTile, fake}. The newline run after every row except the last is
 *       the tokenizer's lone-{@code \n} encoding; after the last row it is the tokenizer's
 *       doubled-{@code \n\n} encoding, because the row's newline and the global block's leading
 *       newline re-tokenize together (to the single piece 1116 in the pinned SmolLM2 tokenizer).
 * </ul>
 *
 * <p>The output owns the expanded IDs and, per image in prompt order, the ordered start positions
 * of each tile's {@code tokensPerTile}-token image run (crops row-major, global thumbnail last —
 * the processor's frame order). {@link #expand(int[], SmolVlmPromptTokens, List, int)} builds the
 * plan — whose constructor enforces that every projected feature-row destination is in bounds for
 * the expanded IDs and strictly increasing — and then verifies that the expanded prompt holds
 * exactly one image token per projected feature row and that every destination holds the image
 * token, before returning. This class never receives native tensors or pixels: it is pure ID-space
 * arithmetic with checked length math.
 */
final class SmolVlmPromptExpander {

  private SmolVlmPromptExpander() {}

  /**
   * Expands the unexpanded prompt against the supplied image grids.
   *
   * @param unexpandedIds the prompt IDs, one {@code <image>} placeholder per supplied image
   * @param tokens validated special IDs and newline encodings
   * @param grids one split grid per supplied image, in placeholder order
   * @param tokensPerTile image tokens per tile (the processor's {@code image_seq_len})
   * @return the expanded prompt plan
   * @throws NullPointerException on a null argument ({@code unexpandedIds}, {@code tokens}, {@code
   *     grids}, or a {@code grids} entry)
   * @throws IllegalArgumentException on a non-positive {@code tokensPerTile}, a grid outside the
   *     6x6 marker space, a placeholder/image count mismatch, a processor-only marker (fake,
   *     row/column, global) in the prompt, a video token in the prompt, or an expanded length that
   *     overflows a signed 32-bit integer
   * @throws IllegalStateException if the built expansion fails the feature-destination invariant:
   *     the expanded prompt must hold exactly one image token per projected feature row, and every
   *     destination must hold the image token (destination bounds and strict ordering are enforced
   *     by the plan's constructor)
   */
  static SmolVlmPromptPlan expand(
      int[] unexpandedIds,
      SmolVlmPromptTokens tokens,
      List<SmolVlmImageGrid> grids,
      int tokensPerTile) {
    Objects.requireNonNull(unexpandedIds, "unexpandedIds");
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(grids, "grids");
    if (tokensPerTile <= 0) {
      throw new IllegalArgumentException("tokensPerTile must be positive: " + tokensPerTile);
    }
    for (SmolVlmImageGrid grid : grids) {
      Objects.requireNonNull(grid, "grids entries");
      if (grid.rows() > SmolVlmPromptTokens.MARKER_GRID_SIDE
          || grid.cols() > SmolVlmPromptTokens.MARKER_GRID_SIDE) {
        throw new IllegalArgumentException(
            "image grid "
                + grid.rows()
                + "x"
                + grid.cols()
                + " exceeds the "
                + SmolVlmPromptTokens.MARKER_GRID_SIDE
                + "x"
                + SmolVlmPromptTokens.MARKER_GRID_SIDE
                + " row/column marker space");
      }
    }

    int placeholderCount = 0;
    for (int id : unexpandedIds) {
      if (id == tokens.imageTokenId()) {
        placeholderCount++;
      }
    }
    if (placeholderCount != grids.size()) {
      throw new IllegalArgumentException(
          "prompt contains "
              + placeholderCount
              + " <image> placeholder(s) but "
              + grids.size()
              + " image(s) were provided");
    }
    rejectNonExpandableTokens(unexpandedIds, tokens);

    // Each placeholder is replaced by its region, not retained: the surrounding tokens plus the
    // regions.
    long length = unexpandedIds.length - placeholderCount;
    for (SmolVlmImageGrid grid : grids) {
      length = Math.addExact(length, expansionLength(grid, tokens, tokensPerTile));
    }
    if (length > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "expanded prompt length " + length + " overflows a signed 32-bit integer");
    }

    int[] expanded = new int[(int) length];
    List<SmolVlmImagePlacement> placements = new ArrayList<>(grids.size());
    int position = 0;
    int imageIndex = 0;
    for (int id : unexpandedIds) {
      if (id == tokens.imageTokenId()) {
        position =
            writeImageRegion(
                expanded, position, grids.get(imageIndex), tokens, tokensPerTile, placements);
        imageIndex++;
      } else {
        expanded[position++] = id;
      }
    }

    SmolVlmPromptPlan plan =
        new SmolVlmPromptPlan(unexpandedIds, expanded, tokensPerTile, List.copyOf(placements));
    verifyFeatureDestinations(plan, tokens.imageTokenId());
    return plan;
  }

  /** Expands the grid's marker sequence; returns the position after the region. */
  private static int writeImageRegion(
      int[] out,
      int position,
      SmolVlmImageGrid grid,
      SmolVlmPromptTokens tokens,
      int tokensPerTile,
      List<SmolVlmImagePlacement> placements) {
    int[] tileStarts;
    if (grid.rows() == 0) {
      out[position++] = tokens.fakeTokenId();
      out[position++] = tokens.globalTokenId();
      tileStarts = new int[] {position};
      Arrays.fill(out, position, position + tokensPerTile, tokens.imageTokenId());
      position += tokensPerTile;
      out[position++] = tokens.fakeTokenId();
    } else {
      int rows = grid.rows();
      int cols = grid.cols();
      int[] rowNewlineRun = tokens.rowNewlineTokenIds();
      int[] globalNewlineRun = tokens.globalBlockNewlineTokenIds();
      int[] starts = new int[grid.tileCount()];
      int tile = 0;
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
          out[position++] = tokens.fakeTokenId();
          out[position++] = tokens.rowColTokenId(row + 1, col + 1);
          starts[tile++] = position;
          Arrays.fill(out, position, position + tokensPerTile, tokens.imageTokenId());
          position += tokensPerTile;
        }
        position = writeRun(out, position, row + 1 < rows ? rowNewlineRun : globalNewlineRun);
      }
      out[position++] = tokens.fakeTokenId();
      out[position++] = tokens.globalTokenId();
      starts[tile++] = position;
      Arrays.fill(out, position, position + tokensPerTile, tokens.imageTokenId());
      position += tokensPerTile;
      out[position++] = tokens.fakeTokenId();
      tileStarts = starts;
    }
    placements.add(new SmolVlmImagePlacement(tileStarts));
    return position;
  }

  private static int writeRun(int[] out, int position, int[] run) {
    System.arraycopy(run, 0, out, position, run.length);
    return position + run.length;
  }

  private static void rejectNonExpandableTokens(int[] unexpandedIds, SmolVlmPromptTokens tokens) {
    int blockHigh = tokens.rowColTokenIdBase() + SmolVlmPromptTokens.ROW_COL_MARKER_COUNT;
    for (int id : unexpandedIds) {
      if (id == tokens.fakeTokenId()) {
        throw new IllegalArgumentException(
            "prompt contains the processor-only fake token (ID "
                + id
                + "); it is generated by expansion and must not appear in the prompt");
      }
      if (id == tokens.globalTokenId()) {
        throw new IllegalArgumentException(
            "prompt contains the processor-only global token (ID "
                + id
                + "); it is generated by expansion and must not appear in the prompt");
      }
      if (id > tokens.rowColTokenIdBase() && id <= blockHigh) {
        throw new IllegalArgumentException(
            "prompt contains a processor-only row/column marker (ID "
                + id
                + "); it is generated by expansion and must not appear in the prompt");
      }
      if (tokens.videoTokenId() >= 0 && id == tokens.videoTokenId()) {
        throw new IllegalArgumentException(
            "prompt contains the video token (ID " + id + "); video is not supported");
      }
    }
  }

  /** The expanded length of one image's marker sequence, in checked long arithmetic. */
  private static long expansionLength(
      SmolVlmImageGrid grid, SmolVlmPromptTokens tokens, int tokensPerTile) {
    // fake + global + image run + fake.
    long length = (long) tokensPerTile + 3;
    if (grid.rows() > 0) {
      int rowNewlineLength = tokens.rowNewlineTokenIds().length;
      int globalNewlineLength = tokens.globalBlockNewlineTokenIds().length;
      for (int row = 0; row < grid.rows(); row++) {
        length = Math.addExact(length, (long) grid.cols() * (2L + tokensPerTile));
        length =
            Math.addExact(length, row + 1 < grid.rows() ? rowNewlineLength : globalNewlineLength);
      }
    }
    return length;
  }

  /**
   * Invariant the builder must satisfy: the expanded prompt holds exactly one image token per
   * projected feature row, and each destination holds the image token. Destination bounds and
   * strict ordering are enforced by the plan's constructor, so only the image-token content checks
   * remain here. Package-private so the package's tests can drive it against hand-constructed plans
   * the builder can never emit.
   */
  static void verifyFeatureDestinations(SmolVlmPromptPlan plan, int imageTokenId) {
    int[] expanded = plan.expandedIds();
    int imageTokens = 0;
    for (int id : expanded) {
      if (id == imageTokenId) {
        imageTokens++;
      }
    }
    if (imageTokens != plan.totalFeatureRows()) {
      throw new IllegalStateException(
          "expanded prompt holds "
              + imageTokens
              + " image tokens but "
              + plan.totalFeatureRows()
              + " feature rows");
    }
    for (int destination : plan.featureDestinations()) {
      if (expanded[destination] != imageTokenId) {
        throw new IllegalStateException(
            "feature destination " + destination + " does not hold the image token");
      }
    }
  }
}
