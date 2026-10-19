package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.vision.Qwen3VlGeometryPlan;

/**
 * Pure Qwen3-VL prompt expansion: replaces each unexpanded image placeholder in a valid vision
 * start, image sentinel, vision end wrapper with the image's merged feature count copies of the
 * image sentinel.
 *
 * <p>The pinned reference (transformers 4.57.6 {@code Qwen3VLProcessor}) expands at the string
 * level: each image's {@code grid_thw.prod() // merge_size**2} tokens replace that image's single
 * sentinel, in prompt order ({@code replace(..., 1)} per image), and the wrapper's vision start /
 * vision end tokens are retained, so a wrapper's three tokens become vision start, mergedRowCount
 * copies of the image sentinel, vision end. Every expanded token re-encodes to its single ID and
 * the tokens outside the substitution regions re-encode unchanged, so the ID-level result this
 * class computes is exactly the reference's: {@code expandedLength = unexpandedLength -
 * wrapperCount + sum(mergedRowCount)}.
 *
 * <p>The accepted pretokenized grammar (WP3 findings, section 6.8): each image is one {@code
 * vision_start, image, vision_end} triple with exactly one image token in the middle; wrappers
 * match images in prompt order; and the prompt may contain any other tokens. Rejected: a video
 * token (when declared), a vision start that is not the head of a complete triple (covers nested
 * and unmatched starts), an image token outside a wrapper (bare, or consecutive, which looks like
 * an already-expanded placeholder that must never be guessed at), an unmatched vision end, and a
 * wrapper/image count mismatch. An empty prompt with no images is a valid degenerate plan.
 *
 * <p>The result owns the expanded IDs, each image's contiguous feature destination range in prompt
 * order, the three-axis prefill positions and the post-prefill delta (see {@link
 * Qwen3VlPromptPlan}). This class never receives native tensors or pixels: it is pure ID-space
 * arithmetic with checked length math.
 */
final class Qwen3VlPromptExpander {

  private Qwen3VlPromptExpander() {}

  /**
   * Expands one unexpanded prompt against the given images, in prompt order.
   *
   * @param unexpandedIds the original prompt IDs (each image's single placeholder); null is
   *     rejected
   * @param tokens the prompt-side token IDs; null is rejected
   * @param images one validated geometry per image, in prompt order; null entries are rejected
   * @return the expanded plan; the unexpanded IDs are retained inside it
   * @throws IllegalArgumentException when the prompt violates the grammar, a geometry is invalid,
   *     the wrapper/image counts differ, or the expanded length overflows
   */
  static Qwen3VlPromptPlan expand(
      int[] unexpandedIds, Qwen3VlPromptTokens tokens, List<Qwen3VlGeometryPlan> images) {
    Objects.requireNonNull(unexpandedIds, "unexpandedIds");
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(images, "images");
    List<int[]> llmGrids = new ArrayList<>(images.size());
    for (Qwen3VlGeometryPlan geometry : images) {
      Objects.requireNonNull(geometry, "images entries");
      llmGrids.add(Qwen3VlPositionPlanner.llmGrid(geometry));
    }

    boolean[] wrapperImage = new boolean[unexpandedIds.length];
    int wrapperCount = scanGrammar(unexpandedIds, tokens, wrapperImage);
    if (wrapperCount != images.size()) {
      throw new IllegalArgumentException(
          "prompt contains "
              + wrapperCount
              + " image placeholder(s) but "
              + images.size()
              + " image(s) were provided");
    }

    long length = unexpandedIds.length;
    for (int[] grid : llmGrids) {
      // Each wrapper's three tokens become start, m image tokens, end: a net m - 1 growth.
      length = Math.addExact(length, (long) mergedTokens(grid) - 1);
    }
    if (length > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "expanded prompt length " + length + " overflows a signed 32-bit integer");
    }

    int[] expanded = new int[(int) length];
    List<Qwen3VlImagePlacement> placements = new ArrayList<>(images.size());
    int position = 0;
    int imageIndex = 0;
    for (int i = 0; i < unexpandedIds.length; i++) {
      if (wrapperImage[i]) {
        int count = mergedTokens(llmGrids.get(imageIndex++));
        placements.add(new Qwen3VlImagePlacement(position, count));
        Arrays.fill(expanded, position, position + count, tokens.imageTokenId());
        position += count;
      } else {
        expanded[position++] = unexpandedIds[i];
      }
    }

    Qwen3VlRopePlan rope = Qwen3VlPositionPlanner.plan(expanded, tokens, llmGrids);
    Qwen3VlPromptPlan plan =
        new Qwen3VlPromptPlan(
            unexpandedIds,
            expanded,
            List.copyOf(placements),
            rope.temporalPositions(),
            rope.heightPositions(),
            rope.widthPositions(),
            rope.delta());
    verifyFeatureDestinations(plan, tokens.imageTokenId());
    return plan;
  }

  /**
   * Scans the pretokenized prompt for the wrapper grammar, marking each wrapper's image token in
   * {@code wrapperImage} and returning the wrapper count.
   */
  private static int scanGrammar(int[] ids, Qwen3VlPromptTokens tokens, boolean[] wrapperImage) {
    int start = tokens.visionStartTokenId();
    int end = tokens.visionEndTokenId();
    int image = tokens.imageTokenId();
    int video = tokens.videoTokenId();
    int wrappers = 0;
    for (int i = 0; i < ids.length; i++) {
      int id = ids[i];
      if (video >= 0 && id == video) {
        throw videoError(id);
      }
      if (id == start) {
        // A start followed by the video token is a (malformed or complete) video wrapper: report
        // the video error instead of the less helpful malformed-wrapper error.
        if (video >= 0 && i + 1 < ids.length && ids[i + 1] == video) {
          throw videoError(video);
        }
        if (i + 2 >= ids.length || ids[i + 1] != image || ids[i + 2] != end) {
          throw new IllegalArgumentException(
              "vision start token (ID "
                  + start
                  + ") at index "
                  + i
                  + " is not the start of a complete vision start, image, vision end wrapper");
        }
        wrapperImage[i + 1] = true;
        wrappers++;
        i += 2; // the loop's increment skips the wrapper's end token
      } else if (id == image) {
        if (i + 1 < ids.length && ids[i + 1] == image) {
          throw new IllegalArgumentException(
              "consecutive image tokens (ID "
                  + image
                  + ") at indices "
                  + i
                  + " and "
                  + (i + 1)
                  + " look like an already-expanded placeholder; pass the unexpanded prompt");
        }
        throw new IllegalArgumentException(
            "bare image token (ID "
                + image
                + ") at index "
                + i
                + " is not wrapped in vision start/end tokens");
      } else if (id == end) {
        throw new IllegalArgumentException(
            "unmatched vision end token (ID " + end + ") at index " + i);
      }
    }
    return wrappers;
  }

  private static IllegalArgumentException videoError(int id) {
    return new IllegalArgumentException(
        "prompt contains the video token (ID " + id + "); video is not supported");
  }

  private static int mergedTokens(int[] grid) {
    return Math.multiplyExact(grid[0], Math.multiplyExact(grid[1], grid[2]));
  }

  /**
   * The feature invariant the plan cannot check on its own: every destination range holds the image
   * token, and no image token is left undelivered.
   *
   * @param plan the plan to verify
   * @param imageTokenId the image token ID the destinations must hold
   * @throws IllegalStateException when a destination holds another token or the expanded prompt
   *     holds more image tokens than planned feature rows
   */
  static void verifyFeatureDestinations(Qwen3VlPromptPlan plan, int imageTokenId) {
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
              + " feature rows are planned");
    }
    for (int destination : plan.featureDestinations()) {
      if (expanded[destination] != imageTokenId) {
        throw new IllegalStateException(
            "feature destination " + destination + " does not hold the image token");
      }
    }
  }
}
