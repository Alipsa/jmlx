package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.nn.KVCachePolicy;

/**
 * The result of expanding one unexpanded SmolVLM prompt: the expanded IDs (owned copy) together
 * with each image's ordered tile feature destinations, and the checked cache-budget arithmetic the
 * generation loop uses.
 *
 * <p>{@code promptPositions} for the finished result is {@link #expandedLength()}; the {@link
 * #unexpandedIds()} are retained (owned copy) so results and capacity errors can report both
 * lengths.
 *
 * @param unexpandedIds the original prompt IDs, owned copy
 * @param expandedIds the expanded prompt IDs, owned copy
 * @param tokensPerTile image tokens per tile (the processor's {@code image_seq_len})
 * @param images one placement per image, in prompt order
 */
record SmolVlmPromptPlan(
    int[] unexpandedIds, int[] expandedIds, int tokensPerTile, List<SmolVlmImagePlacement> images) {

  public SmolVlmPromptPlan {
    unexpandedIds = Objects.requireNonNull(unexpandedIds, "unexpandedIds").clone();
    expandedIds = Objects.requireNonNull(expandedIds, "expandedIds").clone();
    if (tokensPerTile <= 0) {
      throw new IllegalArgumentException("tokensPerTile must be positive: " + tokensPerTile);
    }
    images = List.copyOf(Objects.requireNonNull(images, "images"));
    // Each image occupies at least one unsplit tile: tokensPerTile + 3 marker tokens.
    long minimum = (long) images.size() * ((long) tokensPerTile + 3);
    if (minimum > expandedIds.length) {
      throw new IllegalArgumentException(
          "expanded IDs hold "
              + expandedIds.length
              + " tokens but "
              + images.size()
              + " image(s) need at least "
              + minimum);
    }
  }

  /** The original (unexpanded) prompt IDs; defensive copy. */
  public int[] unexpandedIds() {
    return unexpandedIds.clone();
  }

  /** The expanded prompt IDs; defensive copy. */
  public int[] expandedIds() {
    return expandedIds.clone();
  }

  /** Length of the expanded prompt in tokens. */
  public int expandedLength() {
    return expandedIds.length;
  }

  /** Length of the original (unexpanded) prompt in tokens. */
  public int unexpandedLength() {
    return unexpandedIds.length;
  }

  /** Total tiles across all images. */
  public int totalTiles() {
    long total = 0;
    for (SmolVlmImagePlacement image : images) {
      total = Math.addExact(total, image.tileCount());
    }
    return (int) total;
  }

  /** Total projected feature rows: one per image token in the expanded prompt. */
  public int totalFeatureRows() {
    return Math.multiplyExact(totalTiles(), tokensPerTile);
  }

  /**
   * The cache budget the expanded prompt needs: 0 when {@code maxNewTokens == 0}, otherwise {@code
   * expandedLength + maxNewTokens - 1}.
   *
   * @throws ArithmeticException if the sum overflows a signed 64-bit integer
   */
  public long requiredCacheCapacity(int maxNewTokens) {
    if (maxNewTokens < 0) {
      throw new IllegalArgumentException("maxNewTokens must be non-negative: " + maxNewTokens);
    }
    if (maxNewTokens == 0) {
      return 0L;
    }
    return Math.addExact(Math.addExact((long) expandedLength(), (long) maxNewTokens), -1L);
  }

  /**
   * Validates this plan's cache budget against the policy; the capacity failure message names both
   * the unexpanded and the expanded prompt lengths.
   *
   * @param policy the generation's KV cache policy; null is rejected
   * @param maxNewTokens the configured maximum new tokens (0 = prompt only)
   * @throws IllegalArgumentException when the budget is not retainable, naming both lengths
   */
  public void requireCacheCapacity(KVCachePolicy policy, int maxNewTokens) {
    Objects.requireNonNull(policy, "policy");
    long required = requiredCacheCapacity(maxNewTokens);
    // Clamped: an effectively unbounded token budget is legal on unbounded and sliding caches;
    // only a bounded FULL capacity can be exceeded up front.
    policy.requireCapacity(
        Math.min(required, Integer.MAX_VALUE),
        "generation (expanded prompt "
            + expandedLength()
            + " tokens, unexpanded "
            + unexpandedLength()
            + " tokens)");
  }

  /**
   * Every projected feature-row destination in expanded-ID space, in (image, tile, feature row)
   * order: strictly increasing, one per feature row.
   */
  public int[] featureDestinations() {
    int[] destinations = new int[totalFeatureRows()];
    int i = 0;
    for (SmolVlmImagePlacement image : images) {
      for (int start : image.tileStarts()) {
        for (int row = 0; row < tokensPerTile; row++) {
          destinations[i++] = Math.addExact(start, row);
        }
      }
    }
    return destinations;
  }
}
