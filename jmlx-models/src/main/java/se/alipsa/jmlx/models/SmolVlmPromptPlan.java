package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.nn.KVCachePolicy;

/**
 * The result of expanding one unexpanded SmolVLM prompt: the expanded IDs (owned copy) together
 * with each image's ordered tile feature destinations, and the checked cache-budget arithmetic the
 * generation loop uses.
 *
 * <p>Every plan is valid by construction: the constructor rejects placements whose feature
 * destinations fall out of bounds for the expanded IDs or are not strictly increasing in (image,
 * tile, feature row) order. What a destination <i>holds</i> is not a placement fact — the plan has
 * no image token of its own — so verifying that every destination holds the image token (and that
 * no image token is left undelivered) is the expander's job.
 *
 * <p>{@code promptPositions} for the finished result is {@link #expandedLength()}; the {@link
 * #unexpandedIds()} are retained (owned copy) so results and capacity errors can report both
 * lengths. Equality is by value: the ID arrays elementwise and the placements listwise.
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
    // Every plan is valid by construction: destinations in bounds, strictly increasing. The
    // check is static and takes the components as parameters: the record's field stores happen
    // only after the compact constructor body, so an instance method would read null fields.
    verifyDestinations(expandedIds.length, tokensPerTile, images);
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
    return Math.toIntExact(total);
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
      for (int start : image.tileStartsArray()) {
        for (int row = 0; row < tokensPerTile; row++) {
          destinations[i++] = Math.addExact(start, row);
        }
      }
    }
    return destinations;
  }

  /**
   * Rejects placements whose feature destinations (in {@link #featureDestinations()} order) are not
   * in bounds for the expanded prompt or not strictly increasing; the compact constructor runs it
   * so every plan is valid by construction. Enumerates the destinations itself rather than via
   * {@link #featureDestinations()} for the same reason it is static.
   */
  private static void verifyDestinations(
      int expandedLength, int tokensPerTile, List<SmolVlmImagePlacement> images) {
    int previous = -1;
    for (SmolVlmImagePlacement image : images) {
      for (int start : image.tileStartsArray()) {
        for (int row = 0; row < tokensPerTile; row++) {
          int destination = Math.addExact(start, row);
          if (destination <= previous) {
            throw new IllegalArgumentException(
                "feature destinations must be strictly increasing: "
                    + destination
                    + " after "
                    + previous);
          }
          if (destination >= expandedLength) {
            throw new IllegalArgumentException(
                "feature destination "
                    + destination
                    + " is out of bounds for the expanded prompt ("
                    + expandedLength
                    + " tokens)");
          }
          previous = destination;
        }
      }
    }
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof SmolVlmPromptPlan that)) {
      return false;
    }
    return tokensPerTile == that.tokensPerTile
        && Arrays.equals(unexpandedIds, that.unexpandedIds)
        && Arrays.equals(expandedIds, that.expandedIds)
        && images.equals(that.images);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        Arrays.hashCode(unexpandedIds), Arrays.hashCode(expandedIds), tokensPerTile, images);
  }

  @Override
  public String toString() {
    return "SmolVlmPromptPlan["
        + "unexpandedIds="
        + Arrays.toString(unexpandedIds)
        + ", expandedIds="
        + Arrays.toString(expandedIds)
        + ", tokensPerTile="
        + tokensPerTile
        + ", images="
        + images
        + "]";
  }
}
