package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.nn.KVCachePolicy;

/**
 * The result of expanding one unexpanded Qwen3-VL prompt: the original IDs and the expanded IDs
 * (owned copies), each image's contiguous feature destination range, the three-axis prefill
 * positions, the post-prefill decode offset, and the checked cache-budget arithmetic the generation
 * loop uses.
 *
 * <p>Every plan is valid by construction: the constructor rejects position axes that are not as
 * long as the expanded prompt or that hold a negative coordinate, placements whose destination run
 * falls out of bounds or is not strictly increasing in prompt order, and a delta that is not the
 * largest coordinate plus one minus the expanded length (the reference's definition). What a
 * destination <i>holds</i> is not a placement fact — the plan carries no image token of its own —
 * so verifying that every destination holds the image token, and that no image token is left
 * undelivered, is the expander's job ({@link Qwen3VlPromptExpander}).
 *
 * <p>Physical expanded order, logical cache positions and RoPE coordinates are three separate
 * counters: the causal mask and the KV cache advance by one per expanded token, while the RoPE
 * coordinates follow the planned axes. The prompt's finished position is {@link #expandedLength()};
 * the {@link #unexpandedIds()} are retained (owned copy) so results and capacity errors can report
 * both lengths. Decode coordinates continue from the logical cache position plus {@link
 * #positionDelta()} ({@link #ropePositionAt(int)}), never from a retained-KV length after eviction.
 * Equality is by value: the ID and position arrays elementwise, the placements listwise and the
 * delta numerically.
 *
 * @param unexpandedIds the original prompt IDs
 * @param expandedIds the expanded prompt IDs
 * @param images one destination range per image, in prompt order
 * @param temporalPositions the temporal (T) RoPE axis, one coordinate per expanded token
 * @param heightPositions the height (H) RoPE axis
 * @param widthPositions the width (W) RoPE axis
 * @param positionDelta the largest prefill coordinate plus one minus the expanded length (negative
 *     for image prompts, zero for text-only ones)
 */
record Qwen3VlPromptPlan(
    int[] unexpandedIds,
    int[] expandedIds,
    List<Qwen3VlImagePlacement> images,
    int[] temporalPositions,
    int[] heightPositions,
    int[] widthPositions,
    int positionDelta) {

  public Qwen3VlPromptPlan {
    unexpandedIds = Objects.requireNonNull(unexpandedIds, "unexpandedIds").clone();
    expandedIds = Objects.requireNonNull(expandedIds, "expandedIds").clone();
    images = List.copyOf(Objects.requireNonNull(images, "images"));
    temporalPositions = requireAxis(temporalPositions, "temporalPositions", expandedIds.length);
    heightPositions = requireAxis(heightPositions, "heightPositions", expandedIds.length);
    widthPositions = requireAxis(widthPositions, "widthPositions", expandedIds.length);
    verifyDestinations(expandedIds.length, images);
    verifyDelta(
        positionDelta, temporalPositions, heightPositions, widthPositions, expandedIds.length);
  }

  @Override
  public int[] unexpandedIds() {
    return unexpandedIds.clone();
  }

  @Override
  public int[] expandedIds() {
    return expandedIds.clone();
  }

  @Override
  public int[] temporalPositions() {
    return temporalPositions.clone();
  }

  @Override
  public int[] heightPositions() {
    return heightPositions.clone();
  }

  @Override
  public int[] widthPositions() {
    return widthPositions.clone();
  }

  /** The expanded prompt's token count (its finished logical cache position). */
  public int expandedLength() {
    return expandedIds.length;
  }

  /** The original, unexpanded prompt's token count. */
  public int unexpandedLength() {
    return unexpandedIds.length;
  }

  /** The total merged feature row count across all images, in checked arithmetic. */
  public int totalFeatureRows() {
    long total = 0;
    for (Qwen3VlImagePlacement image : images) {
      total = Math.addExact(total, image.featureCount());
    }
    return Math.toIntExact(total);
  }

  /**
   * The flat, strictly increasing feature destination positions of all images, in prompt order.
   *
   * @return one expanded-ID position per merged feature row
   */
  public int[] featureDestinations() {
    int[] destinations = new int[totalFeatureRows()];
    int i = 0;
    for (Qwen3VlImagePlacement image : images) {
      for (int position = image.featureStart(); position < image.featureEnd(); position++) {
        destinations[i++] = position;
      }
    }
    return destinations;
  }

  /**
   * The checked KV cache budget for a generation request: zero when no new tokens are requested,
   * otherwise the expanded prompt plus the new tokens minus the finished prompt's own position.
   *
   * @param maxNewTokens the maximum number of tokens to generate; must be non-negative
   * @return the capacity the policy must guarantee
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
   * Asks the policy to guarantee this plan's budget for the given request, using the same
   * arithmetic as the 7.3a family.
   *
   * @param policy the generation's KV cache policy; null is rejected
   * @param maxNewTokens the maximum number of tokens to generate
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
   * The shared RoPE coordinate all three axes take at one decode step: the logical cache position
   * plus this plan's delta. A negative result is rejected: decoding before the expanded prompt is
   * fully prefilled has no valid coordinate.
   *
   * @param cachePosition the logical cache position of the decode step; must be non-negative
   * @return the coordinate to use on all three axes
   */
  public int ropePositionAt(int cachePosition) {
    if (cachePosition < 0) {
      throw new IllegalArgumentException("cachePosition must be non-negative: " + cachePosition);
    }
    int position = Math.addExact(cachePosition, positionDelta);
    if (position < 0) {
      throw new IllegalArgumentException(
          "RoPE position at cache position "
              + cachePosition
              + " is negative: "
              + position
              + " (delta "
              + positionDelta
              + "); decode starts at cache position "
              + expandedLength());
    }
    return position;
  }

  private static int[] requireAxis(int[] axis, String name, int expandedLength) {
    int[] copy = Objects.requireNonNull(axis, name).clone();
    if (copy.length != expandedLength) {
      throw new IllegalArgumentException(
          name
              + " must be as long as the expanded prompt: "
              + copy.length
              + " vs "
              + expandedLength);
    }
    for (int position : copy) {
      if (position < 0) {
        throw new IllegalArgumentException(name + " positions must be non-negative: " + position);
      }
    }
    return copy;
  }

  private static void verifyDestinations(int expandedLength, List<Qwen3VlImagePlacement> images) {
    int previous = -1;
    for (Qwen3VlImagePlacement image : images) {
      long end = (long) image.featureStart() + image.featureCount();
      if (end > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "feature destination run " + image + " overflows a signed 32-bit integer");
      }
      for (int position = image.featureStart(); position < end; position++) {
        if (position <= previous) {
          throw new IllegalArgumentException(
              "feature destinations must be strictly increasing: "
                  + position
                  + " after "
                  + previous);
        }
        if (position >= expandedLength) {
          throw new IllegalArgumentException(
              "feature destination "
                  + position
                  + " is out of bounds for the expanded prompt ("
                  + expandedLength
                  + " tokens)");
        }
        previous = position;
      }
    }
  }

  private static void verifyDelta(int positionDelta, int[] t, int[] h, int[] w, int length) {
    if (length == 0) {
      if (positionDelta != 0) {
        throw new IllegalArgumentException(
            "positionDelta must be zero for an empty prompt: " + positionDelta);
      }
      return;
    }
    int max = -1;
    for (int position : t) {
      max = Math.max(max, position);
    }
    for (int position : h) {
      max = Math.max(max, position);
    }
    for (int position : w) {
      max = Math.max(max, position);
    }
    int expected = Math.addExact(Math.addExact(max, 1), -length);
    if (positionDelta != expected) {
      throw new IllegalArgumentException(
          "positionDelta must equal the largest coordinate plus one minus the expanded length: "
              + positionDelta
              + " vs "
              + expected);
    }
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof Qwen3VlPromptPlan that)) {
      return false;
    }
    return positionDelta == that.positionDelta
        && Arrays.equals(unexpandedIds, that.unexpandedIds)
        && Arrays.equals(expandedIds, that.expandedIds)
        && images.equals(that.images)
        && Arrays.equals(temporalPositions, that.temporalPositions)
        && Arrays.equals(heightPositions, that.heightPositions)
        && Arrays.equals(widthPositions, that.widthPositions);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        Arrays.hashCode(unexpandedIds),
        Arrays.hashCode(expandedIds),
        images,
        Arrays.hashCode(temporalPositions),
        Arrays.hashCode(heightPositions),
        Arrays.hashCode(widthPositions),
        positionDelta);
  }

  @Override
  public String toString() {
    return "Qwen3VlPromptPlan["
        + "unexpandedIds="
        + Arrays.toString(unexpandedIds)
        + ", expandedIds="
        + Arrays.toString(expandedIds)
        + ", images="
        + images
        + ", temporalPositions="
        + Arrays.toString(temporalPositions)
        + ", heightPositions="
        + Arrays.toString(heightPositions)
        + ", widthPositions="
        + Arrays.toString(widthPositions)
        + ", positionDelta="
        + positionDelta
        + "]";
  }
}
