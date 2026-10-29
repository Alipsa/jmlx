package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The result of expanding one rendered Phi-3.5-Vision prompt: the rendered text (the unexpanded
 * counterpart, with the image markers in place), the expanded IDs (owned copy, sentinel rows
 * included) and each image's contiguous sentinel destination range.
 *
 * <p>Every plan is valid by construction: the constructor rejects a destination run that falls out
 * of bounds or is not strictly increasing in prompt order. What a destination <i>holds</i> is not a
 * placement fact — the plan carries no sentinel of its own — so verifying that every destination
 * holds its image's sentinel and that no sentinel row is left unplanned is the expander's job
 * ({@link Phi3VPromptExpander}).
 *
 * <p>Unlike the Qwen3-VL plan, this plan carries no position axes and no delta: the decoder assigns
 * the plain positions {@code 0 .. expandedLength() - 1}, because the sentinel rows occupy real
 * positions in the sequence (the reference replaces their embeddings in place, findings §7.1). The
 * prompt's finished position is {@link #expandedLength()}. Equality is by value: the rendered text
 * and the ID array elementwise, the destinations listwise.
 *
 * @param renderedText the rendered prompt text, image markers included
 * @param expandedIds the expanded prompt IDs (sentinel rows {@code -1 .. -n} included)
 * @param images one destination range per image, in prompt order
 */
record Phi3VPromptPlan(String renderedText, int[] expandedIds, List<Phi3VImageDestination> images) {

  public Phi3VPromptPlan {
    renderedText = Objects.requireNonNull(renderedText, "renderedText");
    expandedIds = Objects.requireNonNull(expandedIds, "expandedIds").clone();
    images = List.copyOf(Objects.requireNonNull(images, "images"));
    verifyDestinations(expandedIds.length, images);
  }

  @Override
  public int[] expandedIds() {
    return expandedIds.clone();
  }

  /** The expanded prompt's token count (its finished logical cache position). */
  public int expandedLength() {
    return expandedIds.length;
  }

  /** The total sentinel row count across all images, in checked arithmetic. */
  public int totalSentinelRows() {
    long total = 0;
    for (Phi3VImageDestination image : images) {
      total = Math.addExact(total, image.sentinelCount());
    }
    return Math.toIntExact(total);
  }

  /**
   * The flat, strictly increasing sentinel positions of all images, in prompt order.
   *
   * @return one expanded-ID position per sentinel row
   */
  public int[] sentinelDestinations() {
    int[] destinations = new int[totalSentinelRows()];
    int i = 0;
    for (Phi3VImageDestination image : images) {
      for (int position = image.sentinelStart(); position < image.sentinelEnd(); position++) {
        destinations[i++] = position;
      }
    }
    return destinations;
  }

  private static void verifyDestinations(int expandedLength, List<Phi3VImageDestination> images) {
    int previous = -1;
    for (Phi3VImageDestination image : images) {
      long end = (long) image.sentinelStart() + image.sentinelCount();
      if (end > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "sentinel destination run " + image + " overflows a signed 32-bit integer");
      }
      for (int position = image.sentinelStart(); position < end; position++) {
        if (position <= previous) {
          throw new IllegalArgumentException(
              "sentinel destinations must be strictly increasing: "
                  + position
                  + " after "
                  + previous);
        }
        if (position >= expandedLength) {
          throw new IllegalArgumentException(
              "sentinel destination "
                  + position
                  + " is out of bounds for the expanded prompt ("
                  + expandedLength
                  + " tokens)");
        }
        previous = position;
      }
    }
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof Phi3VPromptPlan that)) {
      return false;
    }
    return renderedText.equals(that.renderedText)
        && Arrays.equals(expandedIds, that.expandedIds)
        && images.equals(that.images);
  }

  @Override
  public int hashCode() {
    return Objects.hash(renderedText, Arrays.hashCode(expandedIds), images);
  }

  @Override
  public String toString() {
    return "Phi3VPromptPlan["
        + "renderedText="
        + renderedText
        + ", expandedIds="
        + Arrays.toString(expandedIds)
        + ", images="
        + images
        + "]";
  }
}
