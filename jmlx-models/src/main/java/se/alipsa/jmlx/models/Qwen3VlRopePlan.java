package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.Objects;

/**
 * The three-axis prefill position data one expanded Qwen3-VL prompt plans: the temporal (T), height
 * (H) and width (W) RoPE coordinate axes, each as long as the expanded prompt, and the post-prefill
 * decode offset. In a text-only prompt all three axes coincide with the plain token index; after an
 * image block the axes diverge (T saturates at the block's shared coordinate, H and W cycle through
 * the LLM grid) and the delta goes negative. The result of {@link Qwen3VlPositionPlanner}; consumed
 * once by the expander into {@link Qwen3VlPromptPlan}. Equality is by value: the axes elementwise
 * and the delta numerically.
 *
 * @param temporalPositions the temporal (T) axis, one coordinate per expanded token
 * @param heightPositions the height (H) axis, one coordinate per expanded token
 * @param widthPositions the width (W) axis, one coordinate per expanded token
 * @param delta the decode offset: the largest prefill coordinate plus one minus the expanded length
 *     (negative for image prompts, zero for text-only ones)
 */
record Qwen3VlRopePlan(
    int[] temporalPositions, int[] heightPositions, int[] widthPositions, int delta) {

  public Qwen3VlRopePlan {
    temporalPositions = Objects.requireNonNull(temporalPositions, "temporalPositions").clone();
    heightPositions = Objects.requireNonNull(heightPositions, "heightPositions").clone();
    widthPositions = Objects.requireNonNull(widthPositions, "widthPositions").clone();
    if (heightPositions.length != temporalPositions.length
        || widthPositions.length != temporalPositions.length) {
      throw new IllegalArgumentException(
          "the three position axes must be equally long: "
              + temporalPositions.length
              + ", "
              + heightPositions.length
              + ", "
              + widthPositions.length);
    }
    requireNonNegative("temporalPositions", temporalPositions);
    requireNonNegative("heightPositions", heightPositions);
    requireNonNegative("widthPositions", widthPositions);
  }

  private static void requireNonNegative(String name, int[] positions) {
    for (int position : positions) {
      if (position < 0) {
        throw new IllegalArgumentException(name + " positions must be non-negative: " + position);
      }
    }
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof Qwen3VlRopePlan that)) {
      return false;
    }
    return delta == that.delta
        && Arrays.equals(temporalPositions, that.temporalPositions)
        && Arrays.equals(heightPositions, that.heightPositions)
        && Arrays.equals(widthPositions, that.widthPositions);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        Arrays.hashCode(temporalPositions),
        Arrays.hashCode(heightPositions),
        Arrays.hashCode(widthPositions),
        delta);
  }

  @Override
  public String toString() {
    return "Qwen3VlRopePlan["
        + "temporalPositions="
        + Arrays.toString(temporalPositions)
        + ", heightPositions="
        + Arrays.toString(heightPositions)
        + ", widthPositions="
        + Arrays.toString(widthPositions)
        + ", delta="
        + delta
        + "]";
  }
}
