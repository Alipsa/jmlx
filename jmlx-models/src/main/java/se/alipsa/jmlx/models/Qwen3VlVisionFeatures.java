package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;

/**
 * The full output of {@link Qwen3VlVisionTower#forward(List)}: the main-merger rows of every input
 * image concatenated on axis 0 ({@code merged}), and, for each DeepStack level in the tower's
 * {@code deepstackIndexes} order, that level's rows of every input image concatenated the same way
 * ({@code deepstackLevels}).
 *
 * <p>{@code merged} and every level have shape {@code [totalMergedRows, outHiddenSize]} where
 * {@code totalMergedRows} is the sum of the input images' post-merge row counts, in input order.
 * The list is an immutable copy: the caller's list is not retained, and per-image arrays handed to
 * the tower are hoisted into the tower's own scope, so the returned arrays outlive each image's
 * temporary scope.
 */
public record Qwen3VlVisionFeatures(MLXArray merged, List<MLXArray> deepstackLevels) {

  /**
   * Compact constructor: requires both components and replaces {@code deepstackLevels} with a
   * defensive copy so the caller's list is never retained.
   */
  public Qwen3VlVisionFeatures {
    Objects.requireNonNull(merged, "merged");
    deepstackLevels = List.copyOf(Objects.requireNonNull(deepstackLevels, "deepstackLevels"));
  }
}
