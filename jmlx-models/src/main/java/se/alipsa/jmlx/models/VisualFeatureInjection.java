package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Frozen WP4 visual-injection compositions for the Qwen3-VL text decoder (req/plans/
 * phase7-3b-reference-findings.md §6.9): the row-gather replacement that swaps the
 * image-placeholder embeddings for the merged visual features, and the masked residual add that
 * injects one DeepStack level into the hidden states at exactly the image-token rows (its add
 * matches the pinned reference's masked clone + add + write-back, §6.6).
 *
 * <p>Both methods are pure compositions of existing core facades ({@code MLX.zeros}, {@code
 * MLXShape.concatenate}, {@code MLXShape.takeAxis}, {@code MLXOps.add}) and never call FFI; no new
 * primitive is required. They reuse {@link Idefics3VisionTower#mergeGatherIndices(int,
 * java.util.List)} as the single shared gather-index algorithm — feature row {@code f} overwrites
 * exactly the {@code f}-th destination and every other row reads itself — so one helper serves both
 * compositions. All validation is checked before any MLX array is written, and every
 * non-destination row comes back bit-identical (the masked add is an exact {@code + 0.0} per
 * element there). The destinations follow the existing semantics: a flat, strictly increasing list
 * in {@code [0, L)} whose runs are the images' contiguous placeholder ranges in prompt order. WP6's
 * {@code Qwen3VlModel} is the consumer; the class is package-private on purpose, with no new public
 * API.
 */
final class VisualFeatureInjection {

  private VisualFeatureInjection() {}

  /**
   * Row-gather replacement: the frozen {@code source = concatenate([embedded, features], 1); gather
   * = mergeGatherIndices(L, destinations); replaced = takeAxis(source, gather, 1)} composition that
   * replaces each destination row with the corresponding feature row and leaves every other row
   * untouched, the same composition {@code SmolVlmModel.merge} uses.
   *
   * @param embedded the original {@code [1, L, hidden]} token embeddings
   * @param features the {@code [1, F, hidden]} merged visual feature rows, one block per image in
   *     prompt order
   * @param destinations the image-token rows to replace, strictly increasing in {@code [0, L)},
   *     whose runs are the images' contiguous placeholder ranges in prompt order
   * @return the {@code [1, L, hidden]} embeddings with the destination rows replaced
   * @throws IllegalArgumentException if the shapes, dtypes, row count or destinations are invalid,
   *     before any MLX array is written
   */
  static MLXArray replaceImageRows(MLXArray embedded, MLXArray features, int[] destinations) {
    int length = validatePair("embedded", embedded, "features", features, destinations);
    int[] gather = gatherIndices(length, destinations);
    MLXArray source = MLXShape.concatenate(new MLXArray[] {embedded, features}, 1);
    return MLXShape.takeAxis(
        source, MLX.array(embedded.scope(), gather, new int[] {gather.length}), 1);
  }

  /**
   * Masked residual add: the frozen {@code zero = MLX.zeros(scope, hidden.shape(), hidden.dtype());
   * source = concatenate([zero, features], 1); mask = takeAxis(source, gather, 1); updated =
   * MLXOps.add(hidden, mask)} composition that adds one DeepStack level into exactly the
   * destination rows, leaving every other row bit-identical through the exact {@code + 0.0} per
   * element.
   *
   * @param scope the scope the zero pad and gather indices are allocated in; the results land per
   *     the standard scope rule (innermost scope among the operands)
   * @param hidden the current {@code [1, L, hidden]} hidden states
   * @param features the {@code [1, F, hidden]} DeepStack level rows, in destination order
   * @param destinations the image-token rows to add into, strictly increasing in {@code [0, L)},
   *     whose runs are the images' contiguous placeholder ranges in prompt order
   * @return the {@code [1, L, hidden]} hidden states with the level added into the destination rows
   * @throws IllegalArgumentException if the shapes, dtypes, row count or destinations are invalid,
   *     before any MLX array is written
   */
  static MLXArray addMaskedFeatures(
      MLXScope scope, MLXArray hidden, MLXArray features, int[] destinations) {
    if (scope == null) {
      throw new IllegalArgumentException("scope must not be null");
    }
    int length = validatePair("hidden", hidden, "features", features, destinations);
    int[] gather = gatherIndices(length, destinations);
    MLXArray zero = MLX.zeros(scope, hidden.shape(), hidden.dtype());
    MLXArray source = MLXShape.concatenate(new MLXArray[] {zero, features}, 1);
    MLXArray mask =
        MLXShape.takeAxis(source, MLX.array(zero.scope(), gather, new int[] {gather.length}), 1);
    return MLXOps.add(hidden, mask);
  }

  /**
   * The shared gather indices: the existing {@code Idefics3VisionTower} algorithm, which validates
   * the destinations (strictly increasing in {@code [0, length)}), checks the {@code length + F}
   * sum with checked arithmetic and returns the identity row mapping with each destination pointing
   * at its feature row.
   */
  private static int[] gatherIndices(int length, int[] destinations) {
    return Idefics3VisionTower.mergeGatherIndices(
        length, IntStream.of(destinations).boxed().collect(Collectors.toList()));
  }

  private static int validatePair(
      String baseName, MLXArray base, String featuresName, MLXArray features, int[] destinations) {
    Objects.requireNonNull(base, baseName);
    Objects.requireNonNull(features, featuresName);
    if (destinations == null) {
      throw new IllegalArgumentException("destinations must not be null");
    }
    if (base.ndim() != 3 || base.shape()[0] != 1) {
      throw new IllegalArgumentException(
          baseName + " must be a [1, L, hidden] array, got " + Arrays.toString(base.shape()));
    }
    int length = base.shape()[1];
    if (features.ndim() != 3
        || features.shape()[0] != 1
        || features.shape()[2] != base.shape()[2]
        || features.dtype() != base.dtype()) {
      throw new IllegalArgumentException(
          featuresName
              + " must be a [1, F, "
              + base.shape()[2]
              + "] array of dtype "
              + base.dtype()
              + " to match "
              + baseName
              + ", got "
              + Arrays.toString(features.shape())
              + " of dtype "
              + features.dtype());
    }
    if (features.shape()[1] != destinations.length) {
      throw new IllegalArgumentException(
          featuresName
              + " row count "
              + features.shape()[1]
              + " must equal the destination count "
              + destinations.length);
    }
    return length;
  }
}
