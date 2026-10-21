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
 * primitive is required. The row-gather replacement reuses {@link Idefics3VisionTower#
 * mergeGatherIndices(int, java.util.List)} — feature row {@code f} overwrites exactly the {@code
 * f}-th destination and every other row reads itself. The masked add instead uses a dedicated index
 * builder whose non-destination rows all read the single zero row at the front of the source, which
 * is what lets its zero pad be one {@code [1, 1, hidden]} row instead of a full {@code [1, L,
 * hidden]} zero tensor: a full-sequence zero allocation plus an {@code (L + F)}-wide concatenation
 * per DeepStack level (three at prefill) would otherwise materialise for values that are exactly
 * unchanged. All validation is checked before any MLX array is written, and every non-destination
 * row comes back unchanged: the replacement leaves its untouched rows bit-identical, and the masked
 * add is an exact {@code + 0.0} per element there, so its non-destination rows come back
 * numerically identical (note IEEE {@code -0.0 + 0.0 = +0.0}, so a negative-zero element in a
 * non-destination row is normalised to {@code +0.0}). The destinations follow the existing
 * semantics: a flat, strictly increasing list in {@code [0, L)} whose runs are the images'
 * contiguous placeholder ranges in prompt order. WP6's {@code Qwen3VlModel} is the consumer; the
 * class is package-private on purpose, with no new public API.
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
   * Masked residual add: the frozen {@code zeroRow = MLX.zeros(scope, {1, 1, hidden},
   * hidden.dtype()); source = concatenate([zeroRow, features], 1); mask = takeAxis(source, gather,
   * 1); updated = MLXOps.add(hidden, mask)} composition that adds one DeepStack level into exactly
   * the destination rows, leaving every other row numerically identical through the exact {@code +
   * 0.0} per element (note IEEE {@code -0.0 + 0.0 = +0.0}, so a negative-zero element in an
   * untouched row is normalised to {@code +0.0}). The dedicated gather index of the source's {@code
   * 1 + F} rows maps every non-destination row to the zero row (source index {@code 0}) and the
   * {@code f}-th destination to source row {@code 1 + f}, which lets the zero pad be a single
   * {@code [1, 1, hidden]} row instead of a full-sequence {@code [1, L, hidden]} zero tensor.
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
    int[] gather = maskedAddGatherIndices(length, features.shape()[1], destinations);
    MLXArray zeroRow = MLX.zeros(scope, new int[] {1, 1, hidden.shape()[2]}, hidden.dtype());
    MLXArray source = MLXShape.concatenate(new MLXArray[] {zeroRow, features}, 1);
    MLXArray mask =
        MLXShape.takeAxis(source, MLX.array(scope, gather, new int[] {gather.length}), 1);
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

  /**
   * The masked add's dedicated gather indices: the {@code length}-index array into a source of
   * {@code 1 + featureCount} rows — one zero row followed by the feature rows — where every
   * non-destination row reads the zero row (source index {@code 0}) and the {@code f}-th
   * destination reads source row {@code 1 + f}. It validates the destinations exactly as {@link
   * Idefics3VisionTower#mergeGatherIndices(int, java.util.List)} does (strictly increasing in
   * {@code [0, length)}) and checks the {@code 1 + featureCount} source-row sum with checked
   * arithmetic, where the shared helper checks the {@code length + featureCount} sum instead.
   */
  private static int[] maskedAddGatherIndices(int length, int featureCount, int[] destinations) {
    Objects.requireNonNull(destinations, "destinations");
    if (length < 0) {
      throw new IllegalArgumentException("length must be non-negative");
    }
    if (destinations.length != featureCount) {
      throw new IllegalArgumentException(
          "featureCount "
              + featureCount
              + " must equal the destination count "
              + destinations.length);
    }
    int previous = -1;
    for (int f = 0; f < featureCount; f++) {
      int destination = destinations[f];
      if (destination < 0 || destination >= length) {
        throw new IllegalArgumentException(
            "feature destination " + destination + " is out of range [0, " + length + ")");
      }
      if (destination <= previous) {
        throw new IllegalArgumentException(
            "feature destinations must be strictly increasing; index "
                + f
                + " has "
                + destination
                + " after "
                + previous);
      }
      previous = destination;
    }
    Math.addExact(1, featureCount);
    int[] indices = new int[length];
    int next = 0;
    for (int i = 0; i < length; i++) {
      if (next < featureCount && destinations[next] == i) {
        indices[i] = 1 + next;
        next++;
      } else {
        indices[i] = 0;
      }
    }
    return indices;
  }

  private static int validatePair(
      String baseName, MLXArray base, String featuresName, MLXArray features, int[] destinations) {
    if (base == null) {
      throw new IllegalArgumentException(baseName + " must not be null");
    }
    if (features == null) {
      throw new IllegalArgumentException(featuresName + " must not be null");
    }
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
