package se.alipsa.jmlx.models;

import java.util.Arrays;
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
 * primitive is required. Both compositions reuse {@link Idefics3VisionTower#
 * mergeGatherIndices(int, java.util.List)} — the shared gather-index algorithm, where feature row
 * {@code f} overwrites exactly the {@code f}-th destination and every other row reads itself. The
 * row-gather replacement applies it directly; the masked add remaps its output over a source of a
 * single zero row followed by the feature rows: every identity row (shared index {@code i < L})
 * remaps to the zero row (source index {@code 0}) and the {@code f}-th destination (shared index
 * {@code L + f}) remaps to source row {@code 1 + f}, which is what lets its zero pad be one {@code
 * [1, 1, hidden]} row instead of a full {@code [1, L, hidden]} zero tensor: a full-sequence zero
 * allocation plus an {@code (L + F)}-wide concatenation per DeepStack level (three at prefill)
 * would otherwise materialise for values that are exactly unchanged. All validation is checked
 * before any MLX array is written, and every non-destination row comes back unchanged: the
 * replacement leaves its untouched rows bit-identical, and the masked add is an exact {@code + 0.0}
 * per element there, so its non-destination rows come back numerically identical (note IEEE {@code
 * -0.0 + 0.0 = +0.0}, so a negative-zero element in a non-destination row is normalised to {@code
 * +0.0}). The destinations follow the existing semantics: a flat, strictly increasing list in
 * {@code [0, L)} whose runs are the images' contiguous placeholder ranges in prompt order. WP6's
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
   * Masked residual add: the frozen {@code zeroRow = MLX.zeros(scope, {1, 1, hidden},
   * hidden.dtype()); source = concatenate([zeroRow, features], 1); mask = takeAxis(source, gather,
   * 1); updated = MLXOps.add(hidden, mask)} composition that adds one DeepStack level into exactly
   * the destination rows, leaving every other row numerically identical through the exact {@code +
   * 0.0} per element (note IEEE {@code -0.0 + 0.0 = +0.0}, so a negative-zero element in an
   * untouched row is normalised to {@code +0.0}). The gather indices are the shared {@code
   * mergeGatherIndices} output remapped over the source's {@code 1 + F} rows: every identity row
   * (shared index {@code i < L}) remaps to the zero row (source index {@code 0}) and the {@code
   * f}-th destination (shared index {@code L + f}) remaps to source row {@code 1 + f}, which lets
   * the zero pad be a single {@code [1, 1, hidden]} row instead of a full-sequence {@code [1, L,
   * hidden]} zero tensor.
   *
   * @param scope the scope the zero pad and gather indices are allocated in; the results land per
   *     the standard scope rule (innermost scope among the operands); like the shapes, dtypes, row
   *     count and destinations, its relation to the operands' scopes (one an ancestor of the other)
   *     is validated before any MLX array is written
   * @param hidden the current {@code [1, L, hidden]} hidden states
   * @param features the {@code [1, F, hidden]} DeepStack level rows, in destination order
   * @param destinations the image-token rows to add into, strictly increasing in {@code [0, L)},
   *     whose runs are the images' contiguous placeholder ranges in prompt order
   * @return the {@code [1, L, hidden]} hidden states with the level added into the destination rows
   * @throws IllegalArgumentException if the shapes, dtypes, row count or destinations are invalid,
   *     or any of {@code scope} and the operands' scopes is unrelated (neither an ancestor of the
   *     other), in every case before any MLX array is written
   */
  static MLXArray addMaskedFeatures(
      MLXScope scope, MLXArray hidden, MLXArray features, int[] destinations) {
    if (scope == null) {
      throw new IllegalArgumentException("scope must not be null");
    }
    int length = validatePair("hidden", hidden, "features", features, destinations);
    checkRelatedScopes("scope", scope, hidden, features);
    int[] gather = gatherIndices(length, destinations);
    // Remap the shared indices: identity row i -> the zero row (0), destination L + f -> 1 + f.
    for (int i = 0; i < gather.length; i++) {
      gather[i] = gather[i] >= length ? gather[i] - length + 1 : 0;
    }
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
   * The masked add's pre-write scope-compatibility check, applying {@link MLXScope#innermost} — the
   * same relatedness test the ops' own {@code scopeOf} resolution applies — to the three pairs the
   * method's chain requires to be related (one an ancestor of the other): {@code scope} against
   * each operand (the final {@code add} resolves against {@code hidden}, {@code concatenate}
   * against {@code features}), and the two operand scopes against each other. The operand-vs.
   * operand pair is not implied by the other two: when {@code scope} is a common ancestor of both
   * operand scopes, sibling operand scopes pass both operand-vs.-scope checks yet fail at the final
   * {@code add}, whose mask lives in the deeper of {@code scope} and {@code features.scope()} —
   * after the zero row was already allocated, breaking the pre-write validation promise.
   *
   * @throws IllegalArgumentException naming {@code paramName} if any of the three pairs is
   *     unrelated
   */
  private static void checkRelatedScopes(
      String paramName, MLXScope paramScope, MLXArray base, MLXArray features) {
    try {
      MLXScope.innermost(paramScope, base.scope());
      MLXScope.innermost(paramScope, features.scope());
      MLXScope.innermost(base.scope(), features.scope());
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          paramName
              + " must be related to hidden's and features' scopes (one an ancestor of the other)",
          e);
    }
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
