package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXRandom;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Package-private ordered logits policy shared by all decoder architectures. */
final class SamplingPipeline implements AutoCloseable {
  private static final int VOCABULARY_AXIS = 2;

  /**
   * Internal selection whose logits array is owned by the caller's activation scope and is valid
   * only until that scope closes. The array contains penalized logits in greedy mode and
   * vocabulary-ordered post-filter logits in sampled mode.
   */
  record Selection(int tokenId, Double logProbability, MLXArray vocabularyLogits) {}

  private final MLXScope generationScope;
  private final GenerationConfig policy;
  private final int vocabularySize;
  private final boolean filtering;
  private final MLXArray rankIndices;
  private final StepBoundaryEvaluator evaluator;
  private MLXArray currentKey;

  SamplingPipeline(MLXScope generationScope, GenerationConfig policy, int vocabularySize) {
    this(generationScope, policy, vocabularySize, StepBoundaryEvaluator.NATIVE);
  }

  /** Direct-path constructor: builds its own {@code rankIndices} in {@code generationScope}. */
  SamplingPipeline(
      MLXScope generationScope,
      GenerationConfig policy,
      int vocabularySize,
      StepBoundaryEvaluator evaluator) {
    this(
        generationScope,
        policy,
        vocabularySize,
        evaluator,
        needsRankIndices(policy) ? newRankIndices(generationScope, vocabularySize) : null);
  }

  /**
   * Batch constructor: every row of a cohort shares the model's vocabulary size, so the cohort
   * builds <em>one</em> {@code rankIndices} array (see {@link #newRankIndices}) and shares it.
   * {@code sharedRankIndices} must be {@code null} exactly when {@link #needsRankIndices} is false
   * for {@code policy}; its owner closes it, never {@link #close()}.
   */
  SamplingPipeline(
      MLXScope generationScope,
      GenerationConfig policy,
      int vocabularySize,
      StepBoundaryEvaluator evaluator,
      MLXArray sharedRankIndices) {
    this.generationScope = Objects.requireNonNull(generationScope, "generationScope");
    this.policy = Objects.requireNonNull(policy, "policy");
    this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
    if (vocabularySize <= 0) {
      throw new IllegalArgumentException("vocabularySize must be positive");
    }
    if (policy.topK() > vocabularySize) {
      throw new IllegalArgumentException(
          "topK " + policy.topK() + " exceeds vocabulary size " + vocabularySize);
    }
    this.vocabularySize = vocabularySize;
    filtering = policy.topK() != 0 || policy.topP() != 1 || policy.minP() != 0;
    if (needsRankIndices(policy) && sharedRankIndices == null) {
      throw new IllegalArgumentException("policy needs rankIndices but none was supplied");
    }
    rankIndices = needsRankIndices(policy) ? sharedRankIndices : null;
    currentKey =
        policy.temperature() > 0
            ? MLXRandom.key(generationScope, policy.seed().orElseThrow())
            : null;
  }

  /** Whether {@code policy}'s filters index tokens by rank (top-k, or top-p of zero). */
  static boolean needsRankIndices(GenerationConfig policy) {
    boolean filtering = policy.topK() != 0 || policy.topP() != 1 || policy.minP() != 0;
    return filtering && (policy.topK() != 0 || policy.topP() == 0);
  }

  /** The {@code [1,1,V]} rank array every rank-based filter compares against. */
  static MLXArray newRankIndices(MLXScope scope, int vocabularySize) {
    return MLXShape.reshape(
        MLX.arange(scope, 0, vocabularySize, 1, DType.INT32), new int[] {1, 1, vocabularySize});
  }

  Selection select(MLXArray modelLogits, PenaltyInputs penaltyInputs, int decodeStep) {
    return select(modelLogits, penaltyInputs, decodeStep, new MLXArray[0]);
  }

  Selection select(
      MLXArray modelLogits, PenaltyInputs penaltyInputs, int decodeStep, MLXArray... cacheArrays) {
    Built built = build(modelLogits, penaltyInputs);
    evaluate(built, cacheArrays);
    return readBack(built, decodeStep);
  }

  /**
   * One row's lazy selection graph, all of it in the scope of the logits it was built from. Nothing
   * has been evaluated: {@code selected} is the chosen ID, {@code finite} and {@code
   * temperedFinite} are 0/1 INT32 scalars ({@code temperedFinite} is null in greedy mode, which has
   * no tempering step), and {@code logProbability} is non-null only for a sampled row that asked
   * for it (a greedy row's log probability is the constant 0).
   */
  record Built(
      MLXArray selected,
      MLXArray finite,
      MLXArray temperedFinite,
      MLXArray logProbability,
      MLXArray vocabularyLogits) {}

  /** Builds the lazy selection graph for one {@code [1,1,V]} logits row; advances the RNG key. */
  Built build(MLXArray modelLogits, PenaltyInputs penaltyInputs) {
    requireLogits(modelLogits);
    MLXArray logits =
        modelLogits.dtype() == DType.FLOAT32 ? modelLogits : MLX.astype(modelLogits, DType.FLOAT32);
    MLXArray finite = MLX.astype(MLXOps.all(MLXOps.isFinite(logits)), DType.INT32);
    MLXArray adjusted = applyPenalties(logits, penaltyInputs);

    if (policy.temperature() == 0) {
      MLXArray selected = MLXOps.argmaxAxis(adjusted, VOCABULARY_AXIS, false);
      return new Built(selected, finite, null, null, adjusted);
    }

    MLXArray temperature = scalar(logits.scope(), policy.temperature());
    MLXArray tempered = MLXOps.divide(adjusted, temperature);
    MLXArray temperedFinite = MLX.astype(MLXOps.all(MLXOps.isFinite(tempered)), DType.INT32);
    MLXArray vocabularyOrdered = filtering ? applyFilters(tempered) : tempered;
    MLXArray drawKey = advanceKey(logits.scope());
    MLXArray selected = MLXRandom.categorical(vocabularyOrdered, VOCABULARY_AXIS, drawKey);
    MLXArray logNormalizer = MLXOps.logSumExpAxis(vocabularyOrdered, VOCABULARY_AXIS, true);
    MLXArray selectedLogProbability =
        policy.logProbabilities()
            ? selectedLogProbability(vocabularyOrdered, selected, logNormalizer)
            : null;
    return new Built(selected, finite, temperedFinite, selectedLogProbability, vocabularyOrdered);
  }

  /**
   * Evaluates {@code built} together with {@code cacheArrays} in one call through the injected
   * boundary evaluator, so selection and the cache update cost a single synchronization.
   */
  void evaluate(Built built, MLXArray... cacheArrays) {
    if (built.temperedFinite() == null) {
      evalWithCaches(cacheArrays, built.finite(), built.selected());
    } else if (built.logProbability() == null) {
      evalWithCaches(cacheArrays, built.finite(), built.temperedFinite(), built.selected());
    } else {
      evalWithCaches(
          cacheArrays,
          built.finite(),
          built.temperedFinite(),
          built.selected(),
          built.logProbability());
    }
  }

  /** Evaluates the stacked readback arrays and {@code cacheArrays} in one boundary call. */
  static void evaluate(StepBoundaryEvaluator evaluator, Batched batched, MLXArray... cacheArrays) {
    int leading = batched.logProbabilities() == null ? 1 : 2;
    MLXArray[] all = new MLXArray[leading + cacheArrays.length];
    all[0] = batched.ints();
    if (leading == 2) {
      all[1] = batched.logProbabilities();
    }
    System.arraycopy(cacheArrays, 0, all, leading, cacheArrays.length);
    evaluator.evaluate(all);
  }

  /** One row's readback; the flags are only meaningful before {@code tokenId} is trusted. */
  record RowReadBack(int tokenId, boolean finite, boolean temperedFinite, Double logProbability) {}

  /** Reads an already-evaluated {@code built} back, enforcing the finite-logit policy. */
  Selection readBack(Built built, int decodeStep) {
    requireFinite(built.finite(), decodeStep);
    if (built.temperedFinite() == null) {
      return new Selection(
          built.selected().toIntArray()[0],
          policy.logProbabilities() ? 0.0 : null,
          built.vocabularyLogits());
    }
    requireTemperedFinite(built.temperedFinite(), decodeStep);
    int token = built.selected().toIntArray()[0];
    Double logProbability =
        built.logProbability() == null ? null : (double) built.logProbability().toFloatArray()[0];
    return new Selection(token, logProbability, built.vocabularyLogits());
  }

  /**
   * Reads an evaluated {@link Batched} back with one native call per array. {@code
   * logProbabilityRows[row]} says whether that row's policy reports a log probability (the greedy
   * constant 0 included), so rows that did not ask for one get {@code null}.
   */
  static RowReadBack[] readBack(Batched batched, boolean[] logProbabilityRows) {
    int[] ints = batched.ints().toIntArray();
    float[] logProbabilities =
        batched.logProbabilities() == null ? null : batched.logProbabilities().toFloatArray();
    RowReadBack[] rows = new RowReadBack[logProbabilityRows.length];
    for (int row = 0; row < rows.length; row++) {
      Double logProbability =
          !logProbabilityRows[row]
              ? null
              : (logProbabilities == null ? Double.valueOf(0.0) : (double) logProbabilities[row]);
      rows[row] =
          new RowReadBack(
              ints[3 * row], ints[3 * row + 1] != 0, ints[3 * row + 2] != 0, logProbability);
    }
    return rows;
  }

  /**
   * A cohort's per-row {@link Built} results stacked into two lazy arrays so that reading them back
   * does not trigger a second synchronization: {@code ints} is INT32 {@code [B,3]} of selected ID,
   * finite flag and tempered-finite flag (a constant 1 for greedy rows, which have no tempered
   * check); {@code logProbabilities} is FLOAT32 {@code [B]} of post-filter log probabilities, null
   * unless some row has one (rows without one contribute 0).
   */
  record Batched(MLXArray ints, MLXArray logProbabilities) {}

  /** Stacks {@code rows} (in row order) into the two readback arrays, in {@code scope}. */
  static Batched stack(MLXScope scope, java.util.List<Built> rows) {
    MLXArray[] perRow = new MLXArray[rows.size()];
    boolean anyLogProbability = false;
    for (int row = 0; row < perRow.length; row++) {
      Built built = rows.get(row);
      MLXArray tempered =
          built.temperedFinite() == null
              ? MLX.full(scope, new int[] {1, 1}, 1, DType.INT32)
              : MLXShape.reshape(built.temperedFinite(), new int[] {1, 1});
      perRow[row] =
          MLXShape.concatenate(
              new MLXArray[] {
                MLXShape.reshape(built.selected(), new int[] {1, 1}),
                MLXShape.reshape(built.finite(), new int[] {1, 1}),
                tempered
              },
              1);
      anyLogProbability |= built.logProbability() != null;
    }
    MLXArray ints = MLXShape.concatenate(perRow, 0);
    if (!anyLogProbability) {
      return new Batched(ints, null);
    }
    MLXArray[] logProbabilities = new MLXArray[rows.size()];
    for (int row = 0; row < logProbabilities.length; row++) {
      MLXArray logProbability = rows.get(row).logProbability();
      logProbabilities[row] =
          logProbability == null
              ? MLX.full(scope, new int[] {1}, 0f, DType.FLOAT32)
              : MLXShape.reshape(logProbability, new int[] {1});
    }
    return new Batched(ints, MLXShape.concatenate(logProbabilities, 0));
  }

  /** The direct path's finite-logit failure, reused so batched rows report identical text. */
  static IllegalStateException nonFiniteLogits(int decodeStep) {
    return new IllegalStateException(
        "sampling stage finite-logit validation failed at decode step "
            + decodeStep
            + ": logits must be finite");
  }

  /** The direct path's tempering failure, reused so batched rows report identical text. */
  static IllegalStateException nonFiniteTempered(int decodeStep) {
    return new IllegalStateException(
        "sampling stage temperature scaling failed at decode step "
            + decodeStep
            + ": tempered logits must be finite");
  }

  private void evalWithCaches(MLXArray[] cacheArrays, MLXArray... selectionArrays) {
    MLXArray[] all = Arrays.copyOf(selectionArrays, selectionArrays.length + cacheArrays.length);
    System.arraycopy(cacheArrays, 0, all, selectionArrays.length, cacheArrays.length);
    evaluator.evaluate(all);
  }

  private MLXArray applyFilters(MLXArray tempered) {
    MLXArray sortedTokenIds = MLXOps.argsortAxis(MLXOps.negative(tempered), VOCABULARY_AXIS);
    MLXArray filtered = MLXShape.takeAlongAxis(tempered, sortedTokenIds, VOCABULARY_AXIS);
    MLXArray negativeInfinity =
        MLX.full(tempered.scope(), new int[] {1}, Float.NEGATIVE_INFINITY, DType.FLOAT32);
    filtered = applyTopK(filtered, negativeInfinity);
    filtered = applyTopP(filtered, negativeInfinity);
    filtered = applyMinP(filtered, negativeInfinity);
    return MLXShape.putAlongAxis(
        MLX.zeros(tempered.scope(), tempered.shape(), DType.FLOAT32),
        sortedTokenIds,
        filtered,
        VOCABULARY_AXIS);
  }

  private MLXArray applyPenalties(MLXArray logits, PenaltyInputs inputs) {
    int[] ids = inputs.rawTokenIds();
    if (ids.length == 0
        || (policy.repetitionPenalty() == 1
            && policy.frequencyPenalty() == 0
            && policy.presencePenalty() == 0)) {
      return logits;
    }
    for (int id : ids) {
      if (id < 0 || id >= vocabularySize) {
        throw new IllegalArgumentException(
            "history token ID " + id + " outside vocabulary [0, " + vocabularySize + ")");
      }
    }
    int[] indexShape = {1, 1, ids.length};
    MLXArray indices = MLX.array(logits.scope(), ids, indexShape);
    MLXArray gathered = MLXShape.takeAlongAxis(logits, indices, VOCABULARY_AXIS);
    MLXArray adjusted = gathered;
    if (policy.repetitionPenalty() != 1) {
      MLXArray zero = scalar(logits.scope(), 0);
      MLXArray penalty = scalar(logits.scope(), policy.repetitionPenalty());
      adjusted =
          MLXOps.where(
              MLXOps.greaterEqual(adjusted, zero),
              MLXOps.divide(adjusted, penalty),
              MLXOps.multiply(adjusted, penalty));
    }
    if (policy.frequencyPenalty() != 0) {
      MLXArray counts = MLX.array(logits.scope(), inputs.rawCounts(), indexShape);
      adjusted =
          MLXOps.subtract(
              adjusted, MLXOps.multiply(counts, scalar(logits.scope(), policy.frequencyPenalty())));
    }
    if (policy.presencePenalty() != 0) {
      adjusted = MLXOps.subtract(adjusted, scalar(logits.scope(), policy.presencePenalty()));
    }
    return MLXShape.putAlongAxis(logits, indices, adjusted, VOCABULARY_AXIS);
  }

  private MLXArray applyTopK(MLXArray sorted, MLXArray negativeInfinity) {
    if (policy.topK() == 0) {
      return sorted;
    }
    MLXArray keep =
        MLXOps.less(
            rankIndices, MLX.full(sorted.scope(), new int[] {1}, policy.topK(), DType.INT32));
    return MLXOps.where(keep, sorted, negativeInfinity);
  }

  private MLXArray applyTopP(MLXArray sorted, MLXArray negativeInfinity) {
    if (policy.topP() == 1) {
      return sorted;
    }
    if (policy.topP() == 0) {
      MLXArray first =
          MLXOps.less(rankIndices, MLX.full(sorted.scope(), new int[] {1}, 1, DType.INT32));
      return MLXOps.where(first, sorted, negativeInfinity);
    }
    MLXArray probabilities = MLXOps.softmaxAxis(sorted, VOCABULARY_AXIS, true);
    MLXArray cumulative = MLXOps.cumulativeSumAxis(probabilities, VOCABULARY_AXIS, false, true);
    MLXArray previous = MLXOps.subtract(cumulative, probabilities);
    MLXArray keep = MLXOps.less(previous, scalar(sorted.scope(), policy.topP()));
    return MLXOps.where(keep, sorted, negativeInfinity);
  }

  private MLXArray applyMinP(MLXArray sorted, MLXArray negativeInfinity) {
    if (policy.minP() == 0) {
      return sorted;
    }
    MLXArray probabilities = MLXOps.softmaxAxis(sorted, VOCABULARY_AXIS, true);
    MLXArray maximum = MLXShape.slice(probabilities, new int[] {0, 0, 0}, new int[] {1, 1, 1});
    MLXArray threshold = MLXOps.multiply(maximum, scalar(sorted.scope(), policy.minP()));
    return MLXOps.where(MLXOps.greaterEqual(probabilities, threshold), sorted, negativeInfinity);
  }

  private MLXArray advanceKey(MLXScope activationScope) {
    MLXArray split = MLXRandom.split(currentKey, 2, activationScope);
    MLXArray successorView =
        MLXShape.squeeze(MLXShape.slice(split, new int[] {0, 0}, new int[] {1, 2}), new int[] {0});
    MLXArray drawKey =
        MLXShape.squeeze(MLXShape.slice(split, new int[] {1, 0}, new int[] {2, 2}), new int[] {0});
    MLXArray successor = MLX.hoist(successorView, generationScope);
    currentKey.close();
    currentKey = successor;
    return drawKey;
  }

  private static MLXArray selectedLogProbability(
      MLXArray logits, MLXArray selected, MLXArray logNormalizer) {
    MLXArray index = MLXShape.expandDims(selected, VOCABULARY_AXIS);
    MLXArray selectedLogit = MLXShape.takeAlongAxis(logits, index, VOCABULARY_AXIS);
    return MLXOps.subtract(selectedLogit, logNormalizer);
  }

  private static MLXArray scalar(MLXScope scope, float value) {
    return MLX.full(scope, new int[] {1}, value, DType.FLOAT32);
  }

  private void requireLogits(MLXArray logits) {
    int[] shape = Objects.requireNonNull(logits, "modelLogits").shape();
    if (!Arrays.equals(shape, new int[] {1, 1, vocabularySize})) {
      throw new IllegalArgumentException(
          "sampling logits must have shape [1, 1, "
              + vocabularySize
              + "], got "
              + Arrays.toString(shape));
    }
  }

  private static void requireFinite(MLXArray finite, int decodeStep) {
    if (finite.toIntArray()[0] == 0) {
      throw nonFiniteLogits(decodeStep);
    }
  }

  private static void requireTemperedFinite(MLXArray finite, int decodeStep) {
    if (finite.toIntArray()[0] == 0) {
      throw nonFiniteTempered(decodeStep);
    }
  }

  @Override
  public void close() {
    if (currentKey != null) {
      currentKey.close();
      currentKey = null;
    }
  }
}
