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

  record Selection(int tokenId, Double logProbability, MLXArray vocabularyLogits) {}

  private final MLXScope generationScope;
  private final GenerationConfig policy;
  private final int vocabularySize;
  private final boolean filtering;
  private final MLXArray rankIndices;
  private MLXArray currentKey;

  SamplingPipeline(MLXScope generationScope, GenerationConfig policy, int vocabularySize) {
    this.generationScope = Objects.requireNonNull(generationScope, "generationScope");
    this.policy = Objects.requireNonNull(policy, "policy");
    if (vocabularySize <= 0) {
      throw new IllegalArgumentException("vocabularySize must be positive");
    }
    if (policy.topK() > vocabularySize) {
      throw new IllegalArgumentException(
          "topK " + policy.topK() + " exceeds vocabulary size " + vocabularySize);
    }
    this.vocabularySize = vocabularySize;
    filtering = policy.topK() != 0 || policy.topP() != 1 || policy.minP() != 0;
    rankIndices =
        filtering && (policy.topK() != 0 || policy.topP() == 0)
            ? MLXShape.reshape(
                MLX.arange(generationScope, 0, vocabularySize, 1, DType.INT32),
                new int[] {1, 1, vocabularySize})
            : null;
    currentKey =
        policy.temperature() > 0
            ? MLXRandom.key(generationScope, policy.seed().orElseThrow())
            : null;
  }

  Selection select(MLXArray modelLogits, PenaltyInputs penaltyInputs, int decodeStep) {
    requireLogits(modelLogits);
    MLXArray logits =
        modelLogits.dtype() == DType.FLOAT32 ? modelLogits : MLX.astype(modelLogits, DType.FLOAT32);
    MLXArray finite = MLX.astype(MLXOps.all(MLXOps.isFinite(logits)), DType.INT32);
    MLXArray adjusted = applyPenalties(logits, penaltyInputs);

    if (policy.temperature() == 0) {
      MLXArray selected = MLXOps.argmaxAxis(adjusted, VOCABULARY_AXIS, false);
      MLX.eval(finite, selected);
      requireFinite(finite, decodeStep);
      return new Selection(
          selected.toIntArray()[0], policy.logProbabilities() ? 0.0 : null, adjusted);
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

    if (selectedLogProbability == null) {
      MLX.eval(finite, temperedFinite, selected);
    } else {
      MLX.eval(finite, temperedFinite, selected, selectedLogProbability);
    }
    requireFinite(finite, decodeStep);
    requireTemperedFinite(temperedFinite, decodeStep);
    int token = selected.toIntArray()[0];
    Double logProbability =
        selectedLogProbability == null ? null : (double) selectedLogProbability.toFloatArray()[0];
    return new Selection(token, logProbability, vocabularyOrdered);
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
      throw new IllegalStateException(
          "sampling stage finite-logit validation failed at decode step "
              + decodeStep
              + ": logits must be finite");
    }
  }

  private static void requireTemperedFinite(MLXArray finite, int decodeStep) {
    if (finite.toIntArray()[0] == 0) {
      throw new IllegalStateException(
          "sampling stage temperature scaling failed at decode step "
              + decodeStep
              + ": tempered logits must be finite");
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
