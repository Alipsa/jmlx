package se.alipsa.jmlx.nn;

import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Test oracle: the Phase 6.3 dense mixture of experts, which runs every expert on every token and
 * masks unselected outputs with {@code where}. Kept verbatim from the pre-gather {@code MoeMlp} so
 * the gathered implementation is checked against an independent, simpler algorithm (routing,
 * tie-breaking and renormalisation must agree exactly).
 */
final class DenseMoeReference extends UnaryLayer {

  private final UnaryLayer router;
  private final List<GatedMlp> experts;
  private final int topK;

  /** Registers the router and experts as children, in checkpoint order. */
  public DenseMoeReference(MLXScope scope, UnaryLayer router, List<GatedMlp> experts, int topK) {
    super(scope);
    this.router = child("router", Objects.requireNonNull(router, "router"));
    List<GatedMlp> copy = List.copyOf(Objects.requireNonNull(experts, "experts"));
    if (topK < 1 || topK > copy.size()) {
      throw new IllegalArgumentException("topK must be between 1 and the number of experts");
    }
    for (int i = 0; i < copy.size(); i++) {
      child("expert" + i, copy.get(i));
    }
    this.experts = copy;
    this.topK = topK;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    Objects.requireNonNull(x, "x");
    int[] inputShape = x.shape();
    if (inputShape.length != 3) {
      throw new IllegalArgumentException("MoE input must have shape [batch, tokens, hidden]");
    }
    MLXArray logits = router.forward(x);
    int[] logitShape = logits.shape();
    if (logitShape.length != 3
        || logitShape[0] != inputShape[0]
        || logitShape[1] != inputShape[1]
        || logitShape[2] != experts.size()) {
      throw new IllegalArgumentException("MoE router must produce [batch, tokens, experts]");
    }

    // Mixtral softmax is evaluated in float32, even with reduced-precision hidden states.
    MLXArray probabilities = MLXOps.softmaxAxis(MLX.astype(logits, DType.FLOAT32), -1, true);
    MLXArray remaining = probabilities;
    MLXArray expertIds = MLX.arange(x.scope(), 0, experts.size(), 1, DType.INT32);
    MLXArray[] indices = new MLXArray[topK];
    MLXArray[] selectedProbabilities = new MLXArray[topK];
    MLXArray selectedSum = null;
    MLXArray negativeInfinity =
        MLX.full(x.scope(), new int[] {1}, Float.NEGATIVE_INFINITY, DType.FLOAT32);
    for (int i = 0; i < topK; i++) {
      // argmax uses the lowest index on ties. Mask each selected index before the next pass.
      indices[i] = MLXOps.argmaxAxis(remaining, -1, true);
      selectedProbabilities[i] = MLXShape.takeAlongAxis(probabilities, indices[i], -1);
      selectedSum =
          selectedSum == null
              ? selectedProbabilities[i]
              : MLXOps.add(selectedSum, selectedProbabilities[i]);
      remaining = MLXOps.where(MLXOps.equal(expertIds, indices[i]), negativeInfinity, remaining);
    }

    MLXArray result = MLX.zeros(x.scope(), inputShape, x.dtype());
    MLXArray zero = MLX.zeros(x.scope(), new int[] {1}, x.dtype());
    MLXArray[] weights = new MLXArray[topK];
    for (int i = 0; i < topK; i++) {
      weights[i] = MLX.astype(MLXOps.divide(selectedProbabilities[i], selectedSum), x.dtype());
    }
    for (int expert = 0; expert < experts.size(); expert++) {
      MLXArray expertOutput = experts.get(expert).forward(x);
      MLXArray expertId = MLX.full(x.scope(), new int[] {1}, expert, DType.INT32);
      for (int i = 0; i < topK; i++) {
        MLXArray chosen = MLXOps.equal(indices[i], expertId);
        // where runs before multiplication so an unselected infinite expert cannot poison output.
        MLXArray masked = MLXOps.where(chosen, expertOutput, zero);
        result = MLXOps.add(result, MLXOps.multiply(masked, weights[i]));
      }
    }
    return result;
  }
}
