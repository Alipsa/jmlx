package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Mixture of gated experts with deterministic top-k routing; only the selected experts are
 * evaluated (see {@link SwitchGlu}).
 */
public final class MoeMlp extends UnaryLayer {

  private final UnaryLayer router;
  private final SwitchGlu experts;
  private final int topK;

  /**
   * Registers the router and the stacked experts as children {@code router} and {@code experts}.
   */
  public MoeMlp(MLXScope scope, UnaryLayer router, SwitchGlu experts, int topK) {
    super(scope);
    this.router = child("router", Objects.requireNonNull(router, "router"));
    this.experts = child("experts", Objects.requireNonNull(experts, "experts"));
    if (topK < 1 || topK > experts.experts()) {
      throw new IllegalArgumentException("topK must be between 1 and the number of experts");
    }
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
        || logitShape[2] != experts.experts()) {
      throw new IllegalArgumentException("MoE router must produce [batch, tokens, experts]");
    }

    // Mixtral softmax is evaluated in float32, even with reduced-precision hidden states.
    MLXArray probabilities = MLXOps.softmaxAxis(MLX.astype(logits, DType.FLOAT32), -1, true);
    MLXArray remaining = probabilities;
    MLXArray expertIds = MLX.arange(x.scope(), 0, experts.experts(), 1, DType.INT32);
    MLXArray[] indices = new MLXArray[topK];
    MLXArray[] selectedProbabilities = new MLXArray[topK];
    MLXArray selectedSum = null;
    MLXArray negativeInfinity =
        MLX.full(x.scope(), new int[] {1}, Float.NEGATIVE_INFINITY, DType.FLOAT32);
    for (int i = 0; i < topK; i++) {
      // argmax uses the lowest index on ties. Mask each selected index before the next pass.
      indices[i] = MLXOps.argmaxAxis(remaining, -1, true);
      // gather_axis has no VJP with respect to its indices and the pinned MLX 0.32.4
      // rejects computing one (SwitchGlu's gather_mm guard is the same requirement).
      // Router gradients reach the router through the gathered probabilities, not
      // through the integer indices.
      selectedProbabilities[i] =
          MLXShape.takeAlongAxis(probabilities, MLXOps.stopGradient(indices[i]), -1);
      selectedSum =
          selectedSum == null
              ? selectedProbabilities[i]
              : MLXOps.add(selectedSum, selectedProbabilities[i]);
      remaining = MLXOps.where(MLXOps.equal(expertIds, indices[i]), negativeInfinity, remaining);
    }

    MLXArray[] weights = new MLXArray[topK];
    for (int i = 0; i < topK; i++) {
      weights[i] = MLX.astype(MLXOps.divide(selectedProbabilities[i], selectedSum), x.dtype());
    }
    // [B, T, K]: argmaxAxis(keepdims=true) yields INT32 [B, T, 1] per slot.
    MLXArray selected = MLXShape.concatenate(indices, -1);
    MLXArray slotWeights = MLXShape.concatenate(weights, -1);
    MLXArray expertOutputs = experts.forward(x, selected); // [B, T, K, H]
    return MLXOps.sum(
        MLXOps.multiply(expertOutputs, MLXShape.expandDims(slotWeights, -1)), new int[] {2}, false);
  }
}
