package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * A fully-connected (affine) layer: {@code y = x @ weight.T + bias}. {@code weight} is registered and stored in the
 * checkpoint's {@code [out, in]} layout -- never transposed before registration; {@link #forward} transposes a fresh
 * view per call via {@link MLXShape#transpose(MLXArray, MLXScope)}'s explicit-target overload, landing it in
 * {@code x.scope()} (the step scope), not {@code weight}'s own (model) scope. This is req/phase4-plan.md §2's
 * withdrawn-cache mitigation record -- do not reintroduce a cached {@code W.T} field.
 */
public final class Linear extends Module implements UnaryModule {

  private final boolean hasBias;

  /**
   * Creates a {@code Linear} layer with the given {@code weight} (shape {@code [out, in]}) and, optionally,
   * {@code bias} (shape {@code [out]}, or {@code null} for no bias).
   */
  public Linear(MLXScope scope, MLXArray weight, MLXArray bias) {
    super(scope);
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException(
          "Linear: weight must be rank 2 [out, in], got shape " + java.util.Arrays.toString(weight.shape()));
    }
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException("Linear: bias must be rank 1 with length weight.shape()[0]="
          + weight.shape()[0] + ", got shape " + java.util.Arrays.toString(bias.shape()));
    }
    param("weight", weight);
    hasBias = bias != null;
    if (hasBias) {
      param("bias", bias);
    }
  }

  @Override
  public MLXArray forward(MLXArray x) {
    MLXArray weightT = MLXShape.transpose(param("weight"), x.scope());
    MLXArray y = MLXOps.matmul(x, weightT);
    return hasBias ? MLXOps.add(y, param("bias")) : y;
  }
}
