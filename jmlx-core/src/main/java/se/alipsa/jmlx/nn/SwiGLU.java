package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** The decoder MLP used by Llama/Qwen: {@code down(silu(gate(x)) * up(x))}. */
public final class SwiGLU extends Module implements UnaryModule {

  private final Linear gateProj;
  private final Linear upProj;
  private final Linear downProj;

  /** Creates the MLP from checkpoint-layout projection weights and optional biases. */
  public SwiGLU(
      MLXScope scope,
      MLXArray gateWeight,
      MLXArray gateBias,
      MLXArray upWeight,
      MLXArray upBias,
      MLXArray downWeight,
      MLXArray downBias) {
    super(scope);
    Objects.requireNonNull(gateWeight, "SwiGLU: gateWeight must not be null");
    Objects.requireNonNull(upWeight, "SwiGLU: upWeight must not be null");
    Objects.requireNonNull(downWeight, "SwiGLU: downWeight must not be null");
    gateProj = child("gateProj", new Linear(scope, gateWeight, gateBias));
    upProj = child("upProj", new Linear(scope, upWeight, upBias));
    downProj = child("downProj", new Linear(scope, downWeight, downBias));
  }

  @Override
  public MLXArray forward(MLXArray x) {
    Objects.requireNonNull(x, "SwiGLU.forward: x must not be null");
    MLXArray gate = gateProj.forward(x);
    MLXArray activatedGate = MLXOps.multiply(gate, MLXOps.sigmoid(gate));
    return downProj.forward(MLXOps.multiply(activatedGate, upProj.forward(x)));
  }
}
