package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** The decoder MLP used by Llama/Qwen: {@code down(silu(gate(x)) * up(x))}. */
public final class SwiGLU extends GatedMlp {

  /** Creates the MLP from checkpoint-layout projection weights and optional biases. */
  public SwiGLU(
      MLXScope scope,
      MLXArray gateWeight,
      MLXArray gateBias,
      MLXArray upWeight,
      MLXArray upBias,
      MLXArray downWeight,
      MLXArray downBias) {
    super(
        scope,
        new Linear(scope, gateWeight, gateBias),
        new Linear(scope, upWeight, upBias),
        new Linear(scope, downWeight, downBias),
        Activation.SILU);
  }
}
