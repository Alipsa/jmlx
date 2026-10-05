package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.memory.MLXScope;

/** Activation applied to an FFN; model-family allowlists are checked separately. */
public enum Activation {
  SILU,
  GELU,
  GELU_TANH,
  RELU,
  QUICK_GELU;

  /** Creates the unary application without parameters. */
  public UnaryLayer layer(MLXScope scope) {
    return switch (this) {
      case SILU -> new SiLU(scope);
      case GELU -> new GELU(scope);
      case GELU_TANH -> new GELU(scope, true);
      case RELU -> new ReLU(scope);
      case QUICK_GELU -> new QuickGELU(scope);
    };
  }
}
