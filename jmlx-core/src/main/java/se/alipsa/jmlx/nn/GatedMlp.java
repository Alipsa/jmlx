package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Gated decoder MLP: {@code down(activation(gate(x)) * up(x))}. */
public class GatedMlp extends UnaryLayer {

  private final UnaryLayer gateProj;
  private final UnaryLayer upProj;
  private final UnaryLayer downProj;
  private final UnaryModule activationLayer;

  /** Registers the three projections under the legacy SwiGLU child names. */
  public GatedMlp(
      MLXScope scope, UnaryLayer gate, UnaryLayer up, UnaryLayer down, Activation activation) {
    super(scope);
    gateProj = child("gateProj", Objects.requireNonNull(gate, "gate"));
    upProj = child("upProj", Objects.requireNonNull(up, "up"));
    downProj = child("downProj", Objects.requireNonNull(down, "down"));
    activationLayer =
        switch (Objects.requireNonNull(activation, "activation")) {
          case SILU -> new SiLU(scope);
          case GELU -> new GELU(scope);
          case GELU_TANH -> new GELU(scope, true);
        };
  }

  @Override
  public MLXArray forward(MLXArray x) {
    Objects.requireNonNull(x, "x");
    MLXArray activated = activationLayer.forward(gateProj.forward(x));
    return downProj.forward(MLXOps.multiply(activated, upProj.forward(x)));
  }
}
