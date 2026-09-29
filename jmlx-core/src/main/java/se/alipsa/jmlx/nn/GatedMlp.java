package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Gated decoder MLP: {@code down(activation(gate(x)) * up(x))}. */
public class GatedMlp extends UnaryLayer {

  private final UnaryLayer gateProj;
  private final UnaryLayer upProj;
  private final UnaryLayer downProj;
  private final Activation activation;

  /** Registers the three projections under the legacy SwiGLU child names. */
  public GatedMlp(
      MLXScope scope, UnaryLayer gate, UnaryLayer up, UnaryLayer down, Activation activation) {
    super(scope);
    gateProj = child("gateProj", Objects.requireNonNull(gate, "gate"));
    upProj = child("upProj", Objects.requireNonNull(up, "up"));
    downProj = child("downProj", Objects.requireNonNull(down, "down"));
    this.activation = Objects.requireNonNull(activation, "activation");
  }

  @Override
  public MLXArray forward(MLXArray x) {
    Objects.requireNonNull(x, "x");
    MLXArray gate = gateProj.forward(x);
    MLXArray activated =
        switch (activation) {
          case SILU -> MLXOps.multiply(gate, MLXOps.sigmoid(gate));
          case GELU -> {
            MLXArray rootTwo =
                MLX.array(gate.scope(), new float[] {(float) Math.sqrt(2)}, new int[] {1});
            MLXArray one = MLX.array(gate.scope(), new float[] {1}, new int[] {1});
            MLXArray half = MLX.array(gate.scope(), new float[] {0.5f}, new int[] {1});
            yield MLXOps.multiply(
                MLXOps.multiply(gate, half),
                MLXOps.add(one, MLXOps.erf(MLXOps.divide(gate, rootTwo))));
          }
          case GELU_TANH -> {
            MLXArray cube = MLXOps.multiply(MLXOps.multiply(gate, gate), gate);
            MLXArray cubicFactor = MLX.array(gate.scope(), new float[] {0.044715f}, new int[] {1});
            MLXArray scale =
                MLX.array(
                    gate.scope(), new float[] {(float) Math.sqrt(2.0 / Math.PI)}, new int[] {1});
            MLXArray one = MLX.array(gate.scope(), new float[] {1}, new int[] {1});
            MLXArray half = MLX.array(gate.scope(), new float[] {0.5f}, new int[] {1});
            MLXArray inner =
                MLXOps.multiply(scale, MLXOps.add(gate, MLXOps.multiply(cubicFactor, cube)));
            yield MLXOps.multiply(MLXOps.multiply(gate, half), MLXOps.add(one, MLXOps.tanh(inner)));
          }
        };
    return downProj.forward(MLXOps.multiply(activated, upProj.forward(x)));
  }
}
