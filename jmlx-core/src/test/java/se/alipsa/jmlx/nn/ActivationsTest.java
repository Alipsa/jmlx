package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Pins Gemma's activation against torch 2.6 functional GELU values. */
@EnabledIfNativeAvailable
class ActivationsTest {
  @Test
  void tanhAndExactGeluMatchIndependentTorchValues() {
    float[] input = {-3, -1, 0, 0.5f, 2};
    float[] tanh = {-0.003637433f, -0.158807993f, 0, 0.345714003f, 1.954597712f};
    float[] exact = {-0.004049867f, -0.158655286f, 0, 0.345731199f, 1.95449996f};
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, input, new int[] {1, 1, 5});
      assertArrayEquals(tanh, mlp(scope, Activation.GELU_TANH).forward(x).toFloatArray(), 1e-5f);
      assertArrayEquals(exact, mlp(scope, Activation.GELU).forward(x).toFloatArray(), 1e-5f);
    }
  }

  private static GatedMlp mlp(MLXScope scope, Activation activation) {
    float[] identity = new float[25];
    for (int i = 0; i < 5; i++) {
      identity[i * 5 + i] = 1;
    }
    MLXArray identityWeight = MLX.array(scope, identity, new int[] {5, 5});
    MLXArray zeroWeight = MLX.zeros(scope, new int[] {5, 5}, DType.FLOAT32);
    MLXArray oneBias = MLX.array(scope, new float[] {1, 1, 1, 1, 1}, new int[] {5});
    return new GatedMlp(
        scope,
        new Linear(scope, identityWeight, null),
        new Linear(scope, zeroWeight, oneBias),
        new Linear(scope, identityWeight, null),
        activation);
  }
}
