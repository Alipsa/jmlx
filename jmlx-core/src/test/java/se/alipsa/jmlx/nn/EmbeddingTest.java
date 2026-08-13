package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class EmbeddingTest {

  private static final float EPS = 1e-5f;

  @Test
  void forwardLooksUpRowsByIndex() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {3, 2});
      Embedding embedding = new Embedding(scope, weight);
      MLXArray indices = MLX.array(scope, new int[] {2, 0}, new int[] {2});

      MLXArray result = embedding.forward(indices);

      assertArrayEquals(new int[] {2, 2}, result.shape());
      assertArrayEquals(new float[] {5, 6, 1, 2}, result.toFloatArray(), EPS);
    }
  }

  @Test
  void constructorRejectsARankOneWeight() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> new Embedding(scope, weight));
    }
  }
}
