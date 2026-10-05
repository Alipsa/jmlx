package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Required CI regression for contiguous view origins and checked element addresses. */
@EnabledIfNativeAvailable
class StridedViewSafetyTest {
  @Test
  void logicalOriginsSurviveSlicesAndTransposes() {
    try (MLXScope s = new MLXScope()) {
      MLXArray a = MLX.array(s, new float[] {0, 1, 2, 3, 4, 5, 6, 7}, new int[] {8});
      MLXArray slice = MLXShape.slice(a, new int[] {3}, new int[] {7});
      assertArrayEquals(
          new float[] {3, 4, 5, 6},
          MLXShape.asStrided(slice, new int[] {4}, new long[] {1}, 0).toFloatArray());
      assertArrayEquals(
          new float[] {6, 5, 4, 3},
          MLXShape.asStrided(slice, new int[] {4}, new long[] {-1}, 3).toFloatArray());
      assertArrayEquals(
          new float[] {4, 4, 4},
          MLXShape.asStrided(slice, new int[] {3}, new long[] {0}, 1).toFloatArray());
      MLXArray transposed = MLXShape.transpose(MLXShape.reshape(a, new int[] {2, 4}));
      assertArrayEquals(
          new float[] {0, 4, 1, 5, 2, 6, 3, 7},
          MLXShape.asStrided(transposed, new int[] {8}, new long[] {1}, 0).toFloatArray());
    }
  }

  @Test
  void checksBoundsOverflowEmptyAndScalar() {
    try (MLXScope s = new MLXScope()) {
      MLXArray a = MLX.array(s, new float[] {1, 2, 3}, new int[] {3});
      assertArrayEquals(
          new float[] {3}, MLXShape.asStrided(a, new int[0], new long[0], 2).toFloatArray());
      assertThrows(
          IllegalArgumentException.class, () -> MLXShape.asStrided(a, new int[0], new long[0], 3));
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXShape.asStrided(a, new int[] {2}, new long[] {-1}, 0));
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXShape.asStrided(a, new int[] {2}, new long[] {1}, 2));
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXShape.asStrided(a, new int[] {3}, new long[] {Long.MAX_VALUE}, 0));
      assertArrayEquals(
          new float[0], MLXShape.asStrided(a, new int[] {0}, new long[] {1}, 0).toFloatArray());
    }
  }

  @Test
  void broadcastNormalizationAndSourceLifetime() {
    MLXArray result;
    try (MLXScope scope = new MLXScope()) {
      MLXArray source = MLX.array(scope, new float[] {2, 4}, new int[] {1, 2});
      MLXArray broadcast = MLXShape.broadcastTo(source, new int[] {3, 2});
      result = MLXShape.asStrided(broadcast, new int[] {6}, new long[] {1}, 0);
      assertArrayEquals(new float[] {2, 4, 2, 4, 2, 4}, result.toFloatArray());
    }
    assertThrows(IllegalStateException.class, result::toFloatArray);
  }
}
