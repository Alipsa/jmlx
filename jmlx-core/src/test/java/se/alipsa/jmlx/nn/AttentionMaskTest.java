package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class AttentionMaskTest {

  @Test
  void prefillAndDecodeUseAbsoluteWindowBoundaries() {
    try (MLXScope scope = new MLXScope()) {
      assertArrayEquals(
          new float[] {1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1},
          MLX.astype(AttentionMask.slidingWindow(scope, 4, 4, 2), DType.FLOAT32).toFloatArray());
      assertArrayEquals(
          new float[] {0, 0, 1, 1, 1},
          MLX.astype(AttentionMask.slidingWindow(scope, 1, 5, 3), DType.FLOAT32).toFloatArray());
      assertArrayEquals(
          new float[] {1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1},
          MLX.astype(AttentionMask.slidingWindow(scope, 4, 4, 4), DType.FLOAT32).toFloatArray());
      assertArrayEquals(
          new float[] {0, 1, 1, 1, 1},
          MLX.astype(AttentionMask.slidingWindow(scope, 8, 4, 1, 5, 4), DType.FLOAT32)
              .toFloatArray());
    }
  }

  @Test
  void rejectsInvalidWindow() {
    try (MLXScope scope = new MLXScope()) {
      assertThrows(
          IllegalArgumentException.class, () -> AttentionMask.slidingWindow(scope, 1, 1, 0));
    }
  }
}
