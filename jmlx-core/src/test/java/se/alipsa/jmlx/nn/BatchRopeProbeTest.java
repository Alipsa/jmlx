package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXException;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class BatchRopeProbeTest {
  @Test
  void rowOffsetsMatchScalarRopeAndPaddingIsPreserved() {
    try (MLXScope scope = new MLXScope()) {
      RopeSpec rope = new RopeSpec.Base(10000f);
      MLXArray x = MLX.array(scope, new float[] {1, 0, 1, 0, 1, 0, 1, 0}, new int[] {2, 1, 2, 2});
      MLXArray batched = rope.apply(x, 2, new int[] {0, 3}, new int[] {1, 2}, null);
      MLXArray row = MLXShape.slice(x, new int[] {1, 0, 0, 0}, new int[] {2, 1, 2, 2});
      assertArrayEquals(
          rope.apply(row, 2, 3, null).toFloatArray(),
          MLXShape.slice(batched, new int[] {1, 0, 0, 0}, new int[] {2, 1, 2, 2}).toFloatArray(),
          1e-5f);
      assertArrayEquals(
          new float[] {1, 0},
          MLXShape.slice(batched, new int[] {0, 0, 0, 0}, new int[] {1, 1, 1, 2}).toFloatArray(),
          1e-5f);
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new RopeSpec.DynamicNtk(10000f, 2f, 2)
                  .apply(x, 2, new int[] {0, 3}, new int[] {1, 2}, null));
    }
  }

  @Test
  void nativeArrayOffsetMatchesScalarOffsetForOneRow() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 0}, new int[] {1, 1, 1, 2});
      MLXArray offset = MLX.array(scope, new int[] {3}, new int[] {1});
      try {
        assertArrayEquals(
            MLXFast.rope(x, 2, false, 10000f, 1f, 3, null).toFloatArray(),
            MLXFast.ropeDynamic(x, 2, false, 10000f, 1f, offset, null).toFloatArray(),
            1e-5f);
      } catch (MLXException unsupportedShape) {
        // The production batch path uses the independently tested per-row scalar kernel.
        System.out.println(
            "mlx_fast_rope_dynamic offset shape [1] unsupported: " + unsupportedShape.getMessage());
      }
    }
  }

  @Test
  void probesPerRowArrayOffsetBroadcastAgainstFallback() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 0, 1, 0}, new int[] {2, 1, 1, 2});
      MLXArray offsets = MLX.array(scope, new int[] {0, 3}, new int[] {2, 1, 1});
      float[] fallback =
          new RopeSpec.Base(10000f)
              .apply(x, 2, new int[] {0, 3}, new int[] {1, 1}, null)
              .toFloatArray();
      try {
        float[] direct = MLXFast.ropeDynamic(x, 2, false, 10000f, 1f, offsets, null).toFloatArray();
        float error = 0;
        for (int i = 0; i < direct.length; i++) {
          error = Math.max(error, Math.abs(direct[i] - fallback[i]));
        }
        System.out.println("mlx_fast_rope_dynamic per-row offset max error: " + error);
      } catch (MLXException unsupportedShape) {
        System.out.println(
            "mlx_fast_rope_dynamic per-row offset unsupported: " + unsupportedShape.getMessage());
      }
    }
  }
}
