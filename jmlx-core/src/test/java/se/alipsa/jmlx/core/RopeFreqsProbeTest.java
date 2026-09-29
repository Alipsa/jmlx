package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.RopeSpec;

/**
 * Probes the pinned mlx-c RoPE contract: freqs are periods, and scale composes with the offset.
 * Dynamic NTK output depends on prefill chunking because cached keys keep their old rotation.
 */
@EnabledIfNativeAvailable
class RopeFreqsProbeTest {
  @Test
  void explicitPeriodsMatchBaseAndComposeWithScale() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {1, 1, 4});
      MLXArray periods = MLX.array(scope, new float[] {1, 100}, new int[] {2});
      assertArrayEquals(
          MLXFast.rope(x, 4, false, 10_000f, 1, 3, null).toFloatArray(),
          MLXFast.rope(x, 4, false, null, 1, 3, periods).toFloatArray(),
          1e-5f);
      assertArrayEquals(
          MLXFast.rope(x, 4, false, 10_000f, 0.25f, 3, null).toFloatArray(),
          MLXFast.rope(x, 4, false, null, 0.25f, 3, periods).toFloatArray(),
          1e-5f);
    }
  }

  @Test
  void yarnLeavesPassThroughCoordinatesUntouched() {
    try (MLXScope scope = new MLXScope()) {
      float[] input = {1, 2, 3, 4, 5, 6, 7, 8};
      MLXArray x = MLX.array(scope, input, new int[] {1, 1, 8});
      RopeSpec yarn = new RopeSpec.Yarn(10_000, 4, 64, 32, 1, 0, 0, null, true);
      float[] actual = yarn.apply(x, 4, 9, null).toFloatArray();
      for (int i = 4; i < 8; i++) {
        assertArrayEquals(new float[] {input[i]}, new float[] {actual[i]});
      }
    }
  }

  @Test
  void dynamicNtkChunkingChangesRotationAfterOriginalContext() {
    RopeSpec dynamic = new RopeSpec.DynamicNtk(10_000, 2, 8);
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {1, 1, 4});
      float[] fromChunk = dynamic.apply(x, 4, 5, null).toFloatArray();
      float[] repeated = new float[12 * 4];
      for (int i = 0; i < 12; i++) {
        System.arraycopy(new float[] {1, 2, 3, 4}, 0, repeated, 4 * i, 4);
      }
      MLXArray full = MLX.array(scope, repeated, new int[] {1, 12, 4});
      MLXArray allRotated = dynamic.apply(full, 4, 0, null);
      float[] fromFull =
          MLXShape.slice(allRotated, new int[] {0, 5, 0}, new int[] {1, 6, 4}).toFloatArray();
      assertNotEquals(fromChunk[1], fromFull[1]);
      assertArrayEquals(fromChunk, dynamic.apply(x, 4, 5, null).toFloatArray());
    }
  }
}
