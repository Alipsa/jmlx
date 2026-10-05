package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Isolated unsupported native configuration probe, never part of check. */
@EnabledIfNativeAvailable
class Phase71ConvolutionProbe {
  @Test
  void groupedThreeDimensionalConvolutionIsUnsupported() {
    System.out.println("PHASE71_CASE_EXECUTED: Phase71ConvolutionProbe");
    try (MLXScope s = new MLXScope()) {
      MLXArray x = MLX.ones(s, new int[] {1, 2, 2, 2, 2}, DType.FLOAT32);
      MLXArray w = MLX.ones(s, new int[] {2, 1, 1, 1, 1}, DType.FLOAT32);
      assertThrows(
          MLXException.class,
          () ->
              MLXConv.conv3d(x, w, new int[] {1, 1, 1}, new int[] {0, 0, 0}, new int[] {1, 1, 1}, 2)
                  .toFloatArray());
    }
  }
}
