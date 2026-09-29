package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestReporter;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Records exact-tie ordering on the pinned native runtime. Argsort ordering is observational: MoE
 * uses successive argmax calls because only argmax's lowest-index tie behavior is guaranteed.
 */
@EnabledIfNativeAvailable
class MoeRoutingProbeTest {

  @Test
  void exactTiesUseLowestIndexWithArgmax(TestReporter reporter) {
    try (MLXScope scope = new MLXScope()) {
      MLXArray scores = MLX.array(scope, new float[] {1, 1, 1, 0, 2, 2, 0, 0}, new int[] {2, 4});
      assertArrayEquals(new int[] {0, 0}, MLXOps.argmaxAxis(scores, -1, false).toIntArray());
      int[] sorted = MLXOps.argsortAxis(MLXOps.negative(scores), -1).toIntArray();
      reporter.publishEntry("argsort exact-tie order", java.util.Arrays.toString(sorted));
    }
  }
}
