package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Opt-in gate (Phase 8.2 WP2): the binding counters must cover the direct allocation/free paths as
 * well as the op facades, so a counting window that only saw a subset of the generated downcalls
 * could never be mistaken for a complete phase-local count. Runs only through the {@code
 * bindingCoverage} task, with the agent installed.
 */
@EnabledIfNativeAvailable
@Tag("binding-coverage")
class BindingCoverageTest {
  @Test
  void countersCoverDirectAllocationFreeAndOpFacade() {
    assertTrue(BindingCounterAgent.isInstalled(), "run through the bindingCoverage task");
    try (MLXScope outer = new MLXScope()) {
      BindingCounterAgent.start();
      try (MLXScope step = outer.newChild()) {
        // Direct allocation path: MLX.array uploads through mlx_array_new_data.
        MLXArray a = MLX.array(step, new float[] {1f, 2f, 3f, 4f}, new int[] {2, 2});
        MLXArray b = MLX.array(step, new float[] {5f, 6f, 7f, 8f}, new int[] {2, 2});
        // Op facade: matmul is the representative operator downcall.
        MLXArray c = MLXOps.matmul(a, b);
        MLX.eval(c);
      }
      // The child scope's close frees the step's arrays through mlx_array_free, inside the
      // window: a window that saw only constructions would miss the direct free path.
      Map<String, Long> counts = BindingCounterAgent.stop();
      assertTrue(
          counts.getOrDefault("mlx_array_new_data", 0L) >= 2,
          "the direct allocation path must be counted (got " + counts + ")");
      assertTrue(
          counts.getOrDefault("mlx_array_free", 0L) >= 2,
          "the direct free path must be counted (got " + counts + ")");
      assertTrue(
          counts.getOrDefault("mlx_matmul", 0L) == 1,
          "the op facade downcall must be counted (got " + counts + ")");
    }
  }
}
