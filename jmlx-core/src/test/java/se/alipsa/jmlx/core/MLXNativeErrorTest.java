package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/initial-plan.md, Testing approach, "Native error path": a genuine native error surfaces as
 * {@link MLXException}, not a process abort. Every other test in this module is a happy path; this is the one proving
 * the §5 error-handler mitigation actually works, since mlx-c's default error handler calls exit(-1).
 */
@EnabledIfNativeAvailable
class MLXNativeErrorTest {

  @Test
  void mismatchedShapesReachNativeAndSurfaceAsMLXException() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      // add() would intercept this Java-side; addUnchecked() is the
      // deliberate bypass so the mismatch genuinely reaches mlx_add.
      MLXException e = assertThrows(MLXException.class, () -> MLX.addUnchecked(a, b));
      // Not just "some non-zero status": the message must carry
      // mlx-c's own, specific-to-this-failure text (e.g. "Shapes (3)
      // and (2) cannot be broadcast."), not a stale message left over
      // from an earlier, unrelated native call (MLX.checked() clears
      // NativeLoader's thread-local immediately before every native
      // call it wraps, precisely so this can't happen).
      assertTrue(e.getMessage().contains("cannot be broadcast"), e.getMessage());
    }
  }
}
