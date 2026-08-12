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
      // [3] and [2] are still broadcast-INcompatible under the NumPy rules
      // MLXOps.requireBroadcastCompatible now mirrors (neither dim is 1, and
      // they are not equal), so mlx_add's own "cannot be broadcast" message
      // below still fires. Do NOT "modernize" this pair to something
      // broadcast-compatible (e.g. [2,2] against [2]): that would silently
      // turn this into a happy path and assertThrows would fail. The
      // (2,2)/(3) pair recorded at initial-plan.md:191 also still throws
      // and additionally exercises the rank-mismatch branch, if a second
      // fixture is ever wanted.
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      MLXArray b = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      // add() would intercept this Java-side; addUnchecked() is the
      // deliberate bypass so the mismatch genuinely reaches mlx_add.
      // addUnchecked is not dead code: now that the Java-side guard is
      // broadcast-aware rather than exact-shape, it is the *only* remaining
      // route to native's error handler that this Java facade exercises --
      // and this test is the sole coverage of the exit(-1) mitigation in
      // NativeLoader.java:119-128. Removing addUnchecked (or routing add()
      // through it unconditionally) would silently drop that coverage.
      MLXException e = assertThrows(MLXException.class, () -> MLXOps.addUnchecked(a, b));
      // Not just "some non-zero status": the message must carry
      // mlx-c's own, specific-to-this-failure text (e.g. "Shapes (3)
      // and (2) cannot be broadcast."), not a stale message left over
      // from an earlier, unrelated native call (NativeOps.checked() clears
      // NativeLoader's thread-local immediately before every native
      // call it wraps, precisely so this can't happen).
      assertTrue(e.getMessage().contains("cannot be broadcast"), e.getMessage());
    }
  }
}
