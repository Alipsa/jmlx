package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * req/initial-plan.md, Testing approach, "Native error path": a genuine
 * native error surfaces as {@link MLXException}, not a process abort. Every
 * other test in this module is a happy path; this is the one proving the §5
 * error-handler mitigation actually works, since mlx-c's default error
 * handler calls exit(-1).
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
            assertThrows(MLXException.class, () -> MLX.addUnchecked(a, b));
        }
    }
}
