package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.ref.WeakReference;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/**
 * req/initial-plan.md, Testing approach, "Memory lifecycle" (scope half) and
 * "Testing the Cleaner backstop".
 */
@EnabledIfNativeAvailable
class MLXScopeTest {

    @Test
    void closeIsIdempotent() {
        MLXScope scope = new MLXScope();
        scope.close();
        assertDoesNotThrow(scope::close);
    }

    @Test
    void allocateAfterCloseThrows() {
        MLXScope scope = new MLXScope();
        scope.close();
        assertThrows(IllegalStateException.class, () -> scope.allocate(8, 8));
    }

    /**
     * "The action provably captures a holder, not the scope" is enforced
     * structurally (a static-nested holder, not the scope, is registered
     * with the Cleaner), not directly assertable. What is observable is the
     * consequence: if the capture rule were violated, the scope could never
     * become unreachable and this loop would never observe a cleared
     * referent, tripping the timeout instead.
     */
    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void cleanerBackstopAllowsAnEscapedScopeToBecomeUnreachable() throws InterruptedException {
        WeakReference<MLXScope> ref = new WeakReference<>(createDetachedScope());

        while (ref.get() != null) {
            System.gc();
            Thread.sleep(50);
        }

        assertNull(ref.get());
    }

    // Isolated in its own frame so no local variable in the calling test
    // method keeps the scope reachable after this method returns. Allocates
    // through MLX.array(), not scope.allocate() directly: the latter is only
    // valid when given a real mlx_array-shaped constructor call, and a raw
    // uninitialized allocation would make mlx_array_free undefined behavior
    // once the backstop runs.
    private static MLXScope createDetachedScope() {
        MLXScope scope = new MLXScope();
        MLX.array(scope, new float[] {1f, 2f, 3f}, new int[] {3});
        return scope;
    }
}
