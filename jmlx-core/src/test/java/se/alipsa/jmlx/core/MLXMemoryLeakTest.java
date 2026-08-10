package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * req/initial-plan.md, Testing approach, "Leak test, specified concretely".
 *
 * <p>"Memory does not grow monotonically" is unfalsifiable: MLX uses a
 * caching allocator, so memory legitimately grows then plateaus. This
 * queries mlx-c's own <b>active</b> memory (not cached, not process RSS,
 * via {@code mlx_get_active_memory} -- {@code memory.h}), runs a warmup
 * phase excluded from measurement so the allocator's steady-state caching
 * doesn't register as a leak, then asserts active memory after a further N
 * iterations is within a fixed threshold of the post-warmup reading.
 *
 * <p>Each iteration allocates a several-hundred-KB array in its own scope
 * and closes it, specifically so that a real per-iteration leak (a broken
 * {@code close()}/{@code Cleaner} backstop) would accumulate to something
 * this test can actually catch, rather than being lost in noise.
 */
@EnabledIfNativeAvailable
class MLXMemoryLeakTest {

    private static final int WARMUP_ITERATIONS = 50;
    private static final int MEASURED_ITERATIONS = 200;
    private static final int ARRAY_ELEMENTS = 100_000; // ~400 KB float32 per iteration

    // Generous relative to one iteration's ~400 KB, but far too small to
    // hide a real per-iteration leak accumulated over 200 iterations.
    private static final long LEAK_THRESHOLD_BYTES = 2_000_000;

    @Test
    void activeMemoryDoesNotGrowAcrossRepeatedScopedIterations() {
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            runIteration();
        }

        long baseline = activeMemoryBytes();

        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            runIteration();
        }

        long after = activeMemoryBytes();

        assertTrue(
            Math.abs(after - baseline) <= LEAK_THRESHOLD_BYTES,
            "active memory grew from " + baseline + " to " + after + " bytes over "
                + MEASURED_ITERATIONS + " iterations (threshold " + LEAK_THRESHOLD_BYTES + " bytes)");
    }

    private static void runIteration() {
        try (MLXScope scope = new MLXScope()) {
            MLXArray a = MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
            MLXArray b = MLX.exp(a);
            MLX.eval(b);
        }
    }

    private static long activeMemoryBytes() {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment out = tmp.allocate(ValueLayout.JAVA_LONG);
            MLX.check(mlx_h.mlx_get_active_memory(out));
            return out.get(ValueLayout.JAVA_LONG, 0);
        }
    }
}
