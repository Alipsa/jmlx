package se.alipsa.jmlx.ffi;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * req/initial-plan.md, Testing approach, "FFI smoke": a downcall reaches the
 * dylib and returns a sane device. This is the §5 gate: it must pass before
 * jmlx-core's memory/array/op layers are built on top of NativeLoader.
 */
class NativeLoaderSmokeTest {

    @Test
    @EnabledIfNativeAvailable
    void deviceQueryReturnsSaneDevice() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dev = mlx_h.mlx_device_new(arena);
            assertEquals(0, mlx_h.mlx_get_default_device(dev));

            MemorySegment typeOut = arena.allocate(ValueLayout.JAVA_INT);
            assertEquals(0, mlx_h.mlx_device_get_type(typeOut, dev));
            int type = typeOut.get(ValueLayout.JAVA_INT, 0);
            // MLX_CPU=0, MLX_GPU=1 -- either is "sane"; verification item 9
            // (asserting the default device is specifically GPU) belongs to
            // the numeric-correctness layer, not this smoke test.
            assertTrue(type == 0 || type == 1, "unexpected device type: " + type);

            assertEquals(0, mlx_h.mlx_device_free(dev));
        }
    }
}
