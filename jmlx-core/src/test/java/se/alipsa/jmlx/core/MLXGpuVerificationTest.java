package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/initial-plan.md, Verification, item 9: a device query alone succeeds even with the metallib missing (mlx-c
 * can report a device type without ever dispatching a kernel); only a kernel dispatch proves the stack. Combines a real
 * kernel dispatch (matmul, already covered elementwise by {@link MLXNumericTest}) with an explicit assertion that the
 * default device is GPU, not just "sane" (the weaker check {@code NativeLoaderSmokeTest} deliberately makes at the
 * FFI-smoke layer).
 */
@EnabledIfNativeAvailable
class MLXGpuVerificationTest {

  @Test
  void defaultDeviceIsGpuAndMatmulDispatchesOnIt() {
    int type;
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment typeOut = tmp.allocate(ValueLayout.JAVA_INT);
      MLX.checked(() -> mlx_h.mlx_device_get_type(typeOut, MLX.defaultDevice()));
      type = typeOut.get(ValueLayout.JAVA_INT, 0);
    }
    assertEquals(mlx_h.MLX_GPU(), type, "default device is not GPU");

    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray b = MLX.array(scope, new float[] {5, 6, 7, 8}, new int[] {2, 2});
      assertArrayEquals(new float[] {19, 22, 43, 50}, MLX.matmul(a, b).toFloatArray(), 1e-3f);
    }
  }
}
