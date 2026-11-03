package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Request ownership and transactional replacement of convolution and recurrent state. */
@EnabledIfNativeAvailable
class RecurrentCacheTest {
  @Test
  void replacementOutlivesStepScopeAndRetiresPreviousHandles() {
    try (MLXScope request = new MLXScope();
        RecurrentCache cache = new RecurrentCache(request, 1, 6, 32, 64, 4, 512)) {
      assertEquals(4L * (6 * 32 * 64 + 3 * 512), cache.logicalStorageBytes());
      assertEquals(0, cache.position());
      final MLXArray original = cache.state();
      try (MLXScope step = request.newChild()) {
        cache.replace(
            MLX.full(step, new int[] {1, 3, 512}, 2f, DType.FLOAT32),
            MLX.full(step, new int[] {1, 6, 32, 64}, 3f, DType.FLOAT32),
            65);
      }
      assertEquals(65, cache.position());
      assertEquals(2f, cache.convolution().toFloatArray()[0]);
      assertEquals(3f, cache.state().toFloatArray()[0]);
      assertThrows(IllegalStateException.class, original::toFloatArray);
      assertFalse(cache.isPoisoned());
    }
  }

  @Test
  void sameScopeInputsHaveDistinctOwnedRetainedHandles() {
    try (MLXScope request = new MLXScope();
        RecurrentCache cache = new RecurrentCache(request, 1, 2, 3, 4, 1, 5)) {
      MLXArray convolution = MLX.zeros(request, new int[] {1, 0, 5}, DType.FLOAT32);
      MLXArray state = MLX.ones(request, new int[] {1, 2, 3, 4}, DType.FLOAT32);
      cache.replace(convolution, state, 1);
      convolution.close();
      state.close();
      assertEquals(1f, cache.state().toFloatArray()[0]);
      assertArrayEquals(new int[] {1, 0, 5}, cache.convolution().shape());
    }
  }

  @Test
  void validationAndPoisoningPreventUnsafeStateReuse() {
    try (MLXScope request = new MLXScope();
        RecurrentCache cache = new RecurrentCache(request, 1, 2, 3, 4, 2, 5)) {
      MLXArray convolution = MLX.zeros(request, new int[] {1, 1, 5}, DType.FLOAT32);
      MLXArray wrong = MLX.zeros(request, new int[] {1, 2, 4, 3}, DType.FLOAT32);
      assertThrows(IllegalArgumentException.class, () -> cache.replace(convolution, wrong, 1));
      assertEquals(0, cache.position());
      assertFalse(cache.isPoisoned());
      cache.poison();
      assertTrue(cache.isPoisoned());
      assertThrows(IllegalStateException.class, () -> cache.replace(convolution, cache.state(), 1));
    }
  }

  @Test
  void rejectsInvalidAndOverflowingGeometryBeforeAllocation() {
    try (MLXScope scope = new MLXScope()) {
      assertThrows(
          IllegalArgumentException.class, () -> new RecurrentCache(scope, 0, 2, 3, 4, 2, 5));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new RecurrentCache(
                  scope,
                  Integer.MAX_VALUE,
                  Integer.MAX_VALUE,
                  Integer.MAX_VALUE,
                  Integer.MAX_VALUE,
                  2,
                  5));
      RecurrentCache cache = new RecurrentCache(scope, 1, 2, 3, 4, 2, 5);
      cache.close();
      cache.close();
      assertThrows(IllegalStateException.class, cache::state);
    }
  }
}
