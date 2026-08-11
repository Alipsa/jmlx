package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** See req/initial-plan.md, Testing approach, "Memory lifecycle" (array half). */
@EnabledIfNativeAvailable
class MLXArrayTest {

  @Test
  void closeIsIdempotent() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1f, 2f, 3f}, new int[] {3});
      a.close();
      assertDoesNotThrow(a::close);
    }
  }

  @Test
  void useAfterCloseThrowsNamingTheArray() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1f}, new int[] {1});
      a.close();
      IllegalStateException e = assertThrows(IllegalStateException.class, a::shape);
      assertTrue(e.getMessage().contains("MLXArray"), e.getMessage());
    }
  }

  @Test
  void closingAnArrayEarlyThenClosingItsScopeDoesNotDoubleFree() {
    MLXScope scope = new MLXScope();
    MLXArray a = MLX.array(scope, new float[] {1f, 2f}, new int[] {2});
    a.close();
    assertDoesNotThrow(scope::close);
  }

  @Test
  void closingAScopeFreesEveryArrayStillOpenOnIt() {
    MLXScope scope = new MLXScope();
    MLX.array(scope, new float[] {1f, 2f, 3f}, new int[] {3});
    MLX.array(scope, new float[] {4f, 5f}, new int[] {2});
    assertDoesNotThrow(scope::close);
  }
}
