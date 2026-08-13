package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/phase3-plan.md §4, Testing approach rows for the {@code mlx_vector_array}-based joint {@link MLX#eval}.
 */
@EnabledIfNativeAvailable
class MLXEvalTest {

  @Test
  void zeroArgEvalIsANoOp() {
    // Pins the n == 0 early return in eval() itself: it returns before ever
    // capturing handles or calling newVectorArray, so this does not exercise
    // (and cannot prove anything about) that method's null-ctx check.
    assertDoesNotThrow(() -> MLX.eval());
  }

  @Test
  void evalOfArraysFromTwoScopesOnOneThreadDoesNotThrow() {
    // The case a same-scope check (like binaryOp's) would wrongly reject:
    // eval allocates no result, so there is no "which scope owns the
    // output" question, and mixing scopes here is legitimate.
    try (MLXScope scope1 = new MLXScope(); MLXScope scope2 = new MLXScope()) {
      MLXArray a = MLX.array(scope1, new float[] {1f, 2f, 3f}, new int[] {3});
      MLXArray b = MLX.array(scope2, new float[] {4f, 5f, 6f}, new int[] {3});
      MLXArray c = MLXOps.exp(a);
      assertDoesNotThrow(() -> MLX.eval(a, b, c));
      assertArrayEquals(new float[] {4f, 5f, 6f}, b.toFloatArray());
    }
  }

  @Test
  void evalTwiceIsIdempotent() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1f, 2f, 3f}, new int[] {3});
      MLXArray b = MLXOps.exp(a);
      assertDoesNotThrow(() -> MLX.eval(a, b));
      // Second eval hits mlx's "already scheduled" fast path (array::eval()
      // is a wait() the second time around, upstream array.cpp:154-161) --
      // this asserts that re-entering it is safe, not just fast.
      assertDoesNotThrow(() -> MLX.eval(a, b));
      assertArrayEquals(new float[] {1f, 2f, 3f}, a.toFloatArray());
    }
  }

  @Test
  void evalAttributesTheFailingArrayByIndexAndLeavesSiblingsReadable() {
    // Allocation-class error, not a shape error: shapes are validated at
    // op-build time (mlx_add etc. throw eagerly), so a shape-mismatch
    // fixture would fail before ever reaching eval and couldn't exercise
    // this path. Built via outer(), not MLX.array(): the direct route needs
    // a float[] of 2^40 elements, 512x past what a JVM array can index.
    try (MLXScope scope = new MLXScope()) {
      MLXArray good1 = MLX.array(scope, new float[] {1f, 2f, 3f}, new int[] {3});
      float[] big = new float[1 << 20];
      MLXArray a = MLX.array(scope, big, new int[] {1 << 20});
      MLXArray bad = MLXOps.outer(a, a); // shape [2^20, 2^20]; ~4 TiB, never actually allocated
      MLXArray good2 = MLX.array(scope, new float[] {4f, 5f}, new int[] {2});

      MLXException e = assertThrows(MLXException.class, () -> MLX.eval(good1, bad, good2));
      assertTrue(e.getMessage().contains("array[1]"), e.getMessage());

      // Both good arrays remain readable after the joint failure and the
      // per-array re-run: the failing array re-throws immediately rather
      // than hanging, and its siblings are untouched.
      assertArrayEquals(new float[] {1f, 2f, 3f}, good1.toFloatArray());
      assertArrayEquals(new float[] {4f, 5f}, good2.toFloatArray());
    }
  }
}
