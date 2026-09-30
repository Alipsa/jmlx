package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Native coverage for the gathered-matmul, stack and target-scope swapaxes facades. */
@EnabledIfNativeAvailable
class MLXGatherOpsTest {

  private static final float EPS = 1e-6f;

  /** Three 2x2 "experts": identity, 2 * identity, and a row swap. */
  private static MLXArray experts(MLXScope scope) {
    return MLX.array(scope, new float[] {1, 0, 0, 1, 2, 0, 0, 2, 0, 1, 1, 0}, new int[] {3, 2, 2});
  }

  @Test
  void rhsIndicesSelectOneMatrixPerBatchEntry() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 1, 2});
      MLXArray idx = MLX.array(scope, new int[] {2, 0}, new int[] {2});
      MLXArray r = MLXOps.gatherMatmul(a, experts(scope), null, idx, false);
      assertArrayEquals(new int[] {2, 1, 2}, r.shape());
      assertArrayEquals(new float[] {2, 1, 3, 4}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void sortedIndicesFlagGivesSameResultForSortedInput() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 1, 2});
      MLXArray idx = MLX.array(scope, new int[] {0, 2}, new int[] {2});
      MLXArray r = MLXOps.gatherMatmul(a, experts(scope), null, idx, true);
      assertArrayEquals(new float[] {1, 2, 4, 3}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void lhsIndicesSelectRowsOfA() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 1, 2});
      MLXArray lhs = MLX.array(scope, new int[] {1, 1}, new int[] {2});
      MLXArray rhs = MLX.array(scope, new int[] {0, 2}, new int[] {2});
      MLXArray r = MLXOps.gatherMatmul(a, experts(scope), lhs, rhs, false);
      assertArrayEquals(new float[] {3, 4, 4, 3}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void broadcastsTokenAxesAgainstTopKIndices() {
    try (MLXScope scope = new MLXScope()) {
      // x: [B=1, T=2, 1, 1, H=2]; idx: [B=1, T=2, K=2] -> [1, 2, 2, 1, 2]
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {1, 2, 1, 1, 2});
      MLXArray idx = MLX.array(scope, new int[] {0, 2, 1, 0}, new int[] {1, 2, 2});
      MLXArray r = MLXOps.gatherMatmul(x, experts(scope), null, idx, false);
      assertArrayEquals(new int[] {1, 2, 2, 1, 2}, r.shape());
      assertArrayEquals(new float[] {1, 2, 2, 1, 6, 8, 3, 4}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void bf16StaysBf16() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a =
          MLX.astype(MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 2}), DType.BFLOAT16);
      MLXArray b = MLX.astype(experts(scope), DType.BFLOAT16);
      MLXArray idx = MLX.array(scope, new int[] {1}, new int[] {1});
      assertEquals(DType.BFLOAT16, MLXOps.gatherMatmul(a, b, null, idx, false).dtype());
    }
  }

  @Test
  void floatIndicesAreRejectedAsMlxException() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 2});
      MLXArray idx = MLX.array(scope, new float[] {0}, new int[] {1});
      assertThrows(
          MLXException.class, () -> MLXOps.gatherMatmul(a, experts(scope), null, idx, false));
    }
  }

  @Test
  void bothIntegerOperandsAreRejectedBeforeNative() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new int[] {1, 2}, new int[] {1, 1, 2});
      MLXArray b = MLX.array(scope, new int[] {1, 0, 0, 1}, new int[] {1, 2, 2});
      MLXArray idx = MLX.array(scope, new int[] {0}, new int[] {1});
      String message =
          assertThrows(
                  IllegalArgumentException.class, () -> MLXOps.gatherMatmul(a, b, null, idx, false))
              .getMessage();
      assertEquals(
          "gatherMatmul: requires at least one inexact dtype, got INT32 and INT32", message);
    }
  }

  @Test
  void resultLandsInInnermostOperandScope() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray b = experts(model);
      MLXArray a = MLX.array(step, new float[] {1, 2}, new int[] {1, 1, 2});
      MLXArray idx = MLX.array(step, new int[] {0}, new int[] {1});
      assertEquals(step, MLXOps.gatherMatmul(a, b, null, idx, false).scope());
    }
  }

  @Test
  void stackInsertsANewAxis() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      MLXArray y = MLX.array(scope, new float[] {3, 4}, new int[] {2});
      MLXArray s = MLXShape.stack(new MLXArray[] {x, y}, 1);
      assertArrayEquals(new int[] {2, 2}, s.shape());
      assertArrayEquals(new float[] {1, 3, 2, 4}, s.toFloatArray(), EPS);
      assertArrayEquals(new int[] {2, 2}, MLXShape.stack(new MLXArray[] {x, y}, 0).shape());
    }
  }

  @Test
  void stackRejectsEmptyAndNullAndMismatchedShapes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      MLXArray z = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> MLXShape.stack(new MLXArray[0], 0));
      assertThrows(
          IllegalArgumentException.class, () -> MLXShape.stack(new MLXArray[] {x, null}, 0));
      assertThrows(MLXException.class, () -> MLXShape.stack(new MLXArray[] {x, z}, 0));
    }
  }

  @Test
  void swapaxesWithTargetAllocatesIntoTheChildScope() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray w = experts(model);
      MLXArray t = MLXShape.swapaxes(w, step, -1, -2);
      assertEquals(step, t.scope());
      assertArrayEquals(new float[] {1, 0, 0, 1, 2, 0, 0, 2, 0, 1, 1, 0}, t.toFloatArray(), EPS);
    }
  }
}
