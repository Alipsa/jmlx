package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Native coverage for the narrow selection facades used by generation sampling. */
@EnabledIfNativeAvailable
class MLXSamplingOpsTest {

  @Test
  void selectionAndReductionFacadesPreserveAxisSemantics() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray values = MLX.array(scope, new float[] {3, 1, 3, 0}, new int[] {1, 4});
      MLXArray order = MLXOps.argsortAxis(values, -1);
      MLXArray sorted = MLXShape.takeAlongAxis(values, order, -1);
      MLXArray cumulative = MLXOps.cumulativeSumAxis(sorted, -1, false, true);
      MLXArray restored =
          MLXShape.putAlongAxis(
              MLX.zeros(scope, new int[] {1, 4}, DType.FLOAT32), order, sorted, -1);
      MLXArray finite = MLX.astype(MLXOps.all(MLXOps.isFinite(values)), DType.INT32);
      MLXArray logSumExp = MLXOps.logSumExpAxis(values, -1, false);

      MLX.eval(order, sorted, cumulative, restored, finite, logSumExp);

      assertArrayEquals(new int[] {3, 1, 0, 2}, order.toIntArray());
      assertArrayEquals(new float[] {0, 1, 3, 3}, sorted.toFloatArray());
      assertArrayEquals(new float[] {0, 1, 4, 7}, cumulative.toFloatArray());
      assertArrayEquals(values.toFloatArray(), restored.toFloatArray());
      assertArrayEquals(new int[] {1}, finite.toIntArray());
      assertEquals(
          Math.log(Math.exp(3) + Math.exp(1) + Math.exp(3) + 1), logSumExp.toFloatArray()[0], 1e-5);
    }
  }

  @Test
  void explicitKeysSplitDeterministicallyAndCategoricalUsesTheRequestedAxis() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray key = MLXRandom.key(scope, 42);
      MLXArray split = MLXRandom.split(key, 2, scope);
      MLXArray logits = MLX.array(scope, new float[] {1, 3, 2, 0, 5, 4, 3, 0}, new int[] {2, 4});
      MLXArray selected = MLXRandom.categorical(logits, -1, key);

      MLX.eval(key, split, selected);

      assertEquals(DType.UINT32, key.dtype());
      assertArrayEquals(new int[] {2}, key.shape());
      assertArrayEquals(new int[] {0, 42}, MLX.astype(key, DType.INT32).toIntArray());
      assertArrayEquals(new int[] {2, 2}, split.shape());
      assertArrayEquals(
          new int[] {-1829035798, -615737125, 255383827, 267815257},
          MLX.astype(split, DType.INT32).toIntArray());
      assertArrayEquals(new int[] {1, 0}, selected.toIntArray());
    }
  }

  @Test
  void explicitRandomValidationFailsBeforeNativeCode() {
    try (MLXScope scope = new MLXScope();
        MLXScope unrelated = new MLXScope()) {
      MLXArray intKey = MLX.array(scope, new int[] {0, 42}, new int[] {2});
      MLXArray shortKey = MLX.astype(MLX.array(scope, new int[] {42}, new int[] {1}), DType.UINT32);
      final MLXArray logits = MLX.array(scope, new float[] {1, 2}, new int[] {1, 2});

      assertThrows(IllegalArgumentException.class, () -> MLXRandom.split(intKey, 2, scope));
      assertThrows(IllegalArgumentException.class, () -> MLXRandom.split(shortKey, 2, scope));
      assertThrows(
          IllegalArgumentException.class, () -> MLXRandom.split(MLXRandom.key(scope, 1), 0, scope));
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXRandom.categorical(logits, 2, MLXRandom.key(scope, 1)));
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXRandom.split(MLXRandom.key(scope, 1), 2, unrelated));
    }
  }
}
