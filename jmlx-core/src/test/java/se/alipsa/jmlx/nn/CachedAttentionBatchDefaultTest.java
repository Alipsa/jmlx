package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class CachedAttentionBatchDefaultTest {

  private static final class Legacy extends CachedAttention {
    Legacy(MLXScope scope) {
      super(scope);
    }

    @Override
    public MLXArray forward(MLXArray x, KVCache cache, MLXArray attentionMask) {
      return x;
    }
  }

  @Test
  void uniformBatchWithLegacyMaskDelegates() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.zeros(scope, new int[] {2, 1, 4}, DType.FLOAT32);
      MLXArray mask = MLX.zeros(scope, new int[] {1, 3}, DType.BOOL);
      assertSame(x, new Legacy(scope).forward(x, null, mask, null, new int[] {1, 1}));
    }
  }

  @Test
  void batchedMaskIsRejectedInsteadOfPassedToLegacyOverload() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.zeros(scope, new int[] {2, 1, 4}, DType.FLOAT32);
      MLXArray mask = MLX.zeros(scope, new int[] {2, 1, 1, 3}, DType.BOOL);
      Legacy attention = new Legacy(scope);
      assertThrows(
          UnsupportedOperationException.class,
          () -> attention.forward(x, null, mask, null, new int[] {1, 1}));
    }
  }
}
