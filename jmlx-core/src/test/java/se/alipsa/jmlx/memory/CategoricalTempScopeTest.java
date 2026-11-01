package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXRandom;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/**
 * PR #71 review finding: {@code MLXRandom.categorical} reshapes single-row logits to the 1-D view
 * the pinned sampler requires, and the no-target {@code MLXShape.reshape} allocates that view in
 * {@code logits.scope()}. With a long-lived logits scope and a per-draw child key scope, every draw
 * then left one reshape handle registered in the parent after the child closed -- unbounded
 * handle/native-metadata accumulation on exactly the reuse pattern this routing exists for. The
 * view must be allocated in the resolved scope (the innermost of the operands' scopes), so the draw
 * scope's close releases it.
 */
@EnabledIfNativeAvailable
class CategoricalTempScopeTest {

  @Test
  void singleRowViewAndResultStayInTheDrawScope() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray logits = MLX.array(parent, new float[] {1, 2, 3, 4}, new int[] {1, 4});
      int parentHandles = parent.handleCount();
      for (int draw = 0; draw < 8; draw++) {
        try (MLXScope drawScope = parent.newChild()) {
          MLXArray key = MLXRandom.key(drawScope, draw);
          MLXArray selected = MLXRandom.categorical(logits, -1, key);
          assertSame(drawScope, selected.scope());
          assertArrayEquals(new int[] {1}, selected.shape());
          assertEquals(DType.INT32, selected.dtype());
        }
      }
      // Each draw's key, views, and result all live in (and die with) the draw scope: the parent
      // must be exactly where it started. The pre-fix code registered one reshape view per draw
      // here, so this is the regression pin for the finding.
      assertEquals(parentHandles, parent.handleCount());
    }
  }
}
