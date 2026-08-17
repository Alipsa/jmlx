package se.alipsa.jmlx.examples;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Linear;

/**
 * See req/initial-plan.md §8: the outline's demo, reduced to this slice's ops. Also demonstrates a
 * Phase 4 {@code Linear} forward pass across a model/step scope split -- see req/phase4-plan.md §5
 * for the scoping rationale that pass exists to illustrate.
 */
public final class HelloMLX {

  private HelloMLX() {}

  static void main(String[] args) {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray b = MLX.array(scope, new float[] {5, 6, 7, 8}, new int[] {2, 2});
      MLXArray row = MLX.array(scope, new float[] {10, 20}, new int[] {2});

      MLXArray sum = MLXOps.add(a, b);
      MLXArray product = MLXOps.matmul(a, b);
      // row's shape [2] differs from a's [2, 2] but broadcasts against it
      // (NumPy rules): row is treated as [[10, 20], [10, 20]] and added
      // to each row of a. The old requireSameShape guard would have
      // rejected this pairing outright.
      MLXArray broadcastSum = MLXOps.add(a, row);

      // add()/matmul() only built the graph so far; eval() is what
      // actually runs them on device -- all three arrays at once, in a
      // single native call.
      MLX.eval(sum, product, broadcastSum);

      System.out.println("a + b       = " + describe(sum));
      System.out.println("a matmul b  = " + describe(product));
      System.out.println("a + row     = " + describe(broadcastSum));

      // Phase 4: a Linear layer's weight/bias live in this outer (model) scope,
      // but forward() is called on an activation from a child (step) scope --
      // demonstrating both of Phase 4's behavioural changes over M0a/M0b-era ops
      // above: cross-scope op resolution (scopeOf/innermost) and child-scope
      // lifetime, without depending on quantization specifically.
      MLXArray weight = MLX.array(scope, new float[] {1, 0, 1, 0, 1, 1}, new int[] {2, 3});
      MLXArray bias = MLX.array(scope, new float[] {10, 20}, new int[] {2});
      Linear linear = new Linear(scope, weight, bias);
      try (MLXScope step = scope.newChild()) {
        MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
        MLXArray y = linear.forward(x);
        MLX.eval(y);
        System.out.println("linear.forward(x) = " + describe(y));
      }
    }
  }

  private static String describe(MLXArray a) {
    return Arrays.toString(a.shape()) + " " + Arrays.toString(a.toFloatArray());
  }
}
