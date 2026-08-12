package se.alipsa.jmlx.examples;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** See req/initial-plan.md §8: the outline's demo, reduced to this slice's ops. */
public final class HelloMLX {

  private HelloMLX() {}

  static void main(String[] args) {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
      MLXArray b = MLX.array(scope, new float[] {5, 6, 7, 8}, new int[] {2, 2});
      MLXArray row = MLX.array(scope, new float[] {10, 20}, new int[] {2});

      MLXArray sum = MLX.add(a, b);
      MLXArray product = MLX.matmul(a, b);
      // row's shape [2] differs from a's [2, 2] but broadcasts against it
      // (NumPy rules): row is treated as [[10, 20], [10, 20]] and added
      // to each row of a. The old requireSameShape guard would have
      // rejected this pairing outright.
      MLXArray broadcastSum = MLX.add(a, row);

      // add()/matmul() only built the graph so far; eval() is what
      // actually runs them on device -- all three arrays at once, in a
      // single native call.
      MLX.eval(sum, product, broadcastSum);

      System.out.println("a + b       = " + describe(sum));
      System.out.println("a matmul b  = " + describe(product));
      System.out.println("a + row     = " + describe(broadcastSum));
    }
  }

  private static String describe(MLXArray a) {
    return Arrays.toString(a.shape()) + " " + Arrays.toString(a.toFloatArray());
  }
}
