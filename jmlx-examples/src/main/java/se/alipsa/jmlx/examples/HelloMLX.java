package se.alipsa.jmlx.examples;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** req/initial-plan.md §8: the outline's demo, reduced to this slice's ops. */
public final class HelloMLX {

    private HelloMLX() {}

    static void main(String[] args) {
        try (MLXScope scope = new MLXScope()) {
            MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 2});
            MLXArray b = MLX.array(scope, new float[] {5, 6, 7, 8}, new int[] {2, 2});

            MLXArray sum = MLX.add(a, b);
            MLXArray product = MLX.matmul(a, b);

            // add()/matmul() only built the graph so far; eval() is what
            // actually runs them on device.
            MLX.eval(sum, product);

            System.out.println("a + b      = " + describe(sum));
            System.out.println("a matmul b = " + describe(product));
        }
    }

    private static String describe(MLXArray a) {
        return Arrays.toString(a.shape()) + " " + Arrays.toString(a.toFloatArray());
    }
}
