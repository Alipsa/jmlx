package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** {@link SiLU} and {@link GELU}: no parameters, so combined into one file rather than one apiece. */
@EnabledIfNativeAvailable
class ActivationsTest {

  // Irrational sigmoid/erf, same reasoning as MLXOpsTest's sigmoid()/erf().
  private static final float EPS = 1e-4f;

  @Test
  void siluMatchesXTimesSigmoidXGolden() {
    try (MLXScope scope = new MLXScope()) {
      SiLU silu = new SiLU(scope);
      MLXArray x = MLX.array(scope, new float[] {0, 1, -1}, new int[] {3});

      MLXArray result = silu.forward(x);

      assertArrayEquals(new float[] {0.0f, 0.7310586f, -0.2689414f}, result.toFloatArray(), EPS);
    }
  }

  /**
   * These are the standard-normal-CDF identity values ({@code GELU(x) = x * Phi(x)}, {@code Phi(1) = 0.8413447},
   * {@code Phi(-1) = 0.1586553}), but asserted against the plain {@code 0.5*x*(1+erf(x/sqrt2))} formula's numeric
   * result, since that is what {@link GELU#forward} actually computes.
   */
  @Test
  void geluMatchesTheExactFormGolden() {
    try (MLXScope scope = new MLXScope()) {
      GELU gelu = new GELU(scope);
      MLXArray x = MLX.array(scope, new float[] {0, 1, -1}, new int[] {3});

      MLXArray result = gelu.forward(x);

      assertArrayEquals(new float[] {0.0f, 0.8413447f, -0.1586553f}, result.toFloatArray(), EPS);
    }
  }
}
