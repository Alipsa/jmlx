package se.alipsa.jmlx.nn;

import java.util.stream.IntStream;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Per-example, per-channel spatial population normalization in channels-last layout. */
public final class InstanceNorm extends Normalization {
  /** Creates a non-affine normalization with epsilon 1e-5. */
  public InstanceNorm(MLXScope scope, int channels) {
    this(scope, channels, 1e-5f, null, null);
  }

  /** Creates normalization with optional checkpoint affine pair. */
  public InstanceNorm(MLXScope scope, int channels, float epsilon, MLXArray weight, MLXArray bias) {
    super(scope, channels, epsilon, weight, bias);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    validate(x, 3, Integer.MAX_VALUE);
    int[] axes = IntStream.range(1, x.ndim() - 1).toArray();
    return affine(normalize(x, MLXOps.mean(x, axes, true), MLXOps.varAxes(x, axes, true, 0)));
  }
}
