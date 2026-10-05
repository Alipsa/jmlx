package se.alipsa.jmlx.nn;

import java.util.stream.IntStream;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Eval-only channels-last batch normalization for ranks 2-4. Without tracking, eval recomputes
 * batch statistics. Registered running statistics are floating parameters: ModuleGrad includes them
 * even after freeze(), since per-key freezing/buffers are deferred to Phase 11.
 */
public final class BatchNorm extends Normalization {
  private final boolean trackRunningStats;

  /** Loaded tracking statistics, optional affine pair, and epsilon 1e-5. */
  public BatchNorm(
      MLXScope scope,
      int channels,
      MLXArray mean,
      MLXArray variance,
      MLXArray weight,
      MLXArray bias) {
    this(scope, channels, 1e-5f, true, mean, variance, weight, bias);
  }

  /** Explicit tracking option; statistics must be absent when tracking is false. */
  public BatchNorm(
      MLXScope scope,
      int channels,
      float epsilon,
      boolean trackRunningStats,
      MLXArray mean,
      MLXArray variance,
      MLXArray weight,
      MLXArray bias) {
    super(scope, channels, epsilon, weight, bias);
    this.trackRunningStats = trackRunningStats;
    if (trackRunningStats) {
      if (mean == null || variance == null) {
        throw new IllegalArgumentException("BatchNorm: tracking requires running statistics");
      }
      vector("running_mean", mean);
      vector("running_var", variance);
    } else if (mean != null || variance != null) {
      throw new IllegalArgumentException("BatchNorm: statistics supplied with tracking disabled");
    }
  }

  @Override
  public MLXArray forward(MLXArray x) {
    if (isTraining()) {
      throw new UnsupportedOperationException("BatchNorm training requires Phase 11");
    }
    validate(x, 2, 4);
    int[] axes = IntStream.range(0, x.ndim() - 1).toArray();
    MLXArray mean = trackRunningStats ? param("running_mean") : MLXOps.mean(x, axes, true);
    MLXArray variance = trackRunningStats ? param("running_var") : MLXOps.varAxes(x, axes, true, 0);
    // Subtract the activation first: parameter-only arithmetic must not grow the model scope.
    MLXArray centered = MLXOps.subtract(x, mean);
    MLXArray broadcastVariance = MLXShape.broadcastTo(variance, x.scope(), x.shape());
    return affine(
        MLXOps.multiply(
            centered, MLXOps.rsqrt(MLXOps.add(broadcastVariance, LayerOps.scalar(x, epsilon)))));
  }
}
