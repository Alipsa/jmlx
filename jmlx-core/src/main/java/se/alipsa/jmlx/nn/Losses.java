package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.IntStream;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;

/** Evaluation losses matching pinned MLX 0.31.2; gradient guarantees are deferred to Phase 11. */
public final class Losses {
  /** Output reduction over every remaining element. */
  public enum Reduction {
    NONE,
    MEAN,
    SUM
  }

  private Losses() {}

  private static MLXArray reduce(MLXArray value, Reduction reduction) {
    return switch (Objects.requireNonNull(reduction)) {
      case NONE -> value;
      case SUM -> MLXOps.sum(value);
      case MEAN -> MLXOps.mean(value, IntStream.range(0, value.ndim()).toArray(), false);
    };
  }

  private static void sameShape(MLXArray x, MLXArray y) {
    if (!Arrays.equals(x.shape(), y.shape())) {
      throw new IllegalArgumentException("loss: inputs must have equal shapes");
    }
  }

  private static MLXArray weights(MLXArray loss, MLXArray weights) {
    if (weights == null) {
      return loss;
    }
    sameShape(loss, weights);
    return MLXOps.multiply(loss, weights);
  }

  private static int axis(MLXArray x, int axis) {
    int normalized = axis < 0 ? axis + x.ndim() : axis;
    if (normalized < 0 || normalized >= x.ndim()) {
      throw new IllegalArgumentException("loss: class axis out of range");
    }
    return normalized;
  }

  private static MLXArray gather(MLXArray x, MLXArray targets, int axis) {
    int normalized = axis(x, axis);
    axis = normalized;
    int[] expected =
        IntStream.range(0, x.ndim()).filter(i -> i != normalized).map(i -> x.shape()[i]).toArray();
    if (!Arrays.equals(expected, targets.shape()) || targets.dtype() != DType.INT32) {
      throw new IllegalArgumentException("loss: labels must be INT32 with class axis removed");
    }
    // Synchronizes tensor labels to validate before the native gather (which does not
    // bounds-check).
    for (int label : targets.toIntArray()) {
      if (label < 0 || label >= x.shape()[axis]) {
        throw new IllegalArgumentException("loss: label " + label + " outside class range");
      }
    }
    return MLXShape.squeeze(
        MLXShape.takeAlongAxis(x, MLXShape.expandDims(targets, axis), axis), new int[] {axis});
  }

  /** Cross entropy, class axis -1, no smoothing/weights, no reduction. */
  public static MLXArray crossEntropy(MLXArray logits, MLXArray targets) {
    return crossEntropy(logits, targets, null, -1, 0, Reduction.NONE);
  }

  /**
   * Integer class labels or same-shaped probabilities; smoothing in [0,1), per-output weights.
   * INT32 labels are synchronously validated on the host before gathering.
   */
  public static MLXArray crossEntropy(
      MLXArray logits,
      MLXArray targets,
      MLXArray weight,
      int classAxis,
      float smoothing,
      Reduction reduction) {
    int axis = axis(logits, classAxis);
    if (!(smoothing >= 0 && smoothing < 1)) {
      throw new IllegalArgumentException("crossEntropy: smoothing must be in [0,1)");
    }
    MLXArray score;
    if (targets.ndim() == logits.ndim()) {
      sameShape(logits, targets);
      score = MLXOps.sum(MLXOps.multiply(logits, targets), new int[] {axis}, false);
    } else {
      score = gather(logits, targets, axis);
    }
    MLXArray loss =
        MLXOps.subtract(
            MLXOps.logSumExpAxis(logits, axis, false),
            MLXOps.multiply(score, LayerOps.scalar(score, 1 - smoothing)));
    if (smoothing > 0) {
      loss =
          MLXOps.subtract(
              loss,
              MLXOps.multiply(
                  MLXOps.mean(logits, new int[] {axis}, false),
                  LayerOps.scalar(logits, smoothing)));
    }
    return reduce(weights(loss, weight), reduction);
  }

  /** NLL with class axis -1 and no reduction; validates and synchronizes INT32 labels. */
  public static MLXArray nll(MLXArray input, MLXArray targets) {
    return nll(input, targets, -1, Reduction.NONE);
  }

  /** General class-axis NLL. */
  public static MLXArray nll(MLXArray input, MLXArray targets, int axis, Reduction reduction) {
    return reduce(MLXOps.negative(gather(input, targets, axis)), reduction);
  }

  /** BCE defaults to logits and mean reduction. */
  public static MLXArray binaryCrossEntropy(MLXArray input, MLXArray targets) {
    return binaryCrossEntropy(input, targets, null, true, Reduction.MEAN);
  }

  /**
   * Same-shaped targets/weights. Probability logarithms are clipped at -100, including endpoints.
   */
  public static MLXArray binaryCrossEntropy(
      MLXArray input, MLXArray targets, MLXArray weight, boolean withLogits, Reduction reduction) {
    sameShape(input, targets);
    MLXArray loss;
    if (withLogits) {
      loss = MLXOps.subtract(LayerOps.softplus(input), MLXOps.multiply(input, targets));
    } else {
      MLXArray one = LayerOps.scalar(input, 1);
      MLXArray lower = LayerOps.scalar(input, -100);
      loss =
          MLXOps.negative(
              MLXOps.add(
                  MLXOps.multiply(targets, MLXOps.maximum(MLXOps.log(input), lower)),
                  MLXOps.multiply(
                      MLXOps.subtract(one, targets),
                      MLXOps.maximum(MLXOps.log(MLXOps.subtract(one, input)), lower))));
    }
    return reduce(weights(loss, weight), reduction);
  }

  /** Mean squared error; same-shaped inputs. */
  public static MLXArray mse(MLXArray predictions, MLXArray targets) {
    return mse(predictions, targets, Reduction.MEAN);
  }

  /** Squared error with explicit reduction. */
  public static MLXArray mse(MLXArray predictions, MLXArray targets, Reduction reduction) {
    sameShape(predictions, targets);
    return reduce(MLXOps.square(MLXOps.subtract(predictions, targets)), reduction);
  }

  /** Mean absolute error; same-shaped inputs. */
  public static MLXArray l1(MLXArray predictions, MLXArray targets) {
    return l1(predictions, targets, Reduction.MEAN);
  }

  /** Absolute error with explicit reduction. */
  public static MLXArray l1(MLXArray predictions, MLXArray targets, Reduction reduction) {
    sameShape(predictions, targets);
    return reduce(MLXOps.abs(MLXOps.subtract(predictions, targets)), reduction);
  }

  /** Smooth L1 with beta 1 and mean reduction. */
  public static MLXArray smoothL1(MLXArray predictions, MLXArray targets) {
    return smoothL1(predictions, targets, 1, Reduction.MEAN);
  }

  /** Pinned piecewise smooth-L1; beta zero selects L1. */
  public static MLXArray smoothL1(
      MLXArray predictions, MLXArray targets, float beta, Reduction reduction) {
    sameShape(predictions, targets);
    if (!(beta >= 0) || !Float.isFinite(beta)) {
      throw new IllegalArgumentException("smoothL1: invalid beta");
    }
    MLXArray difference = MLXOps.abs(MLXOps.subtract(predictions, targets));
    if (beta == 0) {
      return reduce(difference, reduction);
    }
    return reduce(
        MLXOps.where(
            MLXOps.less(difference, LayerOps.scalar(difference, beta)),
            MLXOps.multiply(MLXOps.square(difference), LayerOps.scalar(difference, 0.5f / beta)),
            MLXOps.subtract(difference, LayerOps.scalar(difference, 0.5f * beta))),
        reduction);
  }

  /** KL divergence takes both inputs and targets in log space, axis -1, no reduction. */
  public static MLXArray klDiv(MLXArray input, MLXArray logTargets) {
    return klDiv(input, logTargets, -1, Reduction.NONE);
  }

  /** Pinned exp(target)*(target-input), summed on axis; boundary NaNs are preserved. */
  public static MLXArray klDiv(MLXArray input, MLXArray logTargets, int axis, Reduction reduction) {
    return reduce(
        MLXOps.sum(
            MLXOps.multiply(MLXOps.exp(logTargets), MLXOps.subtract(logTargets, input)),
            new int[] {axis(input, axis)},
            false),
        reduction);
  }

  /** Cosine similarity (not negated), axis 1, epsilon 1e-8, no reduction. */
  public static MLXArray cosineSimilarity(MLXArray x, MLXArray y) {
    return cosineSimilarity(x, y, 1, 1e-8f, Reduction.NONE);
  }

  /** Norm-product denominator clipped to epsilon; explicit axis/reduction. */
  public static MLXArray cosineSimilarity(
      MLXArray x, MLXArray y, int axis, float epsilon, Reduction reduction) {
    int[] axes = {axis(x, axis)};
    MLXArray dot = MLXOps.sum(MLXOps.multiply(x, y), axes, false);
    MLXArray norm =
        MLXOps.multiply(
            MLXOps.sqrt(MLXOps.sum(MLXOps.square(x), axes, false)),
            MLXOps.sqrt(MLXOps.sum(MLXOps.square(y), axes, false)));
    return reduce(
        MLXOps.divide(dot, MLXOps.maximum(norm, LayerOps.scalar(dot, epsilon))), reduction);
  }
}
