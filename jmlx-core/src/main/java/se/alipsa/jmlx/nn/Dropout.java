package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** Inference-only dropout. Eval is identity; training is deferred to Phase 11. */
public final class Dropout extends UnaryLayer {
  private final float probability;

  /** Creates dropout with probability 0.5. */
  public Dropout(MLXScope scope) {
    this(scope, 0.5f);
  }

  /** Probability must be in [0, 1), including finite zero. */
  public Dropout(MLXScope scope, float probability) {
    super(scope);
    if (!(probability >= 0 && probability < 1)) {
      throw new IllegalArgumentException("Dropout: probability must be in [0, 1)");
    }
    this.probability = probability;
  }

  /** Configured drop probability. */
  public float probability() {
    return probability;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    if (isTraining()) {
      throw new UnsupportedOperationException("Dropout training requires Phase 11");
    }
    x.scope();
    return x;
  }
}
