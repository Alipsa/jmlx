package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * The SiLU (a.k.a. swish) activation: {@code x * sigmoid(x)}. No parameters -- {@code scope} is accepted only to
 * satisfy {@link Module}'s constructor contract; nothing is registered.
 */
public final class SiLU extends Module implements UnaryModule {

  /** Creates a {@code SiLU} activation layer. */
  public SiLU(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.multiply(x, MLXOps.sigmoid(x));
  }
}
