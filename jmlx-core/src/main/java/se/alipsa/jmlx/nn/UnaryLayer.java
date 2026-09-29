package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.memory.MLXScope;

/** A registered module that maps one array to another. */
public abstract class UnaryLayer extends Module implements UnaryModule {

  /** Creates a layer owned by {@code scope}. */
  protected UnaryLayer(MLXScope scope) {
    super(scope);
  }
}
