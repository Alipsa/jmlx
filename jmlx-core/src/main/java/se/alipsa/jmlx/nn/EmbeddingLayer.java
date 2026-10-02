package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;

/**
 * A token embedding table that can also act as a tied output head: {@link #forward} looks up rows
 * and {@link #project} multiplies by the table's transpose.
 */
public abstract class EmbeddingLayer extends Module implements UnaryModule {

  /** Creates an embedding layer owned by {@code scope}. */
  protected EmbeddingLayer(se.alipsa.jmlx.memory.MLXScope scope) {
    super(scope);
  }

  /**
   * Projects hidden states through this table's transpose: the output head of a model that ties its
   * input embeddings and output logits.
   */
  public abstract MLXArray project(MLXArray hiddenStates);
}
