package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * A token embedding table that can also act as a tied output head: {@link #forward} looks up rows
 * and {@link #project} multiplies by the table's transpose.
 */
public abstract class EmbeddingLayer extends UnaryLayer {

  /** Creates an embedding layer owned by {@code scope}. */
  protected EmbeddingLayer(MLXScope scope) {
    super(scope);
  }

  /**
   * Projects hidden states through this table's transpose: the output head of a model that ties its
   * input embeddings and output logits.
   */
  public abstract MLXArray project(MLXArray hiddenStates);
}
