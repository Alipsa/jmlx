package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * An embedding table lookup: {@code forward(indices)} returns the row of {@code weight} at each
 * index, via {@link MLXShape#takeAxis(MLXArray, MLXArray, int)} along axis 0. {@code weight} has
 * shape {@code [numEmbeddings, dim]}.
 */
public final class Embedding extends Module implements UnaryModule {

  /**
   * Creates an {@code Embedding} layer with the given {@code weight} (shape {@code [numEmbeddings,
   * dim]}).
   */
  public Embedding(MLXScope scope, MLXArray weight) {
    super(scope);
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException(
          "Embedding: weight must be rank 2 [numEmbeddings, dim], got shape "
              + java.util.Arrays.toString(weight.shape()));
    }
    param("weight", weight);
  }

  @Override
  public MLXArray forward(MLXArray indices) {
    return MLXShape.takeAxis(param("weight"), indices, 0);
  }

  /**
   * Projects hidden states through this embedding table's transpose. This is the output head for a
   * model that ties input embeddings and output logits, without registering the same parameter in
   * two module subtrees.
   */
  public MLXArray project(MLXArray hiddenStates) {
    return MLXOps.matmul(hiddenStates, MLXShape.transpose(param("weight"), hiddenStates.scope()));
  }
}
