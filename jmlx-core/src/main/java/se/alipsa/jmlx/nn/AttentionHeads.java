package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Head reshaping shared by the cached attention implementations. */
final class AttentionHeads {
  private AttentionHeads() {}

  /**
   * Reshapes {@code [batch, sequence, heads * headDim]} to {@code [batch, heads, sequence,
   * headDim]}.
   */
  static MLXArray toHeads(MLXArray projected, int batch, int sequence, int heads, int headDim) {
    int[] expected = {batch, sequence, heads * headDim};
    if (!Arrays.equals(projected.shape(), expected)) {
      throw new IllegalArgumentException(
          "projection must produce "
              + Arrays.toString(expected)
              + ", got "
              + Arrays.toString(projected.shape()));
    }
    return MLXShape.transpose(
        MLXShape.reshape(projected, new int[] {batch, sequence, heads, headDim}),
        new int[] {0, 2, 1, 3});
  }

  /** Expands key/value heads to the query-head count, allocating into {@code target}. */
  static MLXArray repeatKeyValueHeads(
      MLXArray value, MLXScope target, int numKeyValueHeads, int queriesPerKeyValueHead) {
    if (queriesPerKeyValueHead == 1) {
      return value;
    }
    int[] shape = value.shape();
    MLXArray expanded = MLXShape.expandDims(value, target, 2);
    MLXArray broadcast =
        MLXShape.broadcastTo(
            expanded,
            target,
            new int[] {shape[0], numKeyValueHeads, queriesPerKeyValueHead, shape[2], shape[3]});
    return MLXShape.flatten(broadcast, target, 1, 2);
  }
}
