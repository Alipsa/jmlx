package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * An affine-quantized embedding table, in the {@code weight}/{@code scales}/{@code biases} layout
 * that MLX community checkpoints use. {@link #forward} gathers the packed rows, scales and biases
 * for the indices and dequantizes only those rows, so the full table is never expanded; {@link
 * #project} runs the fused quantized matmul against the transposed table, for a tied output head.
 *
 * <p>{@code weight} is {@code [numEmbeddings, packedDim]} {@code UINT32} with {@code packedDim =
 * dim * bits / 32}; {@code scales}/{@code biases} are each {@code [numEmbeddings, dim /
 * groupSize]}. As with {@link QuantizedLinear}, the shapes cannot prove that {@code groupSize} and
 * {@code bits} are the ones the table was quantized with; the caller pairs them.
 */
public final class QuantizedEmbedding extends EmbeddingLayer {
  private static final String MODE = "affine";

  private final int groupSize;
  private final int bits;

  /** Creates a quantized embedding over the given packed table. */
  public QuantizedEmbedding(
      MLXScope scope, MLXArray weight, MLXArray scales, MLXArray biases, int groupSize, int bits) {
    super(scope);
    Objects.requireNonNull(weight, "weight");
    Objects.requireNonNull(scales, "scales");
    Objects.requireNonNull(biases, "biases");
    if (weight.dtype() != se.alipsa.jmlx.core.DType.UINT32 || weight.ndim() != 2) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: weight must be rank-2 UINT32, got "
              + weight.dtype()
              + " "
              + Arrays.toString(weight.shape()));
    }
    if (!scales.dtype().isInexact() || !biases.dtype().isInexact()) {
      throw new IllegalArgumentException("QuantizedEmbedding: scales and biases must be floating");
    }
    if (scales.ndim() != 2
        || !Arrays.equals(scales.shape(), biases.shape())
        || scales.shape()[0] != weight.shape()[0]) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: scales/biases must be [numEmbeddings, dim / groupSize], got "
              + Arrays.toString(scales.shape())
              + " and "
              + Arrays.toString(biases.shape()));
    }
    if (groupSize != 32 && groupSize != 64 && groupSize != 128) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: groupSize must be one of {32, 64, 128}, got " + groupSize);
    }
    if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 && bits != 8) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: bits must be one of {2, 3, 4, 5, 6, 8}, got " + bits);
    }
    long packed = (long) scales.shape()[1] * groupSize * bits / 32;
    if (weight.shape()[1] != packed) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: weight has "
              + weight.shape()[1]
              + " packed columns but scales imply "
              + packed
              + " for groupSize "
              + groupSize
              + ", bits "
              + bits);
    }
    param("weight", weight);
    param("scales", scales);
    param("biases", biases);
    this.groupSize = groupSize;
    this.bits = bits;
  }

  @Override
  public MLXArray forward(MLXArray indices) {
    MLXArray rows = MLXShape.takeAxis(param("weight"), indices, 0);
    MLXArray rowScales = MLXShape.takeAxis(param("scales"), indices, 0);
    MLXArray rowBiases = MLXShape.takeAxis(param("biases"), indices, 0);
    return MLXQuant.dequantize(rows, rowScales, rowBiases, groupSize, bits, MODE, null, null);
  }

  @Override
  public MLXArray project(MLXArray hiddenStates) {
    return MLXQuant.quantizedMatmul(
        hiddenStates,
        param("weight"),
        param("scales"),
        param("biases"),
        true,
        groupSize,
        bits,
        MODE,
        hiddenStates.scope());
  }
}
