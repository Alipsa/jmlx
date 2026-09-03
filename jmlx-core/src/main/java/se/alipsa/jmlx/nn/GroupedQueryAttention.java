package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Decoder self-attention with separate query/key/value projections and grouped-query attention
 * (GQA). This is the checkpoint layout used by Llama 3 and Qwen 2.5: queries have {@code numHeads}
 * heads while keys and values have {@code numKeyValueHeads} heads, each shared by a contiguous
 * group of query heads.
 */
public final class GroupedQueryAttention extends Module {

  private final int numHeads;
  private final int numKeyValueHeads;
  private final int headDim;
  private final int queriesPerKeyValueHead;
  private final float scale;
  private final float ropeTheta;
  private final Linear queryProj;
  private final Linear keyProj;
  private final Linear valueProj;
  private final Linear outProj;

  /**
   * Creates GQA from checkpoint-layout projection weights. Query and output weights are {@code
   * [embedDim, embedDim]}; key/value weights are {@code [numKeyValueHeads * headDim, embedDim]}.
   * Biases are optional, matching the different Llama/Qwen checkpoint conventions.
   */
  public GroupedQueryAttention(
      MLXScope scope,
      int numHeads,
      int numKeyValueHeads,
      float ropeTheta,
      MLXArray queryWeight,
      MLXArray queryBias,
      MLXArray keyWeight,
      MLXArray keyBias,
      MLXArray valueWeight,
      MLXArray valueBias,
      MLXArray outWeight,
      MLXArray outBias) {
    super(scope);
    Objects.requireNonNull(queryWeight, "GroupedQueryAttention: queryWeight must not be null");
    Objects.requireNonNull(keyWeight, "GroupedQueryAttention: keyWeight must not be null");
    Objects.requireNonNull(valueWeight, "GroupedQueryAttention: valueWeight must not be null");
    Objects.requireNonNull(outWeight, "GroupedQueryAttention: outWeight must not be null");
    if (queryWeight.ndim() != 2 || queryWeight.shape()[0] != queryWeight.shape()[1]) {
      throw new IllegalArgumentException(
          "GroupedQueryAttention: queryWeight must be square [embedDim, embedDim], got "
              + Arrays.toString(queryWeight.shape()));
    }
    int embedDim = queryWeight.shape()[0];
    if (numHeads <= 0 || numKeyValueHeads <= 0 || numHeads % numKeyValueHeads != 0) {
      throw new IllegalArgumentException(
          "GroupedQueryAttention: numHeads must be a positive multiple of numKeyValueHeads, got "
              + numHeads
              + " and "
              + numKeyValueHeads);
    }
    if (embedDim % numHeads != 0) {
      throw new IllegalArgumentException(
          "GroupedQueryAttention: embedDim "
              + embedDim
              + " must be divisible by numHeads "
              + numHeads);
    }
    int derivedHeadDim = embedDim / numHeads;
    int keyValueDim = numKeyValueHeads * derivedHeadDim;
    requireWeight("keyWeight", keyWeight, keyValueDim, embedDim);
    requireWeight("valueWeight", valueWeight, keyValueDim, embedDim);
    requireWeight("outWeight", outWeight, embedDim, embedDim);
    this.numHeads = numHeads;
    this.numKeyValueHeads = numKeyValueHeads;
    headDim = derivedHeadDim;
    queriesPerKeyValueHead = numHeads / numKeyValueHeads;
    scale = (float) (1.0 / Math.sqrt(headDim));
    this.ropeTheta = ropeTheta;
    queryProj = child("queryProj", new Linear(scope, queryWeight, queryBias));
    keyProj = child("keyProj", new Linear(scope, keyWeight, keyBias));
    valueProj = child("valueProj", new Linear(scope, valueWeight, valueBias));
    outProj = child("outProj", new Linear(scope, outWeight, outBias));
  }

  /**
   * Applies causal decoder attention to {@code x} ({@code [batch, sequence, embedDim]}). When
   * {@code cache} is supplied, it retains the un-repeated key/value heads across calls; expansion
   * to query-head count occurs only for the attention operation.
   */
  public MLXArray forward(MLXArray x, KVCache cache) {
    Objects.requireNonNull(x, "GroupedQueryAttention.forward: x must not be null");
    int[] shape = x.shape();
    if (shape.length != 3 || shape[2] != numHeads * headDim) {
      throw new IllegalArgumentException(
          "GroupedQueryAttention.forward: x must be [batch, sequence, embedDim="
              + (numHeads * headDim)
              + "], got "
              + Arrays.toString(shape));
    }
    int batch = shape[0];
    int sequence = shape[1];
    int offset = cache == null ? 0 : cache.offset();
    MLXArray q = toHeads(queryProj.forward(x), batch, sequence, numHeads);
    MLXArray k = toHeads(keyProj.forward(x), batch, sequence, numKeyValueHeads);
    MLXArray v = toHeads(valueProj.forward(x), batch, sequence, numKeyValueHeads);
    q = MLXFast.rope(q, headDim, false, ropeTheta, 1.0f, offset, null);
    k = MLXFast.rope(k, headDim, false, ropeTheta, 1.0f, offset, null);
    if (cache != null) {
      cache.append(k, v);
      k = cache.keys();
      v = cache.values();
    }
    MLXArray attended =
        MLXFast.scaledDotProductAttention(
            q,
            repeatKeyValueHeads(k, x.scope()),
            repeatKeyValueHeads(v, x.scope()),
            scale,
            true,
            null,
            null);
    MLXArray merged = MLXShape.flatten(MLXShape.transpose(attended, new int[] {0, 2, 1, 3}), 2, 3);
    return outProj.forward(merged);
  }

  private MLXArray toHeads(MLXArray projected, int batch, int sequence, int heads) {
    return MLXShape.transpose(
        MLXShape.reshape(projected, new int[] {batch, sequence, heads, headDim}),
        new int[] {0, 2, 1, 3});
  }

  private MLXArray repeatKeyValueHeads(MLXArray value, MLXScope target) {
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

  private static void requireWeight(String name, MLXArray weight, int out, int in) {
    if (weight.ndim() != 2 || weight.shape()[0] != out || weight.shape()[1] != in) {
      throw new IllegalArgumentException(
          "GroupedQueryAttention: "
              + name
              + " must be ["
              + out
              + ", "
              + in
              + "], got "
              + Arrays.toString(weight.shape()));
    }
  }
}
