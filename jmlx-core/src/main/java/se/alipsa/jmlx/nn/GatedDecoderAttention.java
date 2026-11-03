package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Full causal attention with per-head query/gate splitting and partial multi-axis rotation. */
public final class GatedDecoderAttention extends Module {
  private final int heads;
  private final int keyValueHeads;
  private final int headDim;
  private final int rotaryDim;
  private final int width;
  private final float scale;
  private final UnaryLayer queryProjection;
  private final UnaryLayer keyProjection;
  private final UnaryLayer valueProjection;
  private final UnaryLayer outputProjection;
  private final RMSNorm queryNorm;
  private final RMSNorm keyNorm;

  /** Constructs attention with already shifted ordinary learned Q/K norm weights. */
  public GatedDecoderAttention(
      MLXScope scope,
      int heads,
      int keyValueHeads,
      int headDim,
      int rotaryDim,
      UnaryLayer queryProjection,
      UnaryLayer keyProjection,
      UnaryLayer valueProjection,
      UnaryLayer outputProjection,
      MLXArray queryNormWeight,
      MLXArray keyNormWeight,
      float epsilon) {
    super(scope);
    if (heads < 1
        || keyValueHeads < 1
        || heads % keyValueHeads != 0
        || headDim < 1
        || rotaryDim < 1
        || rotaryDim > headDim
        || rotaryDim % 2 != 0) {
      throw new IllegalArgumentException("invalid gated attention head/rotary dimensions");
    }
    if (!Float.isFinite(epsilon) || epsilon <= 0) {
      throw new IllegalArgumentException("attention epsilon must be positive and finite");
    }
    this.heads = heads;
    this.keyValueHeads = keyValueHeads;
    this.headDim = headDim;
    this.rotaryDim = rotaryDim;
    try {
      width = Math.multiplyExact(heads, headDim);
      Math.multiplyExact(width, 2);
      Math.multiplyExact(keyValueHeads, headDim);
    } catch (ArithmeticException error) {
      throw new IllegalArgumentException("attention dimension product overflow", error);
    }
    requireNorm(queryNormWeight, headDim);
    requireNorm(keyNormWeight, headDim);
    scale = (float) (1 / Math.sqrt(headDim));
    this.queryProjection = child("queryProjection", Objects.requireNonNull(queryProjection));
    this.keyProjection = child("keyProjection", Objects.requireNonNull(keyProjection));
    this.valueProjection = child("valueProjection", Objects.requireNonNull(valueProjection));
    this.outputProjection = child("outputProjection", Objects.requireNonNull(outputProjection));
    queryNorm = child("queryNorm", new RMSNorm(scope, queryNormWeight, epsilon));
    keyNorm = child("keyNorm", new RMSNorm(scope, keyNormWeight, epsilon));
  }

  /**
   * Builds full attention using already prepared T/H/W rotations. An optional BOOL [T,K] mask
   * replaces implicit causality. Cache updates are provisional until the session evaluates all
   * retained roots; late failure requires poisoning the complete request.
   */
  public MLXArray forward(MLXArray x, KVCache cache, MLXArray mask, MultiAxisRotation rotation) {
    Objects.requireNonNull(rotation, "rotation");
    int[] shape = x.shape();
    if (shape.length != 3 || shape[0] < 1 || shape[1] < 1 || x.dtype() != DType.FLOAT32) {
      throw new IllegalArgumentException("gated attention input must be FLOAT32 [B,T,hidden]");
    }
    int batch = shape[0];
    int sequence = shape[1];
    rotation.validateAgainst(batch, sequence, headDim, rotaryDim);
    long past = cache == null ? 0 : cache.length();
    long ending = past + sequence;
    if (ending > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("attention position overflow");
    }
    if (cache != null) {
      if (cache.isPoisoned()) {
        throw new IllegalStateException("attention cache is poisoned");
      }
      if (cache.policy().evicts()
          || !cache.isUniform()
          || (cache.batchSize() > 0 && cache.rowLength(0) != cache.length())) {
        throw new IllegalArgumentException("gated attention requires uniform full cache retention");
      }
      cache.policy().requireCapacity(ending, "attention positions");
    }
    if (mask != null
        && (mask.dtype() != DType.BOOL
            || !Arrays.equals(mask.shape(), new int[] {sequence, (int) ending}))) {
      throw new IllegalArgumentException("attention mask must be BOOL [T,K]");
    }
    try {
      MLXArray projected = queryProjection.forward(x);
      requireShape(projected, batch, sequence, 2 * width);
      projected = MLXShape.reshape(projected, new int[] {batch, sequence, heads, 2 * headDim});
      // Split within each head. Global flattened halves pair different heads with their gates.
      MLXArray query = window(projected, 3, 0, headDim);
      final MLXArray gate = window(projected, 3, headDim, 2 * headDim);
      query = MLXShape.transpose(queryNorm.forward(query), new int[] {0, 2, 1, 3});
      MLXArray key =
          AttentionHeads.toHeads(keyProjection.forward(x), batch, sequence, keyValueHeads, headDim);
      key = keyNorm.forward(key);
      MLXArray value =
          AttentionHeads.toHeads(
              valueProjection.forward(x), batch, sequence, keyValueHeads, headDim);
      query = rotation.rotate(query);
      key = rotation.rotate(key);
      if (cache != null) {
        cache.append(key, value);
        key = cache.keys();
        value = cache.values();
      }
      int repeat = heads / keyValueHeads;
      MLXArray attended =
          MLXFast.scaledDotProductAttention(
              query,
              AttentionHeads.repeatKeyValueHeads(key, x.scope(), keyValueHeads, repeat),
              AttentionHeads.repeatKeyValueHeads(value, x.scope(), keyValueHeads, repeat),
              scale,
              mask == null,
              mask,
              null);
      MLXArray merged =
          MLXShape.reshape(
              MLXShape.transpose(attended, new int[] {0, 2, 1, 3}),
              new int[] {batch, sequence, width});
      MLXArray flattenedGate = MLXShape.reshape(gate, new int[] {batch, sequence, width});
      MLXArray output =
          outputProjection.forward(MLXOps.multiply(merged, MLXOps.sigmoid(flattenedGate)));
      requireShape(output, batch, sequence, shape[2]);
      return output;
    } catch (RuntimeException | Error error) {
      if (cache != null) {
        cache.poison();
      }
      throw error;
    }
  }

  private static void requireNorm(MLXArray weight, int width) {
    if (weight.dtype() != DType.FLOAT32 || !Arrays.equals(weight.shape(), new int[] {width})) {
      throw new IllegalArgumentException("gated Q/K norm weight must be FLOAT32 [headDim]");
    }
  }

  private static void requireShape(MLXArray a, int... shape) {
    if (!Arrays.equals(a.shape(), shape)) {
      throw new IllegalArgumentException("gated projection dimension mismatch");
    }
  }

  private static MLXArray window(MLXArray a, int axis, int start, int end) {
    int[] stops = a.shape();
    int[] starts = new int[stops.length];
    starts[axis] = start;
    stops[axis] = end;
    return MLXShape.slice(a, starts, stops);
  }
}
