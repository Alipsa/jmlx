package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXConv;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Causal depthwise convolution and gated delta attention with request-owned FLOAT32 state. */
public final class GatedDeltaNet extends Module {
  /** Explicit prototype selection; no production prefill default is implicit. */
  public enum PrefillMethod {
    /** Evaluated per-token state updates. */
    RECURRENT,
    /** Evaluated chunks of at most 64 real positions. */
    CHUNKED
  }

  private final int keyHeads;
  private final int valueHeads;
  private final int keyDim;
  private final int valueDim;
  private final int keyWidth;
  private final int valueWidth;
  private final int channels;
  private final int kernelSize;
  private final UnaryLayer qkvProjection;
  private final UnaryLayer gateProjection;
  private final UnaryLayer betaProjection;
  private final UnaryLayer decayProjection;
  private final RMSNorm norm;
  private final UnaryLayer outputProjection;

  /**
   * Constructs the component from unpacked or affine projection layers and FLOAT32 state weights.
   */
  public GatedDeltaNet(
      MLXScope scope,
      int keyHeads,
      int valueHeads,
      int keyDim,
      int valueDim,
      UnaryLayer qkvProjection,
      UnaryLayer gateProjection,
      UnaryLayer betaProjection,
      UnaryLayer decayProjection,
      MLXArray convolutionWeight,
      MLXArray decayLog,
      MLXArray decayBias,
      MLXArray normWeight,
      float epsilon,
      UnaryLayer outputProjection) {
    super(scope);
    if (keyHeads < 1
        || valueHeads < 1
        || keyDim < 1
        || valueDim < 1
        || valueHeads % keyHeads != 0) {
      throw new IllegalArgumentException(
          "DeltaNet requires positive dimensions and divisible value/key heads");
    }
    if (!Float.isFinite(epsilon) || epsilon <= 0) {
      throw new IllegalArgumentException("DeltaNet epsilon must be finite and positive");
    }
    this.keyHeads = keyHeads;
    this.valueHeads = valueHeads;
    this.keyDim = keyDim;
    this.valueDim = valueDim;
    try {
      keyWidth = Math.multiplyExact(keyHeads, keyDim);
      valueWidth = Math.multiplyExact(valueHeads, valueDim);
      channels = Math.addExact(Math.multiplyExact(2, keyWidth), valueWidth);
    } catch (ArithmeticException error) {
      throw new IllegalArgumentException("DeltaNet dimension product overflow", error);
    }
    if (convolutionWeight.ndim() != 3
        || convolutionWeight.shape()[0] != channels
        || convolutionWeight.shape()[1] < 1
        || convolutionWeight.shape()[2] != 1
        || convolutionWeight.dtype() != DType.FLOAT32) {
      throw new IllegalArgumentException(
          "DeltaNet convolution weight must be FLOAT32 [channels,K,1]");
    }
    kernelSize = convolutionWeight.shape()[1];
    validateVector(decayLog, valueHeads, "decayLog");
    validateVector(decayBias, valueHeads, "decayBias");
    validateVector(normWeight, valueDim, "normWeight");
    param("convolutionWeight", convolutionWeight);
    param("decayLog", decayLog);
    param("decayBias", decayBias);
    this.qkvProjection = child("qkvProjection", Objects.requireNonNull(qkvProjection));
    this.gateProjection = child("gateProjection", Objects.requireNonNull(gateProjection));
    this.betaProjection = child("betaProjection", Objects.requireNonNull(betaProjection));
    this.decayProjection = child("decayProjection", Objects.requireNonNull(decayProjection));
    norm = child("norm", new RMSNorm(scope, normWeight, epsilon));
    this.outputProjection = child("outputProjection", Objects.requireNonNull(outputProjection));
  }

  /** Creates this layer's empty cache in the request scope. */
  public RecurrentCache newCache(MLXScope request, int batch) {
    return new RecurrentCache(request, batch, valueHeads, keyDim, valueDim, kernelSize, channels);
  }

  /**
   * Evaluates the output and complete new state before retiring the old state. The caller supplies
   * the prefill algorithm explicitly; single-token decode always uses the token recurrence.
   */
  public MLXArray forward(MLXArray x, RecurrentCache cache, PrefillMethod prefillMethod) {
    Objects.requireNonNull(prefillMethod, "prefillMethod");
    if (x.ndim() != 3 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.dtype() != DType.FLOAT32) {
      throw new IllegalArgumentException("DeltaNet input must be nonempty FLOAT32 [B,T,hidden]");
    }
    int batch = x.shape()[0];
    int length = x.shape()[1];
    if (!Arrays.equals(cache.state().shape(), new int[] {batch, valueHeads, keyDim, valueDim})
        || !Arrays.equals(
            cache.convolution().shape(), new int[] {batch, kernelSize - 1, channels})) {
      throw new IllegalArgumentException("DeltaNet cache geometry does not match the layer/input");
    }
    if (cache.isPoisoned()) {
      throw new IllegalStateException("DeltaNet cache is poisoned; discard the request");
    }
    try {
      MLXArray projected = qkvProjection.forward(x);
      requireShape(projected, batch, length, channels);
      MLXArray convInput = MLXShape.concatenate(new MLXArray[] {cache.convolution(), projected}, 1);
      MLXArray conv =
          silu(MLXConv.conv1d(convInput, param("convolutionWeight"), 1, 0, 1, channels));
      MLXArray q =
          MLXShape.reshape(
              window(conv, 2, 0, keyWidth), new int[] {batch, length, keyHeads, keyDim});
      MLXArray k =
          MLXShape.reshape(
              window(conv, 2, keyWidth, 2 * keyWidth), new int[] {batch, length, keyHeads, keyDim});
      final MLXArray v =
          MLXShape.reshape(
              window(conv, 2, 2 * keyWidth, channels),
              new int[] {batch, length, valueHeads, valueDim});
      q = MLXOps.multiply(l2Normalize(q), scalar(x.scope(), (float) (1 / Math.sqrt(keyDim))));
      k = l2Normalize(k);
      MLXArray decayInput = decayProjection.forward(x);
      MLXArray betaInput = betaProjection.forward(x);
      requireShape(decayInput, batch, length, valueHeads);
      requireShape(betaInput, batch, length, valueHeads);
      MLXArray shifted = MLXOps.add(decayInput, param("decayBias"));
      MLXArray softplus =
          MLXOps.add(
              MLXOps.maximum(shifted, scalar(x.scope(), 0f)),
              MLXOps.log1p(MLXOps.exp(MLXOps.negative(MLXOps.abs(shifted)))));
      MLXArray rate = MLXOps.negative(MLXOps.exp(inScope(param("decayLog"), x.scope())));
      MLXArray logDecay = MLXOps.multiply(rate, softplus);
      MLXArray beta = MLXOps.sigmoid(betaInput);
      GatedDeltaRule.Result result =
          length == 1 || prefillMethod == PrefillMethod.RECURRENT
              ? GatedDeltaRule.recurrent(q, k, v, logDecay, beta, cache.state())
              : GatedDeltaRule.chunked(q, k, v, logDecay, beta, cache.state());
      MLXArray gate = gateProjection.forward(x);
      requireShape(gate, batch, length, valueWidth);
      gate = MLXShape.reshape(gate, new int[] {batch, length, valueHeads, valueDim});
      MLXArray gated = MLXOps.multiply(norm.forward(result.output()), silu(gate));
      MLXArray output =
          outputProjection.forward(MLXShape.reshape(gated, new int[] {batch, length, valueWidth}));
      requireShape(output, batch, length, x.shape()[2]);
      MLXArray history = window(convInput, 1, length, length + kernelSize - 1);
      MLX.eval(output, history, result.state());
      cache.replace(history, result.state(), length);
      return output;
    } catch (RuntimeException | Error error) {
      cache.poison();
      throw error;
    }
  }

  private static MLXArray l2Normalize(MLXArray x) {
    MLXArray squared = MLXOps.sum(MLXOps.square(x), new int[] {3}, true);
    return MLXOps.multiply(x, MLXOps.rsqrt(MLXOps.add(squared, scalar(x.scope(), 1e-6f))));
  }

  private static MLXArray scalar(MLXScope scope, float value) {
    return MLX.full(scope, new int[] {1}, value, DType.FLOAT32);
  }

  private static MLXArray silu(MLXArray x) {
    return MLXOps.multiply(x, MLXOps.sigmoid(x));
  }

  private static MLXArray inScope(MLXArray a, MLXScope target) {
    return MLXShape.reshape(MLXShape.expandDims(a, target, 0), a.shape());
  }

  private static MLXArray window(MLXArray a, int axis, int start, int end) {
    int[] starts = new int[a.ndim()];
    int[] stops = a.shape();
    starts[axis] = start;
    stops[axis] = end;
    return MLXShape.slice(a, starts, stops);
  }

  private static void requireShape(MLXArray a, int... expected) {
    if (!Arrays.equals(a.shape(), expected)) {
      throw new IllegalArgumentException(
          "DeltaNet projection shape mismatch: expected " + Arrays.toString(expected));
    }
  }

  private static void validateVector(MLXArray a, int length, String name) {
    if (a.dtype() != DType.FLOAT32 || !Arrays.equals(a.shape(), new int[] {length})) {
      throw new IllegalArgumentException(name + " must be a FLOAT32 vector of length " + length);
    }
  }
}
