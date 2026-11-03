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
  // Native constants hoisted out of forward(): per-token decode must not pay for per-call
  // scalar creation or a fresh view of the decay log. An operand in the (ancestor) module scope
  // is legal, and these values are immutable once the weights are fixed.
  private final MLXArray keyScale;
  private final MLXArray zero;
  private final MLXArray l2Epsilon;
  private final MLXArray decayRate;

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
    keyScale = scalar(scope, (float) (1 / Math.sqrt(keyDim)));
    zero = scalar(scope, 0f);
    l2Epsilon = scalar(scope, 1e-6f);
    MLXArray exp = MLXOps.exp(decayLog);
    decayRate = MLXOps.negative(exp);
    // Every array the model keeps is evaluated during loading; these constants are no
    // exception, so the first request pays no one-time evaluation of them.
    MLX.eval(keyScale, zero, l2Epsilon, decayRate);
    // exp is a construction input, not a model array: release it once decayRate is
    // evaluated from it, so the model's retained set is exactly its parameters.
    exp.close();
  }

  /** Creates this layer's empty cache in the request scope. */
  public RecurrentCache newCache(MLXScope request, int batch) {
    return new RecurrentCache(request, batch, valueHeads, keyDim, valueDim, kernelSize, channels);
  }

  /**
   * Builds output and retains provisional request state. The caller must evaluate output and all
   * retained cache arrays together before committing a request step; a later evaluation failure
   * poisons the entire session. Single-token decode inserts no forced evaluation. The caller
   * supplies the prefill algorithm explicitly.
   */
  public MLXArray forward(MLXArray x, RecurrentCache cache, PrefillMethod prefillMethod) {
    return forward(x, cache, prefillMethod, () -> {});
  }

  /** Builds a step with cooperative checks before projections and between chunked prefill scans. */
  public MLXArray forward(
      MLXArray x, RecurrentCache cache, PrefillMethod prefillMethod, Runnable checkpoint) {
    Objects.requireNonNull(checkpoint, "checkpoint");
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
      checkpoint.run();
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
      q = MLXOps.multiply(l2Normalize(q), keyScale);
      k = l2Normalize(k);
      MLXArray decayInput = decayProjection.forward(x);
      MLXArray betaInput = betaProjection.forward(x);
      requireShape(decayInput, batch, length, valueHeads);
      requireShape(betaInput, batch, length, valueHeads);
      MLXArray shifted = MLXOps.add(decayInput, param("decayBias"));
      MLXArray softplus =
          MLXOps.add(
              MLXOps.maximum(shifted, zero),
              MLXOps.log1p(MLXOps.exp(MLXOps.negative(MLXOps.abs(shifted)))));
      MLXArray logDecay = MLXOps.multiply(decayRate, softplus);
      MLXArray beta = MLXOps.sigmoid(betaInput);
      GatedDeltaRule.Result result =
          length == 1
              ? GatedDeltaRule.step(q, k, v, logDecay, beta, cache.state())
              : prefillMethod == PrefillMethod.RECURRENT
                  ? GatedDeltaRule.recurrent(q, k, v, logDecay, beta, cache.state())
                  : GatedDeltaRule.chunked(q, k, v, logDecay, beta, cache.state(), checkpoint);
      MLXArray gate = gateProjection.forward(x);
      requireShape(gate, batch, length, valueWidth);
      gate = MLXShape.reshape(gate, new int[] {batch, length, valueHeads, valueDim});
      MLXArray gated = MLXOps.multiply(norm.forward(result.output()), silu(gate));
      MLXArray output =
          outputProjection.forward(MLXShape.reshape(gated, new int[] {batch, length, valueWidth}));
      requireShape(output, batch, length, x.shape()[2]);
      // Gather only K-1 real rows: a contiguous slice may retain the complete projected prefill.
      // Gather cannot donate its arbitrarily indexed input, and preserves every FLOAT32 bit.
      MLXArray history =
          MLXShape.takeAxis(
              convInput, MLX.arange(x.scope(), length, length + kernelSize - 1, 1, DType.INT32), 1);
      cache.replaceDeferred(history, result.state(), length);
      return output;
    } catch (RuntimeException | Error error) {
      cache.poison();
      throw error;
    }
  }

  private MLXArray l2Normalize(MLXArray x) {
    MLXArray squared = MLXOps.sum(MLXOps.square(x), new int[] {3}, true);
    return MLXOps.multiply(x, MLXOps.rsqrt(MLXOps.add(squared, l2Epsilon)));
  }

  private static MLXArray scalar(MLXScope scope, float value) {
    return MLX.full(scope, new int[] {1}, value, DType.FLOAT32);
  }

  private static MLXArray silu(MLXArray x) {
    return MLXOps.multiply(x, MLXOps.sigmoid(x));
  }

  private static MLXArray window(MLXArray a, int axis, int start, int end) {
    int[] stops = a.shape();
    int[] starts = new int[stops.length];
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
