package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Request-owned FLOAT32 convolution history and delta-rule state. State uses [B,Hv,Dk,Dv],
 * convolution history [B,K-1,C]. A recurrent prefix cannot be truncated to a tail.
 *
 * <p>Returned arrays are borrowed: callers must not close them. Replacements are retained in the
 * request scope before the previous handles are retired. A failed native update poisons this cache;
 * discard the request rather than reusing partially advanced layer state.
 */
public final class RecurrentCache implements AutoCloseable {
  private final MLXScope scope;
  private final int[] convolutionShape;
  private final int[] stateShape;
  private final long logicalStorageBytes;
  private MLXArray convolution;
  private MLXArray state;
  private int position;
  private boolean poisoned;
  private boolean closed;

  /** Creates zero state with checked dimensions, owned by the supplied request scope. */
  public RecurrentCache(
      MLXScope scope,
      int batch,
      int valueHeads,
      int keyDim,
      int valueDim,
      int kernelSize,
      int convolutionChannels) {
    this.scope = Objects.requireNonNull(scope, "scope");
    for (int dimension :
        new int[] {batch, valueHeads, keyDim, valueDim, kernelSize, convolutionChannels}) {
      if (dimension < 1) {
        throw new IllegalArgumentException("recurrent cache dimensions must be positive");
      }
    }
    convolutionShape = new int[] {batch, kernelSize - 1, convolutionChannels};
    stateShape = new int[] {batch, valueHeads, keyDim, valueDim};
    try {
      logicalStorageBytes =
          Math.multiplyExact(4L, Math.addExact(elements(convolutionShape), elements(stateShape)));
    } catch (ArithmeticException error) {
      throw new IllegalArgumentException("recurrent cache size overflow", error);
    }
    convolution = MLX.zeros(scope, convolutionShape, DType.FLOAT32);
    try {
      state = MLX.zeros(scope, stateShape, DType.FLOAT32);
    } catch (RuntimeException | Error error) {
      convolution.close();
      throw error;
    }
  }

  /** Borrowed retained convolution history, excluding synthetic padding. */
  public MLXArray convolution() {
    requireOpen();
    return convolution;
  }

  /** Borrowed recurrent state in key-by-value orientation. */
  public MLXArray state() {
    requireOpen();
    return state;
  }

  /** Number of successfully committed real positions. */
  public int position() {
    requireOpen();
    return position;
  }

  /** Logical retained tensor bytes; this is not an allocator or process-memory bound. */
  public long logicalStorageBytes() {
    return logicalStorageBytes;
  }

  /** Whether a failed forward made further updates unsafe. */
  public boolean isPoisoned() {
    return poisoned;
  }

  /** Marks this cache unusable after any failure in its containing hybrid session. */
  public void poison() {
    poisoned = true;
  }

  /**
   * Evaluates and atomically replaces this layer's state, advancing by real input positions.
   * Validation precedes evaluation. Native/ownership failures poison the cache without advancing
   * its position. The containing session must also discard every other layer's state on failure.
   */
  public void replace(MLXArray nextConvolution, MLXArray nextState, int realPositions) {
    replaceState(nextConvolution, nextState, realPositions, true);
  }

  /**
   * Retains provisional inference state without evaluation; the caller evaluates all step roots.
   */
  void replaceDeferred(MLXArray nextConvolution, MLXArray nextState, int realPositions) {
    replaceState(nextConvolution, nextState, realPositions, false);
  }

  private void replaceState(
      MLXArray nextConvolution, MLXArray nextState, int realPositions, boolean evaluate) {
    requireOpen();
    if (poisoned) {
      throw new IllegalStateException("RecurrentCache is poisoned; discard the request");
    }
    validate(nextConvolution, convolutionShape, "convolution");
    validate(nextState, stateShape, "state");
    if (realPositions < 1) {
      throw new IllegalArgumentException("realPositions must be positive");
    }
    MLXScope inputScope = MLXScope.innermost(nextConvolution.scope(), nextState.scope());
    int nextPosition = Math.addExact(position, realPositions);
    MLXArray retainedConvolution = null;
    MLXArray retainedState = null;
    try (MLXScope retain = inputScope.newChild()) {
      // A child view followed by hoist also gives a distinct owned handle for same-scope inputs.
      MLXArray convView = inScope(nextConvolution, retain);
      MLXArray stateView = inScope(nextState, retain);
      if (evaluate) {
        MLX.eval(convView, stateView);
      }
      retainedConvolution = MLX.hoist(convView, scope);
      retainedState = MLX.hoist(stateView, scope);
      convolution.close();
      state.close();
      convolution = retainedConvolution;
      state = retainedState;
      position = nextPosition;
    } catch (RuntimeException | Error error) {
      if (retainedConvolution != null) {
        retainedConvolution.close();
      }
      if (retainedState != null) {
        retainedState.close();
      }
      poisoned = true;
      throw error;
    }
  }

  private void validate(MLXArray a, int[] expected, String name) {
    Objects.requireNonNull(a, name);
    if (a.dtype() != DType.FLOAT32 || !Arrays.equals(a.shape(), expected)) {
      throw new IllegalArgumentException(
          "recurrent " + name + " must be FLOAT32 " + Arrays.toString(expected));
    }
    if (!scope.isAncestorOf(a.scope())) {
      throw new IllegalArgumentException(
          "recurrent " + name + " must belong to the request scope or a child");
    }
  }

  private static MLXArray inScope(MLXArray a, MLXScope target) {
    return MLXShape.reshape(MLXShape.expandDims(a, target, 0), a.shape());
  }

  private static long elements(int[] shape) {
    long result = 1;
    for (int dimension : shape) {
      result = Math.multiplyExact(result, dimension);
    }
    if (result > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "recurrent cache tensor exceeds the supported element count");
    }
    return result;
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("RecurrentCache is closed");
    }
  }

  /** Releases retained handles; the owning request scope remains open. Idempotent. */
  @Override
  public void close() {
    if (!closed) {
      convolution.close();
      state.close();
      closed = true;
    }
  }
}
