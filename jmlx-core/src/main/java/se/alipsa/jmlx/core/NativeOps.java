package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.IntSupplier;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Package-private plumbing shared by every op class in {@code se.alipsa.jmlx.core}: the resolved default device/stream,
 * the status-checking helpers, and the {@code (res, operand..., stream)} op-body shapes ({@link #binaryOp},
 * {@link #unaryOp}, {@link #shapeOp}) that {@link MLXOps} and {@link MLXShape} build every op on. See
 * req/phase4-plan.md §1 -- this class is the split's foundation layer: every other class in this package may depend on
 * it, but it depends on nothing else here.
 */
final class NativeOps {

  private NativeOps() {}

  // Same reasoning as MLX's and MLXScope's static initializers: jextract
  // binds each downcall's method handle lazily, the first time that
  // function is actually called, and that first call fails unless the
  // dylib is already loaded by then. Every op in MLXOps/MLXShape (and,
  // later, MLXFast/MLXQuant/MLXRandom) reaches native only through this
  // class's checked()/binaryOp()/unaryOp()/shapeOp(), so guarding here
  // covers all of them transitively -- this is the one place a caller who
  // never happens to touch MLX or MLXScope directly could still be "first".
  static {
    NativeLoader.ensureLoaded();
  }

  private static final Arena FACADE_ARENA = Arena.ofShared();
  private static final MemorySegment DEFAULT_DEVICE = resolveDefaultDevice();
  static final MemorySegment DEFAULT_STREAM = resolveDefaultStream();

  private static MemorySegment resolveDefaultDevice() {
    MemorySegment dev = mlx_h.mlx_device_new(FACADE_ARENA);
    checked(() -> mlx_h.mlx_get_default_device(dev));
    return dev;
  }

  private static MemorySegment resolveDefaultStream() {
    MemorySegment stream = mlx_h.mlx_stream_new(FACADE_ARENA);
    checked(() -> mlx_h.mlx_get_default_stream(stream, DEFAULT_DEVICE));
    return stream;
  }

  /** Opaque {@code mlx_device} handle; valid for the process lifetime. Backs {@link MLX#defaultDevice()}. */
  static MemorySegment defaultDevice() {
    return DEFAULT_DEVICE;
  }

  static MLXArray binaryOp(String opName, MLXArray a, MLXArray b, BinaryOp op) {
    MLXScope scope = scopeOf(opName, a, b);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(opName, () -> op.apply(res, a.handle(), b.handle(), DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  /**
   * Innermost scope among all non-null {@code operands} (req/phase4-plan.md §2). Touches every non-null operand's
   * {@link MLXArray#scope()}, so each one's {@code checkAccess()} runs -- this is what stops a multi-operand op from
   * silently bypassing {@link MLXScope}'s confinement contract for every operand but the one its target happens to be
   * picked from.
   *
   * <p>
   * Precondition: at least one operand is non-null -- the caller always passes the primary input ({@code x}, {@code
   * q}, {@code w}), which is never nullable in any mlx-c signature. Violating it throws
   * {@link IllegalArgumentException} naming {@code op}, not an empty-reduce exception: the reachable case is a caller
   * (e.g. inside a {@code layerNorm} body) that forgot to pass its primary input, and the message should say that
   * rather than "empty".
   *
   * @throws IllegalArgumentException if any two operand scopes are unrelated (siblings, or two independent roots)
   */
  static MLXScope scopeOf(String op, MLXArray... operands) {
    MLXScope result = null;
    for (MLXArray a : operands) {
      if (a == null) {
        continue;
      }
      MLXScope s = a.scope();
      result = result == null ? s : MLXScope.innermost(result, s);
    }
    if (result == null) {
      throw new IllegalArgumentException(op + ": scopeOf requires at least one non-null operand");
    }
    return result;
  }

  @FunctionalInterface
  interface BinaryOp {
    int apply(MemorySegment res, MemorySegment a, MemorySegment b, MemorySegment stream);
  }

  static MLXArray unaryOp(String opName, MLXArray a, MLXScope target, UnaryOp op) {
    MemorySegment res = mlx_h.mlx_array_new(target);
    checked(opName, () -> op.apply(res, a.handle(), DEFAULT_STREAM));
    return new MLXArray(target, res);
  }

  static MLXArray unaryOp(String opName, MLXArray a, UnaryOp op) {
    return unaryOp(opName, a, a.scope(), op);
  }

  @FunctionalInterface
  interface UnaryOp {
    int apply(MemorySegment res, MemorySegment a, MemorySegment stream);
  }

  /**
   * Wraps the {@code (res, a, const int*, size_t, stream)} native shape shared by {@code mlx_broadcast_to},
   * {@code mlx_squeeze_axes} and {@code mlx_transpose_axes}: a confined {@link Arena} owns {@code param}'s native copy
   * for the lifetime of the call, exactly as {@code reshape} inlines it for {@code mlx_reshape}.
   */
  static MLXArray shapeOp(String opName, MLXArray a, int[] param, ShapeOp op) {
    MLXScope scope = a.scope();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeParam = tmp.allocateFrom(ValueLayout.JAVA_INT, param);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      checked(opName, () -> op.apply(res, a.handle(), nativeParam, param.length, DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  @FunctionalInterface
  interface ShapeOp {
    int apply(MemorySegment res, MemorySegment a, MemorySegment param, long paramNum, MemorySegment stream);
  }

  /**
   * Wraps the {@code (res, a, const int*, size_t, bool, stream)} native shape shared by {@code mlx_sum_axes} and
   * {@code mlx_mean_axes}: a confined {@link Arena} owns {@code axes}'s native copy for the lifetime of the call,
   * exactly as {@code shapeOp} does for its param.
   */
  static MLXArray reduceOp(String opName, MLXArray a, int[] axes, boolean keepdims, ReduceOp op) {
    MLXScope scope = a.scope();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeAxes = tmp.allocateFrom(ValueLayout.JAVA_INT, axes);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      checked(opName, () -> op.apply(res, a.handle(), nativeAxes, axes.length, keepdims, DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  @FunctionalInterface
  interface ReduceOp {
    int apply(MemorySegment res, MemorySegment a, MemorySegment axes, long axesNum, boolean keepdims,
        MemorySegment stream);
  }

  /**
   * Wraps the {@code (res, a, int, int, stream)} native shape shared by {@code mlx_swapaxes}: applies a
   * two-argument-plus-stream operation.
   */
  static MLXArray axis2Op(String opName, MLXArray a, int axis1, int axis2, Axis2Op op) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(opName, () -> op.apply(res, a.handle(), axis1, axis2, DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  @FunctionalInterface
  interface Axis2Op {
    int apply(MemorySegment res, MemorySegment a, int axis1, int axis2, MemorySegment stream);
  }

  /**
   * Runs a status-returning native call and throws {@link MLXException} on failure. Clears {@link NativeLoader}'s
   * thread-local error message immediately before invoking {@code nativeCall}, not just after it fails: some entry
   * points (see {@link MLX#array}) fire the error handler without a status for this method to see. Without the
   * clear-before step, a stale message left behind by one of those would sit there until the next failing checked call,
   * which would then misreport it as its own.
   */
  static void checked(IntSupplier nativeCall) {
    checked(null, nativeCall);
  }

  /**
   * Same as {@link #checked(IntSupplier)}, but names {@code opName} in the failure message on a non-zero status.
   * Package-private, not {@code private}: {@link MLXArray#toFloatArray()} is a cross-class caller with its own op-level
   * name to attribute failures to.
   */
  static void checked(String opName, IntSupplier nativeCall) {
    NativeLoader.clearLastNativeError();
    int status = nativeCall.getAsInt();
    if (status != 0) {
      throw nativeFailure((opName == null ? "" : opName + ": ") + "mlx-c call failed with status " + status);
    }
  }

  static MLXException nativeFailure(String message) {
    String nativeMessage = NativeLoader.lastNativeError();
    NativeLoader.clearLastNativeError();
    return new MLXException(message + (nativeMessage != null ? ": " + nativeMessage : ""));
  }
}
