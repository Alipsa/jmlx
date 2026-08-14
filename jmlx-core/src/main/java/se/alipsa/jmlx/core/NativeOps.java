package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.util.function.IntSupplier;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_array_;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Package-private plumbing shared by every op class in {@code se.alipsa.jmlx.core}: the resolved
 * default device/stream, the status-checking helpers, and the {@code (res, operand..., stream)}
 * op-body shapes ({@link #binaryOp}, {@link #unaryOp}, {@link #shapeOp}) that {@link MLXOps} and
 * {@link MLXShape} build every op on. See req/phase4-plan.md §1 -- this class is the split's
 * foundation layer: every other class in this package may depend on it, but it depends on nothing
 * else here.
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

  /**
   * Opaque {@code mlx_device} handle; valid for the process lifetime. Backs {@link
   * MLX#defaultDevice()}.
   */
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
   * Innermost scope among all non-null {@code operands} (req/phase4-plan.md §2). Touches every
   * non-null operand's {@link MLXArray#scope()}, so each one's {@code checkAccess()} runs -- this
   * is what stops a multi-operand op from silently bypassing {@link MLXScope}'s confinement
   * contract for every operand but the one its target happens to be picked from.
   *
   * <p>Precondition: at least one operand is non-null -- the caller always passes the primary input
   * ({@code x}, {@code q}, {@code w}), which is never nullable in any mlx-c signature. Violating it
   * throws {@link IllegalArgumentException} naming {@code op}, not an empty-reduce exception: the
   * reachable case is a caller (e.g. inside a {@code layerNorm} body) that forgot to pass its
   * primary input, and the message should say that rather than "empty".
   *
   * @throws IllegalArgumentException if any two operand scopes are unrelated (siblings, or two
   *     independent roots)
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

  /**
   * Allocates the result into {@code target} instead of {@code a.scope()} -- e.g. {@code
   * Linear.forward} computing {@code transpose(W, x.scope())} so the view lands in the step scope
   * rather than leaking into the model scope (req/phase4-plan.md §5). {@code target} must be
   * related to {@code a.scope()} -- either an ancestor OR a descendant of it -- checked via {@link
   * MLXScope#innermost}, not {@link MLXScope#isAncestorOf} alone: unlike {@link MLX#hoist}, which
   * only ever narrows toward an ancestor, this overload is also used to push a result INTO a
   * descendant (the step scope), so both directions must be legal. Without this check, a caller
   * passing an unrelated {@code target} would silently allocate a result referencing {@code a} into
   * a scope that could close before (or long after) {@code a}'s own scope, breaking the invariant
   * that a result's scope is always related to every operand it references.
   */
  static MLXArray unaryOp(String opName, MLXArray a, MLXScope target, UnaryOp op) {
    MLXScope.innermost(a.scope(), target);
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
   * Wraps the {@code (res, a, const int*, size_t, stream)} native shape shared by {@code
   * mlx_broadcast_to}, {@code mlx_squeeze_axes} and {@code mlx_transpose_axes}: a confined {@link
   * Arena} owns {@code param}'s native copy for the lifetime of the call, exactly as {@code
   * reshape} inlines it for {@code mlx_reshape}.
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
    int apply(
        MemorySegment res,
        MemorySegment a,
        MemorySegment param,
        long paramNum,
        MemorySegment stream);
  }

  /**
   * Wraps the {@code (res, a, const int*, size_t, bool, stream)} native shape shared by {@code
   * mlx_sum_axes} and {@code mlx_mean_axes}: a confined {@link Arena} owns {@code axes}'s native
   * copy for the lifetime of the call, exactly as {@code shapeOp} does for its param.
   */
  static MLXArray reduceOp(String opName, MLXArray a, int[] axes, boolean keepdims, ReduceOp op) {
    MLXScope scope = a.scope();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeAxes = tmp.allocateFrom(ValueLayout.JAVA_INT, axes);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      checked(
          opName,
          () -> op.apply(res, a.handle(), nativeAxes, axes.length, keepdims, DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  @FunctionalInterface
  interface ReduceOp {
    int apply(
        MemorySegment res,
        MemorySegment a,
        MemorySegment axes,
        long axesNum,
        boolean keepdims,
        MemorySegment stream);
  }

  /**
   * Wraps the {@code (res, a, int, int, stream)} native shape shared by {@code mlx_swapaxes}:
   * applies a two-argument-plus-stream operation.
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
   * The native "null" for a by-value nullable {@code mlx_array} parameter (e.g. {@code
   * mlx_fast_rms_norm}'s {@code weight}, {@code mlx_random_normal}'s {@code key}): a zero-{@code
   * ctx} struct, never {@link MemorySegment#NULL} -- passing {@code MemorySegment.NULL} where mlx-c
   * expects a by-value struct is a segfault, not an exception (req/phase4-plan.md, Research
   * findings). {@link SegmentAllocator#allocate}'s contract makes no zero-fill guarantee (only
   * {@link Arena#ofConfined()}/{@code ofShared()} document zero-initialization, and {@code
   * SegmentAllocator.slicingAllocator} explicitly hands back reused, non-zeroed slices), so the
   * struct is filled with zero explicitly rather than relying on the allocator.
   */
  static MemorySegment nullableHandle(MLXArray a, SegmentAllocator tmp) {
    if (a != null) {
      return a.handle();
    }
    return tmp.allocate(mlx_array_.layout()).fill((byte) 0);
  }

  /**
   * Copies {@code handles} into a freshly allocated contiguous {@code mlx_array[]} buffer -- the
   * raw struct-array shape both {@code mlx_vector_array_new_data} ({@link MLX#eval}'s vector
   * construction) and {@code mlx_vector_array_set_data} ({@link MLXGrad}'s upcall, writing an
   * existing vector's ctx) accept as their {@code data} parameter. A byte-for-byte struct copy, not
   * a ctx get/set round-trip, so this stays correct if {@code mlx_array_} ever gains a field.
   * {@code allocator} must be a confined {@link Arena} (never an {@link MLXScope}) for the same
   * reason {@link MLX#eval}'s vector allocator must be: the buffer is not an {@code mlx_array}
   * handle this method's caller will ever pass to {@code mlx_array_free}, so allocating it through
   * a scope would corrupt that scope's handle-tracking invariant. Typed as {@link Arena}, not the
   * wider {@link SegmentAllocator} both {@code mlx_array_.allocateArray} and {@link
   * MemorySegment#copy} would otherwise accept -- {@code MLXScope} implements {@code
   * SegmentAllocator} too, and widening this parameter would let a caller pass one in and compile,
   * silently reintroducing the corruption this javadoc warns against.
   */
  static MemorySegment copyHandlesInto(MemorySegment[] handles, Arena allocator) {
    int n = handles.length;
    MemorySegment buf = mlx_array_.allocateArray(n, allocator);
    long elementSize = mlx_array_.sizeof();
    for (int i = 0; i < n; i++) {
      MemorySegment.copy(handles[i], 0L, buf, i * elementSize, elementSize);
    }
    return buf;
  }

  /**
   * Runs a status-returning native call and throws {@link MLXException} on failure. Clears {@link
   * NativeLoader}'s thread-local error message immediately before invoking {@code nativeCall}, not
   * just after it fails: some entry points (see {@link MLX#array}) fire the error handler without a
   * status for this method to see. Without the clear-before step, a stale message left behind by
   * one of those would sit there until the next failing checked call, which would then misreport it
   * as its own.
   */
  static void checked(IntSupplier nativeCall) {
    checked(null, nativeCall);
  }

  /**
   * Same as {@link #checked(IntSupplier)}, but names {@code opName} in the failure message on a
   * non-zero status. Package-private, not {@code private}: {@link MLXArray#toFloatArray()} is a
   * cross-class caller with its own op-level name to attribute failures to.
   */
  static void checked(String opName, IntSupplier nativeCall) {
    NativeLoader.clearLastNativeError();
    int status = nativeCall.getAsInt();
    if (status != 0) {
      throw nativeFailure(
          (opName == null ? "" : opName + ": ") + "mlx-c call failed with status " + status);
    }
  }

  static MLXException nativeFailure(String message) {
    String nativeMessage = NativeLoader.lastNativeError();
    NativeLoader.clearLastNativeError();
    return new MLXException(message + (nativeMessage != null ? ": " + nativeMessage : ""));
  }
}
