package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.IntSupplier;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_array_;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.ffi.mlx_vector_array_;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Static facade over the mlx-c ops used by this slice. Every op here only builds the lazy computation graph; nothing
 * runs on the GPU/CPU until {@link #eval} (or the implicit eval inside {@link MLXArray#toFloatArray()}) triggers it.
 * See req/initial-plan.md §7 and req/phase3-plan.md, which describes the broadcast/matmul guards, the additional ops,
 * and the {@code mlx_vector_array}-based {@link #eval} added in this phase.
 *
 * <p>
 * Every op's result is allocated in the same scope as its first {@code MLXArray} operand; {@link #array} takes the
 * scope explicitly since it has no operand to infer one from.
 *
 * <p>
 * {@link #defaultDevice()}/{@link #defaultStream()} are resolved once, lazily, from whatever mlx-c's own default device
 * is at first use, and cached for the process lifetime -- this slice does not expose device switching, so there is no
 * stale-cache hazard in practice.
 */
public final class MLX {

  private MLX() {}

  // Must run before any field below touches mlx_h: jextract binds each
  // downcall's method handle lazily, in a private per-function holder
  // class, the first time that function is actually called -- and that
  // first call fails unless the dylib is already loaded by then.
  // @EnabledIfNativeAvailable covers this for tests by construction; a
  // plain main() has nothing else that would call it first.
  static {
    NativeLoader.ensureLoaded();
  }

  private static final Arena FACADE_ARENA = Arena.ofShared();
  private static final MemorySegment DEFAULT_DEVICE = resolveDefaultDevice();
  private static final MemorySegment DEFAULT_STREAM = resolveDefaultStream();

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

  /** Opaque {@code mlx_device} handle; valid for the process lifetime. */
  public static MemorySegment defaultDevice() {
    return DEFAULT_DEVICE;
  }

  /** Opaque {@code mlx_stream} handle; valid for the process lifetime. */
  public static MemorySegment defaultStream() {
    return DEFAULT_STREAM;
  }

  /** Creates a FLOAT32 array in {@code scope} from row-major {@code data} laid out as {@code shape}. */
  public static MLXArray array(MLXScope scope, float[] data, int[] shape) {
    long expected = 1;
    for (int dim : shape) {
      expected *= dim;
    }
    if (expected != data.length) {
      throw new IllegalArgumentException("shape " + java.util.Arrays.toString(shape) + " (size " + expected
          + ") does not match data length " + data.length);
    }
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeData = tmp.allocateFrom(ValueLayout.JAVA_FLOAT, data);
      MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
      // mlx_array_new_data has no status return: on failure it calls
      // the error handler and hands back a struct with a null ctx
      // (native/scratch/mlx-c/mlx/c/array.cpp:238-251), instead of
      // anything checked() can see. Detect that explicitly rather than
      // letting the failure surface far from its cause, the first time
      // something dereferences the empty handle.
      NativeLoader.clearLastNativeError();
      MemorySegment handle =
          mlx_h.mlx_array_new_data(scope, nativeData, nativeShape, shape.length, mlx_h.MLX_FLOAT32());
      if (mlx_array_.ctx(handle).address() == 0) {
        throw nativeFailure("mlx_array_new_data");
      }
      return new MLXArray(scope, handle);
    }
  }

  /** Elementwise sum of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if they differ. */
  public static MLXArray add(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "add");
    return addUnchecked(a, b);
  }

  /** Elementwise difference of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if they differ. */
  public static MLXArray subtract(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "subtract");
    return binaryOp("subtract", a, b, mlx_h::mlx_subtract);
  }

  /** Elementwise product of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if they differ. */
  public static MLXArray multiply(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "multiply");
    return binaryOp("multiply", a, b, mlx_h::mlx_multiply);
  }

  /** Elementwise quotient of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if they differ. */
  public static MLXArray divide(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "divide");
    return binaryOp("divide", a, b, mlx_h::mlx_divide);
  }

  /**
   * Matrix product of {@code a} and {@code b}. Either may be rank-1 (promoted internally, matching mlx's own
   * vector-matrix / matrix-vector rules); rank &ge; 2 operands batch-broadcast over every axis but the last two. Rank-0
   * operands are rejected, as mlx itself rejects them.
   */
  public static MLXArray matmul(MLXArray a, MLXArray b) {
    requireMatmulCompatible(a, b);
    return binaryOp("matmul", a, b, mlx_h::mlx_matmul);
  }

  /**
   * NumPy broadcast compatibility between {@code a} and {@code b}, right-aligned: for every dimension pair, one of
   * {@code d1 == d2}, {@code d1 == 1} or {@code d2 == 1} must hold (upstream {@code mlx/utils.cpp:136-167},
   * {@code broadcast_shapes}). Deliberately written as {@code d1 == 1}/{@code d2 == 1}, not {@code d1 <= 1}/
   * {@code d2 <= 1}: the latter would wrongly accept a {@code 0} dimension against a {@code 3}, which native rejects.
   */
  private static void requireBroadcastCompatible(MLXArray a, MLXArray b, String op) {
    int[] sa = a.shape();
    int[] sb = b.shape();
    int n = Math.max(sa.length, sb.length);
    for (int i = 0; i < n; i++) {
      int da = i < n - sa.length ? 1 : sa[i - (n - sa.length)];
      int db = i < n - sb.length ? 1 : sb[i - (n - sb.length)];
      if (da != db && da != 1 && db != 1) {
        throw new IllegalArgumentException(
            op + ": incompatible shapes " + java.util.Arrays.toString(sa) + " and " + java.util.Arrays.toString(sb));
      }
    }
  }

  /**
   * Mirrors {@code matmul}'s real native rules (upstream {@code ops.cpp:3192-3267}), in evaluation order: reject rank-0
   * on either operand; promote a rank-1 {@code a} to {@code [1, a.shape(0)]} and a rank-1 {@code b} to
   * {@code [b.shape(0), 1]} (mlx's own {@code expand_dims}); compare the inner dimension on the <em>promoted</em>
   * shapes, not the raw ones -- indexing {@code b.shape(-2)} on a raw rank-1 {@code b} would be out of bounds; require
   * {@link DType#isInexact()} on both operands, mirroring native's {@code issubdtype(out_type, inexact)} rather than
   * narrowing it to {@code == FLOAT32}; then require the batch dimensions (every axis but the last two of the promoted
   * shapes) to be broadcast-compatible.
   */
  private static void requireMatmulCompatible(MLXArray a, MLXArray b) {
    int[] sa = a.shape();
    int[] sb = b.shape();
    if (sa.length == 0 || sb.length == 0) {
      throw new IllegalArgumentException("matmul: rank-0 operand not supported (shapes " + java.util.Arrays.toString(sa)
          + " and " + java.util.Arrays.toString(sb) + ")");
    }
    int[] pa = sa.length == 1 ? new int[] {1, sa[0]} : sa;
    int[] pb = sb.length == 1 ? new int[] {sb[0], 1} : sb;
    if (pa[pa.length - 1] != pb[pb.length - 2]) {
      throw new IllegalArgumentException(
          "matmul: incompatible shapes " + java.util.Arrays.toString(sa) + " and " + java.util.Arrays.toString(sb));
    }
    if (!a.dtype().isInexact() || !b.dtype().isInexact()) {
      throw new IllegalArgumentException("matmul: requires inexact dtypes, got " + a.dtype() + " and " + b.dtype());
    }
    int battA = pa.length - 2;
    int battB = pb.length - 2;
    int batchDims = Math.max(battA, battB);
    for (int i = 0; i < batchDims; i++) {
      int da = i < batchDims - battA ? 1 : pa[i - (batchDims - battA)];
      int db = i < batchDims - battB ? 1 : pb[i - (batchDims - battB)];
      if (da != db && da != 1 && db != 1) {
        throw new IllegalArgumentException("matmul: incompatible batch dimensions " + java.util.Arrays.toString(sa)
            + " and " + java.util.Arrays.toString(sb));
      }
    }
  }

  /**
   * Directional broadcast check for {@code broadcastTo(a, targetShape)}: requires
   * {@code broadcast_shapes(a.shape(), targetShape) == targetShape} (upstream {@code ops.cpp:1601-1613}), which is
   * <em>not</em> the symmetric predicate {@link #requireBroadcastCompatible} uses -- {@code broadcastTo([3], [1])}
   * satisfies the symmetric rule but native rejects it. Also rejects any negative element of {@code targetShape}
   * outright, independent of the shape arithmetic: {@code broadcastTo([1], [-1])} would otherwise pass here, since
   * {@code 1 == 1} makes the directional check succeed too, and the failure would surface as {@link MLXException} from
   * native instead of {@link IllegalArgumentException} from Java.
   */
  private static void requireBroadcastableTo(MLXArray a, int[] targetShape) {
    for (int t : targetShape) {
      if (t < 0) {
        throw new IllegalArgumentException(
            "broadcastTo: target shape " + java.util.Arrays.toString(targetShape) + " has a negative dimension");
      }
    }
    int[] sa = a.shape();
    if (sa.length > targetShape.length) {
      throw new IllegalArgumentException("broadcastTo: cannot broadcast " + java.util.Arrays.toString(sa) + " to "
          + java.util.Arrays.toString(targetShape));
    }
    int diff = targetShape.length - sa.length;
    for (int i = 0; i < sa.length; i++) {
      int ad = sa[i];
      int td = targetShape[i + diff];
      if (ad != td && ad != 1) {
        throw new IllegalArgumentException("broadcastTo: cannot broadcast " + java.util.Arrays.toString(sa) + " to "
            + java.util.Arrays.toString(targetShape));
      }
    }
  }

  /**
   * Skips the Java-side shape check {@link #add} otherwise applies before ever reaching native. Deliberate bypass
   * (req/initial-plan.md, Testing approach) so a test can exercise a genuine native error path and prove it surfaces as
   * {@link MLXException} rather than a process abort.
   */
  static MLXArray addUnchecked(MLXArray a, MLXArray b) {
    return binaryOp("add", a, b, mlx_h::mlx_add);
  }

  private static MLXArray binaryOp(String opName, MLXArray a, MLXArray b, BinaryOp op) {
    MLXScope scope = a.scope();
    // The result is allocated in a's scope; without this check, b's scope
    // (and its thread-confinement guard) is never touched at all -- b.handle()
    // below reads it directly -- so an operand from another (possibly
    // another thread's) scope would silently bypass MLXScope's confinement
    // contract instead of being rejected here.
    if (scope != b.scope()) {
      throw new IllegalArgumentException(opName + ": operands belong to different MLXScopes");
    }
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(() -> op.apply(res, a.handle(), b.handle(), DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  @FunctionalInterface
  private interface BinaryOp {
    int apply(MemorySegment res, MemorySegment a, MemorySegment b, MemorySegment stream);
  }

  private static MLXArray unaryOp(String opName, MLXArray a, UnaryOp op) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(() -> op.apply(res, a.handle(), DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  @FunctionalInterface
  private interface UnaryOp {
    int apply(MemorySegment res, MemorySegment a, MemorySegment stream);
  }

  /**
   * Wraps the {@code (res, a, const int*, size_t, stream)} native shape shared by {@code mlx_broadcast_to},
   * {@code mlx_squeeze_axes} and {@code mlx_transpose_axes}: a confined {@link Arena} owns {@code param}'s native copy
   * for the lifetime of the call, exactly as {@link #reshape} inlines it for {@code mlx_reshape}.
   */
  private static MLXArray shapeOp(String opName, MLXArray a, int[] param, ShapeOp op) {
    MLXScope scope = a.scope();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeParam = tmp.allocateFrom(ValueLayout.JAVA_INT, param);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      checked(() -> op.apply(res, a.handle(), nativeParam, param.length, DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  @FunctionalInterface
  private interface ShapeOp {
    int apply(MemorySegment res, MemorySegment a, MemorySegment param, long paramNum, MemorySegment stream);
  }

  /** Elementwise natural exponential. */
  public static MLXArray exp(MLXArray a) {
    return unaryOp("exp", a, mlx_h::mlx_exp);
  }

  /** Elementwise natural logarithm. */
  public static MLXArray log(MLXArray a) {
    return unaryOp("log", a, mlx_h::mlx_log);
  }

  /** Elementwise sine. */
  public static MLXArray sin(MLXArray a) {
    return unaryOp("sin", a, mlx_h::mlx_sin);
  }

  /** Elementwise cosine. */
  public static MLXArray cos(MLXArray a) {
    return unaryOp("cos", a, mlx_h::mlx_cos);
  }

  /** Sums every element to a rank-0 scalar array. */
  public static MLXArray sum(MLXArray a) {
    // mlx_sum(res, a, keepdims, s) carries an extra bool beyond unaryOp's
    // (res, a, stream) shape, so it does not fit that helper -- kept as its
    // own body rather than forcing it through a shape unaryOp doesn't have.
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(() -> mlx_h.mlx_sum(res, a.handle(), false, DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  /**
   * Generalized inner product of {@code a} and {@code b}: contracts their last axes, matching {@code mlx_inner} (and
   * NumPy's {@code inner}). Requires {@code a.shape()[-1] == b.shape()[-1]} unless either operand is rank-0, in which
   * case native treats the call as a scalar multiply and no last axis exists on that side to compare.
   */
  public static MLXArray inner(MLXArray a, MLXArray b) {
    int[] sa = a.shape();
    int[] sb = b.shape();
    if (sa.length > 0 && sb.length > 0 && sa[sa.length - 1] != sb[sb.length - 1]) {
      throw new IllegalArgumentException("inner: last dimensions disagree: " + java.util.Arrays.toString(sa) + " and "
          + java.util.Arrays.toString(sb));
    }
    return binaryOp("inner", a, b, mlx_h::mlx_inner);
  }

  /**
   * Outer product of {@code a} and {@code b}: every pairwise product of their flattened elements, matching
   * {@code mlx_outer}. Guards {@code a.size() <= Integer.MAX_VALUE} because native reshapes {@code a} via a
   * {@code static_cast<int>(a.size())} with no bounds check of its own (upstream {@code ops.cpp:5448-5451}); {@code b}
   * goes through {@code flatten} instead, which needs no such cast, so the guard is deliberately one-sided.
   */
  public static MLXArray outer(MLXArray a, MLXArray b) {
    if (a.size() > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("outer: a has " + a.size() + " elements, exceeding Integer.MAX_VALUE");
    }
    return binaryOp("outer", a, b, mlx_h::mlx_outer);
  }

  /** Reshapes {@code a} to {@code shape}, which must have the same total element count. */
  public static MLXArray reshape(MLXArray a, int[] shape) {
    long targetSize = 1;
    for (int dim : shape) {
      targetSize *= dim;
    }
    if (targetSize != a.size()) {
      throw new IllegalArgumentException("reshape: shape " + java.util.Arrays.toString(shape) + " (size " + targetSize
          + ") does not match array size " + a.size());
    }
    MLXScope scope = a.scope();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      checked(() -> mlx_h.mlx_reshape(res, a.handle(), nativeShape, shape.length, DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  /** Broadcasts {@code a} to {@code targetShape}, per NumPy's directional broadcasting rules. */
  public static MLXArray broadcastTo(MLXArray a, int[] targetShape) {
    requireBroadcastableTo(a, targetShape);
    return shapeOp("broadcastTo", a, targetShape, mlx_h::mlx_broadcast_to);
  }

  /** Removes every size-1 axis from {@code a}'s shape. */
  public static MLXArray squeeze(MLXArray a) {
    return unaryOp("squeeze", a, mlx_h::mlx_squeeze);
  }

  /** Removes the given {@code axes} from {@code a}'s shape; native requires each to currently have size 1. */
  public static MLXArray squeeze(MLXArray a, int[] axes) {
    return shapeOp("squeeze", a, axes, mlx_h::mlx_squeeze_axes);
  }

  /** Reverses every axis. */
  public static MLXArray transpose(MLXArray a) {
    return unaryOp("transpose", a, mlx_h::mlx_transpose);
  }

  /** Permutes {@code a}'s axes according to {@code axes}, a permutation of {@code 0 .. a.ndim() - 1}. */
  public static MLXArray transpose(MLXArray a, int[] axes) {
    return shapeOp("transpose", a, axes, mlx_h::mlx_transpose_axes);
  }

  /**
   * Slices {@code a} along every axis using {@code start} (inclusive) and {@code stop} (exclusive), with every axis
   * implicitly strided by 1. Equivalent to {@link #slice(MLXArray, int[], int[], int[])} with an all-ones
   * {@code strides}.
   */
  public static MLXArray slice(MLXArray a, int[] start, int[] stop) {
    requireSliceLengths(a, start, stop, null);
    int[] strides = new int[a.ndim()];
    java.util.Arrays.fill(strides, 1);
    return sliceNative(a, start, stop, strides);
  }

  /**
   * Slices {@code a} along every axis using {@code start} (inclusive), {@code stop} (exclusive) and {@code strides}.
   * Mirrors native's own {@code normalize_slice} (upstream {@code ops.cpp:656-696}): negative {@code start}/
   * {@code stop} are normalized NumPy-style ({@code n + i}); a negative stride reverses that axis; {@code stop} is
   * clamped to the axis length rather than bounds-checked. A zero stride is division-by-zero -- C++ undefined behaviour
   * reachable from this call, silently returning an empty axis on Apple Silicon rather than crashing or erroring -- so
   * it is rejected here with a message phrased after NumPy's own ("slice step cannot be zero") rather than as a
   * jmlx-specific restriction.
   */
  public static MLXArray slice(MLXArray a, int[] start, int[] stop, int[] strides) {
    requireSliceLengths(a, start, stop, strides);
    for (int i = 0; i < strides.length; i++) {
      if (strides[i] == 0) {
        throw new IllegalArgumentException("slice: strides[" + i + "] must not be 0 (slice step cannot be zero)");
      }
    }
    return sliceNative(a, start, stop, strides);
  }

  /**
   * Mirrors upstream's own length check ({@code ops.cpp:757-763}, {@code "[slice] Invalid number of indices or
   * strides for array with dimension N."}) but names which of {@code start}/{@code stop}/{@code strides} disagreed.
   * {@code strides} is {@code null} for the 3-arg {@link #slice(MLXArray, int[], int[])} overload, which synthesizes an
   * all-ones {@code strides} of the correct length itself and so has nothing to check there.
   */
  private static void requireSliceLengths(MLXArray a, int[] start, int[] stop, int[] strides) {
    int nd = a.ndim();
    if (start.length != nd || stop.length != nd || (strides != null && strides.length != nd)) {
      throw new IllegalArgumentException("slice: start (length " + start.length + "), stop (length " + stop.length + ")"
          + (strides != null ? ", strides (length " + strides.length + ")" : "") + " must all equal a.ndim() (" + nd
          + ")");
    }
  }

  private static MLXArray sliceNative(MLXArray a, int[] start, int[] stop, int[] strides) {
    MLXScope scope = a.scope();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeStart = tmp.allocateFrom(ValueLayout.JAVA_INT, start);
      MemorySegment nativeStop = tmp.allocateFrom(ValueLayout.JAVA_INT, stop);
      MemorySegment nativeStrides = tmp.allocateFrom(ValueLayout.JAVA_INT, strides);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      checked(() -> mlx_h.mlx_slice(res, a.handle(), nativeStart, start.length, nativeStop, stop.length, nativeStrides,
          strides.length, DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  // Not code evaluation: mirrors mlx-c's mlx_eval, which forces every
  // lazily-built computation graph reachable from `arrays` to actually run
  // on device, in a single scheduling pass rather than one pass per array.
  /**
   * Explicitly triggers computation for the given arrays, via a single {@code mlx_eval} over an
   * {@code mlx_vector_array} rather than one {@code mlx_array_eval} per array. This schedules all N graphs before
   * waiting on any of them, so all N sets of intermediates are live simultaneously -- <b>peak memory can go up</b>
   * relative to evaluating one array at a time, which lets each graph's temporaries be released before the next begins.
   * See req/phase3-plan.md §4 for a measured figure (peak bytes for the loop versus the vector, over the same N-array
   * workload).
   *
   * <p>
   * There is deliberately no same-scope check here (unlike {@link #binaryOp}): {@code eval} allocates no result, so
   * there is no "which scope owns the output" question to settle, and evaluating arrays from two scopes on one thread
   * is legitimate.
   *
   * <p>
   * On failure, re-runs the per-array {@code mlx_array_eval} loop to attribute which array caused it -- the one
   * diagnostic the joint form loses -- and rethrows with that array's index; if the re-run cannot reproduce the
   * failure, the original exception is rethrown unchanged rather than masked.
   */
  public static void eval(MLXArray... arrays) {
    int n = arrays.length;
    if (n == 0) {
      return;
    }
    // Pass 1 CAPTURES; it does not merely touch. handle() is thread-checked
    // (MLXArray.ensureOpen() -> MLXScope.checkAccess()), so this both
    // validates every array up front and yields the segments the copy loop
    // below uses -- the "cannot throw mid-build" invariant then holds by
    // construction rather than by reading each handle a second time.
    MemorySegment[] handles = new MemorySegment[n];
    for (int i = 0; i < n; i++) {
      handles[i] = arrays[i].handle();
    }
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment vec = newVectorArray(handles, tmp);
      try {
        checked(() -> mlx_h.mlx_eval(vec));
      } catch (MLXException e) {
        // Restores the per-array attribution the joint eval loses. MUST sit
        // here, inside the try-with-resources and BEFORE the finally frees
        // vec: it re-evaluates the INDIVIDUAL handles via mlx_array_eval,
        // not the vector, which is exactly what failed.
        throw attributeEvalFailure(e, handles);
      } finally {
        mlx_h.mlx_vector_array_free(vec);
      }
    }
  }

  /**
   * Builds an {@code mlx_vector_array} over {@code handles}, backed by a raw struct-array copy allocated from
   * {@code allocator}. Factored out of {@link #eval} because {@code mlx_async_eval} (upstream {@code transforms.h:31})
   * takes the same {@code mlx_vector_array} and would reuse this verbatim.
   *
   * <p>
   * {@code allocator} MUST be a confined {@link Arena} (never an {@link MLXScope}): the vector's backing struct array
   * is not an {@code mlx_array} this method's caller will ever hand to {@code mlx_array_free}, so allocating it through
   * a scope would corrupt that scope's handle-tracking invariant -- a wrong-type delete, not a caught exception.
   */
  private static MemorySegment newVectorArray(MemorySegment[] handles, Arena allocator) {
    int n = handles.length;
    MemorySegment buf = mlx_array_.allocateArray(n, allocator);
    long elementSize = mlx_array_.sizeof();
    for (int i = 0; i < n; i++) {
      // A raw MemorySegment.copy of sizeof() bytes, not a ctx get/set
      // round-trip: a byte-for-byte struct copy stays correct if mlx_array_
      // ever gains a field.
      MemorySegment.copy(handles[i], 0L, buf, i * elementSize, elementSize);
    }
    // mlx_vector_array_new_data is statusless and returns a null-ctx struct
    // on failure (vector.cpp:41-54), the same hazard class MLX.array already
    // handles above. clearLastNativeError() must run immediately before this
    // call, not just before mlx_eval below, for the reason given in
    // MLX.checked's javadoc: some entry points fire the error handler
    // without a status for checked() to see.
    NativeLoader.clearLastNativeError();
    MemorySegment vec = mlx_h.mlx_vector_array_new_data(allocator, buf, n);
    // size == 0 returns a non-null ctx (vector.cpp:45 heap-allocates an
    // empty std::vector), so this check will not misfire on eval()'s empty
    // varargs case -- but eval() never reaches here for n == 0 regardless,
    // since it returns before capturing handles.
    if (mlx_vector_array_.ctx(vec).address() == 0) {
      throw nativeFailure("mlx_vector_array_new_data");
    }
    return vec;
  }

  /**
   * Re-runs the per-array {@code mlx_array_eval} loop to name the offender after the joint {@code mlx_eval} above
   * failed; rethrows {@code original} unchanged if the re-run cannot reproduce the failure, so this never masks the
   * error it is trying to describe.
   */
  private static MLXException attributeEvalFailure(MLXException original, MemorySegment[] handles) {
    for (int i = 0; i < handles.length; i++) {
      // `h`, not handles[i], inside the lambda: checked() takes an
      // IntSupplier, and the loop counter is mutated each iteration, so
      // capturing it directly is "local variables referenced from a lambda
      // expression must be final or effectively final". `i` in the message
      // below is fine -- string concatenation is not a capture.
      final MemorySegment h = handles[i];
      try {
        checked(() -> mlx_h.mlx_array_eval(h));
      } catch (MLXException perArray) {
        return new MLXException("eval: array[" + i + "] failed", perArray);
      }
    }
    return original;
  }

  /**
   * Runs a status-returning native call and throws {@link MLXException} on failure. Clears {@link NativeLoader}'s
   * thread-local error message immediately before invoking {@code nativeCall}, not just after it fails: some entry
   * points (see {@link #array}) fire the error handler without a status for this method to see. Without the
   * clear-before step, a stale message left behind by one of those would sit there until the next failing checked call,
   * which would then misreport it as its own.
   */
  static void checked(IntSupplier nativeCall) {
    NativeLoader.clearLastNativeError();
    int status = nativeCall.getAsInt();
    if (status != 0) {
      throw nativeFailure("mlx-c call failed with status " + status);
    }
  }

  private static MLXException nativeFailure(String message) {
    String nativeMessage = NativeLoader.lastNativeError();
    NativeLoader.clearLastNativeError();
    return new MLXException(message + (nativeMessage != null ? ": " + nativeMessage : ""));
  }
}
