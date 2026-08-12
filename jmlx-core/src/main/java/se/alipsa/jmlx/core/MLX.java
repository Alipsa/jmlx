package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.IntSupplier;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_array_;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Static facade over the mlx-c ops used by this slice. Every op here only builds the lazy computation graph; nothing
 * runs on the GPU/CPU until {@link #eval} (or the implicit eval inside {@link MLXArray#toFloatArray()}) triggers it.
 * See req/initial-plan.md §7.
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

  /** Elementwise natural exponential. */
  public static MLXArray exp(MLXArray a) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(() -> mlx_h.mlx_exp(res, a.handle(), DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  /** Sums every element to a rank-0 scalar array. */
  public static MLXArray sum(MLXArray a) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(() -> mlx_h.mlx_sum(res, a.handle(), false, DEFAULT_STREAM));
    return new MLXArray(scope, res);
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

  /** Reverses every axis. */
  public static MLXArray transpose(MLXArray a) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(() -> mlx_h.mlx_transpose(res, a.handle(), DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  // Not code evaluation: mirrors mlx-c's mlx_array_eval, which forces a
  // lazily-built computation graph to actually run on device.
  /** Explicitly triggers computation for the given arrays. */
  public static void eval(MLXArray... arrays) {
    for (MLXArray a : arrays) {
      checked(() -> mlx_h.mlx_array_eval(a.handle()));
    }
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
