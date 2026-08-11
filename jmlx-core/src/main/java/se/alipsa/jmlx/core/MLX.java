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

  /** Elementwise sum of two same-shaped arrays. */
  public static MLXArray add(MLXArray a, MLXArray b) {
    requireSameShape(a, b, "add");
    return addUnchecked(a, b);
  }

  /** Elementwise difference of two same-shaped arrays. */
  public static MLXArray subtract(MLXArray a, MLXArray b) {
    requireSameShape(a, b, "subtract");
    return binaryOp(a, b, mlx_h::mlx_subtract);
  }

  /** Elementwise product of two same-shaped arrays. */
  public static MLXArray multiply(MLXArray a, MLXArray b) {
    requireSameShape(a, b, "multiply");
    return binaryOp(a, b, mlx_h::mlx_multiply);
  }

  /** Elementwise quotient of two same-shaped arrays. */
  public static MLXArray divide(MLXArray a, MLXArray b) {
    requireSameShape(a, b, "divide");
    return binaryOp(a, b, mlx_h::mlx_divide);
  }

  /** Matrix product of two rank-2 arrays with compatible shapes. */
  public static MLXArray matmul(MLXArray a, MLXArray b) {
    int[] sa = a.shape();
    int[] sb = b.shape();
    if (sa.length != 2 || sb.length != 2 || sa[1] != sb[0]) {
      throw new IllegalArgumentException(
          "matmul: incompatible shapes " + java.util.Arrays.toString(sa) + " and " + java.util.Arrays.toString(sb));
    }
    return binaryOp(a, b, mlx_h::mlx_matmul);
  }

  private static void requireSameShape(MLXArray a, MLXArray b, String op) {
    int[] sa = a.shape();
    int[] sb = b.shape();
    if (!java.util.Arrays.equals(sa, sb)) {
      throw new IllegalArgumentException(
          op + ": shape mismatch " + java.util.Arrays.toString(sa) + " vs " + java.util.Arrays.toString(sb));
    }
  }

  /**
   * Skips the Java-side shape check {@link #add} otherwise applies before ever reaching native. Deliberate bypass
   * (req/initial-plan.md, Testing approach) so a test can exercise a genuine native error path and prove it surfaces as
   * {@link MLXException} rather than a process abort.
   */
  static MLXArray addUnchecked(MLXArray a, MLXArray b) {
    return binaryOp(a, b, mlx_h::mlx_add);
  }

  private static MLXArray binaryOp(MLXArray a, MLXArray b, BinaryOp op) {
    MLXScope scope = a.scope();
    // The result is allocated in a's scope; without this check, b's scope
    // (and its thread-confinement guard) is never touched at all -- b.handle()
    // below reads it directly -- so an operand from another (possibly
    // another thread's) scope would silently bypass MLXScope's confinement
    // contract instead of being rejected here.
    if (scope != b.scope()) {
      throw new IllegalArgumentException("operands belong to different MLXScopes");
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

  /** Reverses every axis; there is no partial-permutation overload in this slice. */
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
