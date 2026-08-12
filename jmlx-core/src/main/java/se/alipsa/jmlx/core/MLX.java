package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_array_;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.ffi.mlx_vector_array_;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Static facade over the mlx-c ops used by this slice. Every op here only builds the lazy computation graph; nothing
 * runs on the GPU/CPU until {@link #eval} (or the implicit eval inside {@link MLXArray#toFloatArray()}) triggers it.
 * See req/initial-plan.md §7, req/phase3-plan.md and req/phase4-plan.md §1.
 *
 * <p>
 * req/phase4-plan.md M0a split this class: it keeps array creation, {@code eval} and the default device/stream
 * accessors; every other op moved to a sibling in this package, split by kind -- {@link MLXOps}
 * (elementwise/comparisons/reductions/{@code matmul}/{@code inner}/{@code outer}), {@link MLXShape}
 * ({@code reshape}/{@code broadcastTo}/{@code squeeze}/{@code transpose}/{@code slice}), {@link MLXFast} (the
 * {@code fast.h} family: {@code rmsNorm}/{@code layerNorm}/{@code rope}/SDPA), {@link MLXQuant}
 * ({@code quantize}/{@code dequantize}/{@code quantizedMatmul}) and {@link MLXRandom}
 * ({@code seed}/{@code normal}/{@code uniform}). This class does not delegate to them -- duplicating their javadoc here
 * would double the evidence base and let one copy go stale.
 *
 * <p>
 * Every op's result is allocated in the same scope as its first {@code MLXArray} operand; {@link #array} takes the
 * scope explicitly since it has no operand to infer one from.
 *
 * <p>
 * {@link #defaultDevice()}/{@link #defaultStream()} are resolved once, lazily, from whatever mlx-c's own default device
 * is at first use, and cached for the process lifetime -- this slice does not expose device switching, so there is no
 * stale-cache hazard in practice. The resolved values live in {@link NativeOps}, which every op class (including this
 * one, for {@code eval}) needs direct access to; these methods are the public read of that shared state, not a
 * delegated op body.
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

  /** Opaque {@code mlx_device} handle; valid for the process lifetime. */
  public static MemorySegment defaultDevice() {
    return NativeOps.defaultDevice();
  }

  /** Opaque {@code mlx_stream} handle; valid for the process lifetime. */
  public static MemorySegment defaultStream() {
    return NativeOps.DEFAULT_STREAM;
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
        throw NativeOps.nativeFailure("mlx_array_new_data");
      }
      return new MLXArray(scope, handle);
    }
  }

  // Not code evaluation: mirrors mlx-c's mlx_eval, which forces every
  // lazily-built computation graph reachable from `arrays` to actually run
  // on device, in a single scheduling pass rather than one pass per array.
  /**
   * Explicitly triggers computation for the given arrays, via a single {@code mlx_eval} over an
   * {@code mlx_vector_array} rather than one {@code mlx_array_eval} per array. This schedules all N graphs before
   * waiting on any of them, so all N sets of intermediates are live simultaneously, in principle at the cost of peak
   * memory relative to evaluating one array at a time (which would let each graph's temporaries be released before the
   * next begins). In practice, per req/phase3-plan.md §4, that effect was measured as unobservable through this
   * facade's ownership model: every op result -- intermediate or final -- is registered with its {@link MLXScope} the
   * instant the op is invoked and is only freed at scope close, regardless of eval strategy, so nothing this facade
   * builds is ever eligible for differential early free. Two rounds of measurement (single-op and multi-op lazy chains,
   * 6 trials total) found a delta of exactly 0 bytes. The underlying native-level tradeoff remains real in principle
   * for mlx-c graphs whose intermediates are <em>not</em> independently referenced by anything outside the graph, but
   * that case wasn't -- and structurally couldn't be -- exercised by this facade's measurement.
   *
   * <p>
   * There is deliberately no same-scope check here (unlike {@code binaryOp}): {@code eval} allocates no result, so
   * there is no "which scope owns the output" question to settle, and evaluating arrays from two scopes on one thread
   * is legitimate.
   *
   * <p>
   * On failure, re-runs the per-array {@code mlx_array_eval} loop to identify the first array whose own evaluation
   * reproduces the failure -- the one diagnostic the joint form loses -- and rethrows with that array's index; if the
   * re-run cannot reproduce the failure, the original exception is rethrown unchanged rather than masked.
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
        NativeOps.checked("eval", () -> mlx_h.mlx_eval(vec));
      } catch (MLXException e) {
        // Restores the per-array attribution the joint eval loses, by
        // re-evaluating the INDIVIDUAL handles via mlx_array_eval, not the
        // vector (which is exactly what failed). Its position relative to
        // the finally below is immaterial: it only touches `handles`, which
        // are segments owned by each array's own MLXScope, not `vec` or
        // `tmp` -- it would be equally correct after either was released.
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
    // NativeOps.checked's javadoc: some entry points fire the error handler
    // without a status for checked() to see.
    NativeLoader.clearLastNativeError();
    MemorySegment vec = mlx_h.mlx_vector_array_new_data(allocator, buf, n);
    // size == 0 returns a non-null ctx (vector.cpp:45 heap-allocates an
    // empty std::vector), so this check will not misfire on eval()'s empty
    // varargs case -- but eval() never reaches here for n == 0 regardless,
    // since it returns before capturing handles.
    if (mlx_vector_array_.ctx(vec).address() == 0) {
      throw NativeOps.nativeFailure("mlx_vector_array_new_data");
    }
    return vec;
  }

  /**
   * Re-runs the per-array {@code mlx_array_eval} loop to name the offender after the joint {@code mlx_eval} above
   * failed; rethrows {@code original} unchanged if the re-run cannot reproduce the failure, so this never masks the
   * error it is trying to describe. When the re-run does reproduce it, the returned exception's message folds in the
   * per-array native text (not just its index) so it is visible from {@code getMessage()} without navigating
   * {@code getCause()}, and {@code original} -- the joint failure that triggered this re-run -- is attached as a
   * suppressed exception rather than discarded.
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
        NativeOps.checked(() -> mlx_h.mlx_array_eval(h));
      } catch (MLXException perArray) {
        MLXException attributed = new MLXException("eval: array[" + i + "] failed: " + perArray.getMessage(), perArray);
        attributed.addSuppressed(original);
        return attributed;
      }
    }
    return original;
  }
}
