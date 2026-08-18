package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.IntSupplier;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_array_;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.ffi.mlx_optional_dtype_;
import se.alipsa.jmlx.ffi.mlx_optional_float_;
import se.alipsa.jmlx.ffi.mlx_optional_int_;
import se.alipsa.jmlx.ffi.mlx_stream_;
import se.alipsa.jmlx.ffi.mlx_vector_array_;
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

  /**
   * The process-wide default CPU stream, resolved once and never freed, exactly like {@link
   * #DEFAULT_STREAM}. {@code mlx_default_cpu_stream_new} returns MLX's *existing* default CPU
   * stream (a stable scheduler index across every call) rather than minting a new one -- {@code
   * mlx_stream_new_device} would instead register a fresh scheduler stream on every call, leaking
   * one per invocation since nothing would ever free it. {@link MLXIO#loadSafetensors}/ {@link
   * MLXIO#loadGguf} need this instead of {@link #DEFAULT_STREAM} specifically because both loaders'
   * arrays are backed by a lazy {@code Load} primitive whose {@code eval_gpu} is unimplemented in
   * the pinned {@code mlx-metal==0.31.2} wheel (req/plans/phase5-m1-plan.md's amendment).
   */
  static final MemorySegment DEFAULT_CPU_STREAM = resolveDefaultCpuStream();

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
   * {@code mlx_default_cpu_stream_new} has no status return: on failure it fires the error handler
   * and hands back {@code mlx_stream_new_()} (a null-{@code ctx} struct), the same statusless shape
   * {@link MLX#array(MLXScope, float[], int[])} already guards for {@code mlx_array_new_data}
   * (confirmed against {@code stream.cpp}'s {@code catch} branch) -- {@code checked()} would never
   * see the failure, so it is detected explicitly instead, exactly like that site.
   */
  private static MemorySegment resolveDefaultCpuStream() {
    NativeLoader.clearLastNativeError();
    MemorySegment stream = mlx_h.mlx_default_cpu_stream_new(FACADE_ARENA);
    if (mlx_stream_.ctx(stream).address() == 0) {
      throw nativeFailure("mlx_default_cpu_stream_new");
    }
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
   * Wraps the {@code (res, a, int, stream)} native shape shared by {@code mlx_expand_dims}, {@code
   * mlx_tril} and {@code mlx_triu} (req/phase4-plan.md §7's Native surface table). Unlike {@code
   * axis2Op}, this is a single caller-supplied int -- an axis for {@code expandDims}, a diagonal
   * offset {@code k} for {@code tril}/{@code triu} -- not necessarily an axis in every case, so the
   * parameter is named generically in the functional interface below.
   */
  static MLXArray axisOp(String opName, MLXArray a, int param, AxisOp op) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(opName, () -> op.apply(res, a.handle(), param, DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  @FunctionalInterface
  interface AxisOp {
    int apply(MemorySegment res, MemorySegment a, int param, MemorySegment stream);
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
   * The native encoding of an {@code mlx_optional_float} by-value struct (req/phase4-plan.md §7's
   * Native surface table: 8 bytes, {@code {float value; bool has_value;}}, no native constructor
   * function -- allocate and set both fields, the same shape spec §6 describes for the sibling
   * {@code mlx_optional_int}). {@code value == null} encodes "absent" ({@code has_value=false});
   * the {@code value} field is left at {@code 0f} in that case, which mlx-c never reads (see {@code
   * mlx_optional_float_.value}'s getter contract -- it is caller-supplied storage, not itself
   * consulted for validity).
   */
  static MemorySegment optFloat(Arena tmp, Float value) {
    MemorySegment seg = mlx_optional_float_.allocate(tmp);
    mlx_optional_float_.value(seg, value != null ? value : 0f);
    mlx_optional_float_.has_value(seg, value != null);
    return seg;
  }

  /**
   * The native encoding of an {@code mlx_optional_int} by-value struct (req/phase4-plan.md §8's
   * Native surface table: 8 bytes, {@code {int value; bool has_value;}}, no native constructor
   * function -- same shape as {@link #optFloat}, for {@code mlx_quantize}/{@code mlx_dequantize}/
   * {@code mlx_quantized_matmul}'s {@code group_size}/{@code bits} parameters). {@code value ==
   * null} encodes "absent" ({@code has_value=false}); native applies its own default in that case
   * (confirmed empirically, req/plans/phase4-m4-plan.md's Findings section) -- this facade does not
   * duplicate it.
   */
  static MemorySegment optInt(Arena tmp, Integer value) {
    MemorySegment seg = mlx_optional_int_.allocate(tmp);
    mlx_optional_int_.value(seg, value != null ? value : 0);
    mlx_optional_int_.has_value(seg, value != null);
    return seg;
  }

  /**
   * The native encoding of an {@code mlx_optional_dtype} by-value struct -- same 8-byte {@code {int
   * value; bool has_value;}} shape as {@link #optInt}/{@link #optFloat} (confirmed by reading the
   * generated {@code mlx_optional_dtype_} binding directly: byte-identical layout, only the field's
   * C type name differs). {@code value == null} encodes "absent"; {@code mlx_dequantize} then
   * defaults to {@code scales}' own dtype (confirmed empirically, req/plans/phase4-m4-plan.md's
   * Findings section) -- FLOAT32 whenever {@code scales} is FLOAT32, but not unconditionally:
   * {@code scales} need not be FLOAT32 (e.g. {@link MLXQuant#quantize} on a weight already {@code
   * astype}'d to FLOAT16 produces FLOAT16 {@code scales}, confirmed empirically), so a caller
   * relying on an absent {@code dtype} to mean FLOAT32 must first confirm {@code scales}' own dtype
   * is FLOAT32.
   */
  static MemorySegment optDtype(Arena tmp, DType value) {
    MemorySegment seg = mlx_optional_dtype_.allocate(tmp);
    mlx_optional_dtype_.value(seg, value != null ? value.nativeValue() : 0);
    mlx_optional_dtype_.has_value(seg, value != null);
    return seg;
  }

  private static final Map<String, MemorySegment> CSTR_CACHE = new ConcurrentHashMap<>();

  /**
   * Returns a NUL-terminated C string for {@code s}, allocated in {@link #FACADE_ARENA} at most
   * once per distinct literal -- for the closed-set {@code const char*} parameters this facade's
   * mlx-c surface uses (SDPA's {@code mask_mode}), which never need per-call allocation
   * (req/phase4-plan.md §7's Native surface table). Callers store the result in a {@code private
   * static final MemorySegment} field per distinct literal, exactly like {@link #DEFAULT_STREAM}.
   * Backed by an intern cache keyed on {@code s} so that even an accidental per-call use never
   * grows {@link #FACADE_ARENA} beyond one segment per distinct literal ever seen -- the intended
   * static-initializer-only discipline is structural, not just documented.
   */
  static MemorySegment cstr(String s) {
    return CSTR_CACHE.computeIfAbsent(s, FACADE_ARENA::allocateFrom);
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
   * Wraps an op taking an {@code mlx_vector_array} <em>input</em> by value (e.g. {@code
   * mlx_concatenate_axis}) plus a caller-supplied {@code axis}. Resolves the result's scope via
   * {@link #scopeOf} over every element of {@code xs} -- the ancestor rule generalized to N
   * operands (req/phase4-plan.md §3). Builds the {@code mlx_vector_array} itself via {@link
   * #copyHandlesInto} rather than delegating to {@code MLX.newVectorArray}: this class depends on
   * nothing else in this package (see the class javadoc), and calling into {@code MLX} here would
   * be the one exception. {@code vec} must be freed after the call: its backing {@code
   * std::vector<array>} holds a value copy of every operand, each a live refcount, so an unfreed
   * {@code vec} permanently leaks one refcount per operand, per call.
   */
  static MLXArray vectorInOp(String opName, MLXArray[] xs, int axis, VectorInOp op) {
    MLXScope scope = scopeOf(opName, xs);
    MemorySegment[] handles = new MemorySegment[xs.length];
    for (int i = 0; i < xs.length; i++) {
      handles[i] = xs[i].handle();
    }
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment buf = copyHandlesInto(handles, tmp);
      // mlx_vector_array_new_data is statusless -- same null-ctx-on-failure hazard MLX.array and
      // MLX.newVectorArray already guard against explicitly.
      NativeLoader.clearLastNativeError();
      MemorySegment vec = mlx_h.mlx_vector_array_new_data(tmp, buf, handles.length);
      if (mlx_vector_array_.ctx(vec).address() == 0) {
        throw nativeFailure("mlx_vector_array_new_data");
      }
      try {
        MemorySegment res = mlx_h.mlx_array_new(scope);
        checked(opName, () -> op.apply(res, vec, axis, DEFAULT_STREAM));
        return new MLXArray(scope, res);
      } finally {
        mlx_h.mlx_vector_array_free(vec);
      }
    }
  }

  @FunctionalInterface
  interface VectorInOp {
    int apply(MemorySegment res, MemorySegment arrays, int axis, MemorySegment stream);
  }

  /**
   * Wraps an op producing an {@code mlx_vector_array} <em>output</em> (e.g. {@code mlx_split}).
   * {@code target} is an explicit parameter, not inferred from an operand, purely for legibility
   * (req/phase4-plan.md §3: "its inputs may be none" was a false justification in an earlier draft
   * of that document -- {@code mlx_split}'s {@code a} operand does exist -- the real reason is that
   * this helper's whole hazard is two allocators with opposite correct answers on adjacent lines,
   * and naming the target in the signature keeps that contrast visible at the call site).
   *
   * <p>{@code op} is invoked with only {@code (vec, stream)}; any other native parameters (the
   * source array, {@code num_splits}, an axis, ...) are captured by the caller's lambda, the same
   * pattern {@link #checked(String, IntSupplier)} already uses throughout this class.
   */
  static MLXArray[] vectorOutOp(String opName, MLXScope target, VectorOutOp op) {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment vec = mlx_h.mlx_vector_array_new(tmp); // tmp -- NOT target
      try {
        checked(opName, () -> op.apply(vec, DEFAULT_STREAM));
        long n = mlx_h.mlx_vector_array_size(vec);
        MLXArray[] out = new MLXArray[(int) n];
        for (int i = 0; i < n; i++) {
          MemorySegment h = mlx_h.mlx_array_new(target); // target -- NOT tmp
          final long idx = i;
          checked(opName, () -> mlx_h.mlx_vector_array_get(h, vec, idx));
          out[i] = new MLXArray(target, h);
        }
        return out;
      } finally {
        mlx_h.mlx_vector_array_free(vec);
      }
    }
  }

  @FunctionalInterface
  interface VectorOutOp {
    int apply(MemorySegment vec, MemorySegment stream);
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

  /**
   * Reads a borrowed, NUL-terminated {@code const char*}/{@code char*} into a Java {@code String}.
   * {@code ptr} must outlive this call but is never freed by it -- every accessor {@link MLXIO}
   * uses this for ({@code mlx_string_data}, {@code mlx_vector_string_get}, the two map iterator/get
   * families) returns a pointer into storage some other handle already owns (req/phase5-plan.md's
   * Research findings). {@code reinterpret} is required before {@code getString}: a raw pointer
   * value read out of a struct field or written into an out-param slot comes back as a zero-length
   * segment with no declared bounds, and {@code getString} on one throws {@code
   * IndexOutOfBoundsException} rather than reading anything.
   */
  static String readNativeString(MemorySegment ptr) {
    return ptr.reinterpret(Long.MAX_VALUE).getString(0);
  }

  /**
   * Classifies one of the two {@code mlx_map_string_to_*_iterator_next} calls' three-way status
   * (confirmed against {@code map.cpp} directly, req/plans/phase5-m1-plan.md's Findings): {@code 0}
   * means the out-params were written and there is a current entry, {@code 2} means the iterator
   * had already reached the map's end (ordinary loop termination, not failure), and anything else
   * is a genuine error routed through the same {@link MLXException} path {@link #checked} uses.
   */
  static boolean mapIteratorNext(String opName, IntSupplier nativeCall) {
    NativeLoader.clearLastNativeError();
    int status = nativeCall.getAsInt();
    if (status == 0) {
      return true;
    }
    if (status == 2) {
      return false;
    }
    throw nativeFailure(opName + ": mlx-c call failed with status " + status);
  }

  /**
   * Runs one of GGUF's {@code mlx_io_gguf_has_metadata_*} predicates, which return {@code 2} not
   * only when the key is present under a different bucket but also whenever the key is absent from
   * the metadata map entirely (confirmed against {@code io_types.cpp}'s {@code
   * IMPLEMENT_GGUF_HAS_METADATA} macro). {@link MLXIO}'s tensor-reading path never reaches this
   * method at all -- {@code readGgufTensors} calls {@code get_array} unconditionally, with no
   * probing in front of it (req/plans/phase5-m1-plan.md's amendment). Status {@code 2} here means a
   * caller-requested metadata key ({@code loadGguf}'s {@code metadata*Keys} parameters) is
   * genuinely absent from the file -- the ordinary case {@code
   * loadGgufRequestedMetadataKeyAbsentIsOmittedNotThrown} exercises, not a failure. Unlike {@link
   * #mapIteratorNext}, status {@code 2} here still leaves the caller's {@code flag} out-param
   * correctly set (to {@code false}); {@link #checked} would misreport this ordinary case as a bare
   * failure instead.
   */
  static void hasMetadataProbe(String opName, IntSupplier nativeCall) {
    NativeLoader.clearLastNativeError();
    int status = nativeCall.getAsInt();
    if (status == 0 || status == 2) {
      return;
    }
    throw nativeFailure(opName + ": mlx-c call failed with status " + status);
  }
}
