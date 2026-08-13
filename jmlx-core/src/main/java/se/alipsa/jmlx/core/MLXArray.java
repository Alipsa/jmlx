package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * A native {@code mlx_array} handle owned by an {@link MLXScope}. See req/initial-plan.md §6. Not thread-safe: confined
 * to the scope's owning thread, same as the scope itself. That confinement is enforced, not just documented: every
 * handle read goes through {@link #ensureOpen()}, which delegates to {@link MLXScope#checkAccess()} and throws
 * {@link IllegalStateException} on foreign-thread or closed-scope access.
 */
public final class MLXArray implements AutoCloseable {

  private final MLXScope scope;
  private final MemorySegment handle;
  private volatile boolean closed = false;

  MLXArray(MLXScope scope, MemorySegment handle) {
    this.scope = scope;
    this.handle = handle;
  }

  /**
   * This array's owning scope. Public (req/phase4-plan.md §2) so creation ops with no operand to infer a scope from
   * (e.g. an RoPE positions array, a causal mask) and {@code nn}-package layers can target a specific scope explicitly,
   * the same way {@code MLX.array} already takes one. {@code forward()} methods must never allocate here directly (they
   * should target the activation's own scope, or an explicit-target overload for weight-derived views) -- nothing
   * enforces that beyond this javadoc; see req/phase4-plan.md §2 and §5 for why, and {@code MLXMemoryLeakTest} for the
   * only test that would notice a violation.
   */
  public MLXScope scope() {
    ensureOpen();
    return scope;
  }

  MemorySegment handle() {
    ensureOpen();
    return handle;
  }

  /** Returns this array's shape, one element per dimension. */
  public int[] shape() {
    ensureOpen();
    int nd = ndim();
    MemorySegment shapePtr = mlx_h.mlx_array_shape(handle).reinterpret((long) nd * ValueLayout.JAVA_INT.byteSize());
    int[] result = new int[nd];
    MemorySegment.copy(shapePtr, ValueLayout.JAVA_INT, 0, result, 0, nd);
    return result;
  }

  /** Returns this array's element type. */
  public DType dtype() {
    ensureOpen();
    return DType.fromNative(mlx_h.mlx_array_dtype(handle));
  }

  /** Returns the number of dimensions in this array's shape. */
  public int ndim() {
    ensureOpen();
    return (int) mlx_h.mlx_array_ndim(handle);
  }

  /** Returns the total number of elements across all dimensions. */
  public long size() {
    ensureOpen();
    return mlx_h.mlx_array_size(handle);
  }

  /**
   * Evaluates this array (if not already evaluated) and reads it back as a flat, row-major {@code float[]}.
   *
   * <p>
   * Requires an inexact dtype (req/phase4-plan.md §4): {@code FLOAT32} reads back directly; {@code FLOAT16}/
   * {@code BFLOAT16} go through an {@code mlx_astype(FLOAT32)} step first, since there is no {@code
   * mlx_array_data_float16}/{@code _bfloat16} to read those natively. Exact dtypes ({@code INT32}, {@code UINT32},
   * {@code BOOL}) are rejected rather than silently cast -- astype-ing an {@code INT32} array to float would hide the
   * bug this check exists to surface. Use {@link #toIntArray()} for {@code INT32}.
   *
   * <p>
   * Forces a contiguous row-major copy (mlx-c's {@code mlx_contiguous}, {@code allow_col_major=false}) before reading
   * the data pointer. Lazy ops like transpose can yield a strided view; reading raw data without this would return
   * plausible-looking values in the wrong element order -- a silent-wrong-answer bug, not a crash.
   */
  public float[] toFloatArray() {
    ensureOpen();
    DType dtype = dtype();
    if (!dtype.isInexact()) {
      throw new IllegalStateException(
          "toFloatArray() requires an inexact dtype (FLOAT32/FLOAT16/BFLOAT16), got " + dtype);
    }
    try (Arena tmp = Arena.ofConfined()) {
      // tmp only owns the 8-byte mlx_array structs below; the ctx heap
      // allocations mlx_astype/mlx_contiguous fill in are owned by mlx-c
      // and freed solely by mlx_array_free. Two independent, inner-first
      // finally blocks -- not one shared cleanup -- because the astype
      // step is conditional (skipped entirely for FLOAT32) while the
      // contiguous step always runs, and each handle must be freed exactly
      // once regardless of which step throws.
      MemorySegment astyped = null;
      try {
        MemorySegment source = handle;
        if (dtype != DType.FLOAT32) {
          MemorySegment target = mlx_h.mlx_array_new(tmp);
          NativeOps.checked("toFloatArray",
              () -> mlx_h.mlx_astype(target, handle, DType.FLOAT32.nativeValue(), MLX.defaultStream()));
          astyped = target;
          source = target;
        }
        return readContiguousFloats(tmp, source);
      } finally {
        if (astyped != null) {
          mlx_h.mlx_array_free(astyped);
        }
      }
    }
  }

  /** Evaluates this array (if not already evaluated) and reads it back as a flat, row-major {@code int[]}. */
  public int[] toIntArray() {
    ensureOpen();
    if (dtype() != DType.INT32) {
      throw new IllegalStateException("toIntArray() requires INT32, got " + dtype());
    }
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment contiguous = mlx_h.mlx_array_new(tmp);
      try {
        NativeOps.checked("toIntArray", () -> mlx_h.mlx_contiguous(contiguous, handle, false, MLX.defaultStream()));
        NativeOps.checked("toIntArray", () -> mlx_h.mlx_array_eval(contiguous));
        long n = mlx_h.mlx_array_size(contiguous);
        if (n > Integer.MAX_VALUE) {
          throw new IllegalStateException("toIntArray() cannot represent " + n + " elements in a Java array"
              + " (limit " + Integer.MAX_VALUE + ")");
        }
        MemorySegment data = mlx_h.mlx_array_data_int32(contiguous).reinterpret(n * ValueLayout.JAVA_INT.byteSize());
        int[] result = new int[(int) n];
        MemorySegment.copy(data, ValueLayout.JAVA_INT, 0, result, 0, (int) n);
        return result;
      } finally {
        mlx_h.mlx_array_free(contiguous);
      }
    }
  }

  /**
   * Shared read-back body for {@link #toFloatArray()}: forces a contiguous row-major copy of {@code source} (which may
   * be {@code this.handle} or an astyped copy of it) and reads it into a {@code float[]}. {@code allocator} owns only
   * the contiguous handle this method allocates -- freed by its own {@code finally}, independent of whatever else the
   * caller allocated from the same arena.
   */
  private float[] readContiguousFloats(Arena allocator, MemorySegment source) {
    MemorySegment contiguous = mlx_h.mlx_array_new(allocator);
    try {
      NativeOps.checked("toFloatArray", () -> mlx_h.mlx_contiguous(contiguous, source, false, MLX.defaultStream()));
      NativeOps.checked("toFloatArray", () -> mlx_h.mlx_array_eval(contiguous));
      long n = mlx_h.mlx_array_size(contiguous);
      if (n > Integer.MAX_VALUE) {
        throw new IllegalStateException(
            "toFloatArray() cannot represent " + n + " elements in a Java array (limit " + Integer.MAX_VALUE + ")");
      }
      MemorySegment data = mlx_h.mlx_array_data_float32(contiguous).reinterpret(n * ValueLayout.JAVA_FLOAT.byteSize());
      float[] result = new float[(int) n];
      MemorySegment.copy(data, ValueLayout.JAVA_FLOAT, 0, result, 0, (int) n);
      return result;
    } finally {
      mlx_h.mlx_array_free(contiguous);
    }
  }

  /** Frees this array's native handle. Safe to call more than once. */
  @Override
  public void close() {
    if (closed) {
      return;
    }
    // free() first: if it throws (e.g. called from a non-owning thread),
    // closed must stay false, or this array becomes permanently unusable
    // via ensureOpen() while its native handle is still live. Matches the
    // ordering MLXScope.close() uses for the same reason.
    scope.free(handle);
    closed = true;
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("MLXArray[" + Integer.toHexString(System.identityHashCode(this)) + "] is closed");
    }
    scope.checkAccess();
  }
}
