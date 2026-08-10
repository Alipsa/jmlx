package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * A native {@code mlx_array} handle owned by an {@link MLXScope}. See
 * req/initial-plan.md §6. Not thread-safe: confined to the scope's owning
 * thread, same as the scope itself.
 */
public final class MLXArray {

    private final MLXScope scope;
    private final MemorySegment handle;
    private volatile boolean closed = false;

    MLXArray(MLXScope scope, MemorySegment handle) {
        this.scope = scope;
        this.handle = handle;
    }

    MLXScope scope() {
        ensureOpen();
        return scope;
    }

    MemorySegment handle() {
        ensureOpen();
        return handle;
    }

    public int[] shape() {
        ensureOpen();
        int nd = ndim();
        MemorySegment shapePtr = mlx_h.mlx_array_shape(handle).reinterpret((long) nd * ValueLayout.JAVA_INT.byteSize());
        int[] result = new int[nd];
        MemorySegment.copy(shapePtr, ValueLayout.JAVA_INT, 0, result, 0, nd);
        return result;
    }

    public DType dtype() {
        ensureOpen();
        return DType.fromNative(mlx_h.mlx_array_dtype(handle));
    }

    public int ndim() {
        ensureOpen();
        return (int) mlx_h.mlx_array_ndim(handle);
    }

    public long size() {
        ensureOpen();
        return mlx_h.mlx_array_size(handle);
    }

    /**
     * Evaluates this array (if not already evaluated) and reads it back as a
     * flat, row-major {@code float[]}.
     *
     * <p>Forces a contiguous row-major copy first (mlx-c's
     * {@code mlx_contiguous}, {@code allow_col_major=false}) before reading
     * the data pointer. Lazy ops like transpose can yield a strided view;
     * reading raw data without this would return plausible-looking values in
     * the wrong element order -- a silent-wrong-answer bug, not a crash.
     */
    public float[] toFloatArray() {
        ensureOpen();
        if (dtype() != DType.FLOAT32) {
            throw new IllegalStateException("toFloatArray() requires FLOAT32, got " + dtype());
        }
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment contiguous = mlx_h.mlx_array_new(tmp);
            MLX.check(mlx_h.mlx_contiguous(contiguous, handle, false, MLX.defaultStream()));
            MLX.check(mlx_h.mlx_array_eval(contiguous));
            long n = mlx_h.mlx_array_size(contiguous);
            MemorySegment data =
                mlx_h.mlx_array_data_float32(contiguous).reinterpret(n * ValueLayout.JAVA_FLOAT.byteSize());
            float[] result = new float[(int) n];
            MemorySegment.copy(data, ValueLayout.JAVA_FLOAT, 0, result, 0, (int) n);
            MLX.check(mlx_h.mlx_array_free(contiguous));
            return result;
        }
    }

    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        scope.free(handle);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                "MLXArray[" + Integer.toHexString(System.identityHashCode(this)) + "] is closed");
        }
    }
}
