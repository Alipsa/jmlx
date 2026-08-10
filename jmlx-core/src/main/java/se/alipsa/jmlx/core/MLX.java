package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Static facade over the mlx-c ops used by this slice. Every op here only
 * builds the lazy computation graph; nothing runs on the GPU/CPU until
 * {@link #eval} (or the implicit eval inside {@link MLXArray#toFloatArray()})
 * triggers it. See req/initial-plan.md §7.
 *
 * <p>Every op's result is allocated in the same scope as its first
 * {@code MLXArray} operand; {@link #array} takes the scope explicitly since
 * it has no operand to infer one from.
 *
 * <p>{@link #defaultDevice()}/{@link #defaultStream()} are resolved once,
 * lazily, from whatever mlx-c's own default device is at first use, and
 * cached for the process lifetime -- this slice does not expose device
 * switching, so there is no stale-cache hazard in practice.
 */
public final class MLX {

    private MLX() {}

    private static final Arena FACADE_ARENA = Arena.ofShared();
    private static final MemorySegment DEFAULT_DEVICE = resolveDefaultDevice();
    private static final MemorySegment DEFAULT_STREAM = resolveDefaultStream();

    private static MemorySegment resolveDefaultDevice() {
        MemorySegment dev = mlx_h.mlx_device_new(FACADE_ARENA);
        check(mlx_h.mlx_get_default_device(dev));
        return dev;
    }

    private static MemorySegment resolveDefaultStream() {
        MemorySegment stream = mlx_h.mlx_stream_new(FACADE_ARENA);
        check(mlx_h.mlx_get_default_stream(stream, DEFAULT_DEVICE));
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

    public static MLXArray array(MLXScope scope, float[] data, int[] shape) {
        long expected = 1;
        for (int dim : shape) {
            expected *= dim;
        }
        if (expected != data.length) {
            throw new IllegalArgumentException(
                "shape " + java.util.Arrays.toString(shape) + " (size " + expected
                    + ") does not match data length " + data.length);
        }
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment nativeData = tmp.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
            MemorySegment handle =
                mlx_h.mlx_array_new_data(scope, nativeData, nativeShape, shape.length, mlx_h.MLX_FLOAT32());
            return new MLXArray(scope, handle);
        }
    }

    public static MLXArray add(MLXArray a, MLXArray b) {
        requireSameShape(a, b, "add");
        return addUnchecked(a, b);
    }

    public static MLXArray subtract(MLXArray a, MLXArray b) {
        requireSameShape(a, b, "subtract");
        return binaryOp(a, b, mlx_h::mlx_subtract);
    }

    public static MLXArray multiply(MLXArray a, MLXArray b) {
        requireSameShape(a, b, "multiply");
        return binaryOp(a, b, mlx_h::mlx_multiply);
    }

    public static MLXArray divide(MLXArray a, MLXArray b) {
        requireSameShape(a, b, "divide");
        return binaryOp(a, b, mlx_h::mlx_divide);
    }

    public static MLXArray matmul(MLXArray a, MLXArray b) {
        if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0]) {
            throw new IllegalArgumentException(
                "matmul: incompatible shapes " + java.util.Arrays.toString(a.shape())
                    + " and " + java.util.Arrays.toString(b.shape()));
        }
        return binaryOp(a, b, mlx_h::mlx_matmul);
    }

    private static void requireSameShape(MLXArray a, MLXArray b, String op) {
        if (!java.util.Arrays.equals(a.shape(), b.shape())) {
            throw new IllegalArgumentException(
                op + ": shape mismatch " + java.util.Arrays.toString(a.shape())
                    + " vs " + java.util.Arrays.toString(b.shape()));
        }
    }

    /**
     * Skips the Java-side shape check {@link #add} otherwise applies before
     * ever reaching native. Deliberate bypass (req/initial-plan.md, Testing
     * approach) so a test can exercise a genuine native error path and prove
     * it surfaces as {@link MLXException} rather than a process abort.
     */
    static MLXArray addUnchecked(MLXArray a, MLXArray b) {
        return binaryOp(a, b, mlx_h::mlx_add);
    }

    private static MLXArray binaryOp(MLXArray a, MLXArray b, BinaryOp op) {
        MLXScope scope = a.scope();
        MemorySegment res = mlx_h.mlx_array_new(scope);
        check(op.apply(res, a.handle(), b.handle(), DEFAULT_STREAM));
        return new MLXArray(scope, res);
    }

    @FunctionalInterface
    private interface BinaryOp {
        int apply(MemorySegment res, MemorySegment a, MemorySegment b, MemorySegment stream);
    }

    public static MLXArray exp(MLXArray a) {
        MLXScope scope = a.scope();
        MemorySegment res = mlx_h.mlx_array_new(scope);
        check(mlx_h.mlx_exp(res, a.handle(), DEFAULT_STREAM));
        return new MLXArray(scope, res);
    }

    /** Sums every element to a rank-0 scalar array. */
    public static MLXArray sum(MLXArray a) {
        MLXScope scope = a.scope();
        MemorySegment res = mlx_h.mlx_array_new(scope);
        check(mlx_h.mlx_sum(res, a.handle(), false, DEFAULT_STREAM));
        return new MLXArray(scope, res);
    }

    public static MLXArray reshape(MLXArray a, int[] shape) {
        long targetSize = 1;
        for (int dim : shape) {
            targetSize *= dim;
        }
        if (targetSize != a.size()) {
            throw new IllegalArgumentException(
                "reshape: shape " + java.util.Arrays.toString(shape) + " (size " + targetSize
                    + ") does not match array size " + a.size());
        }
        MLXScope scope = a.scope();
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
            MemorySegment res = mlx_h.mlx_array_new(scope);
            check(mlx_h.mlx_reshape(res, a.handle(), nativeShape, shape.length, DEFAULT_STREAM));
            return new MLXArray(scope, res);
        }
    }

    /** Reverses every axis; there is no partial-permutation overload in this slice. */
    public static MLXArray transpose(MLXArray a) {
        MLXScope scope = a.scope();
        MemorySegment res = mlx_h.mlx_array_new(scope);
        check(mlx_h.mlx_transpose(res, a.handle(), DEFAULT_STREAM));
        return new MLXArray(scope, res);
    }

    // Not code evaluation: mirrors mlx-c's mlx_array_eval, which forces a
    // lazily-built computation graph to actually run on device.
    /** Explicitly triggers computation for the given arrays. */
    public static void eval(MLXArray... arrays) {
        for (MLXArray a : arrays) {
            check(mlx_h.mlx_array_eval(a.handle()));
        }
    }

    static void check(int status) {
        if (status != 0) {
            String nativeMessage = se.alipsa.jmlx.ffi.NativeLoader.lastNativeError();
            se.alipsa.jmlx.ffi.NativeLoader.clearLastNativeError();
            throw new MLXException(
                "mlx-c call failed with status " + status
                    + (nativeMessage != null ? ": " + nativeMessage : ""));
        }
    }
}
