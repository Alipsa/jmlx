package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_array_;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Home for {@code seed}, {@code normal} and {@code uniform} (req/phase4-plan.md §1, §5), landed in
 * M1 for weight initialization onto a class created empty during M0a's pure-motion facade split, so
 * M1 could add ops to an address that already existed rather than growing {@link MLX} past the
 * point §1 named as its split trigger. See {@link MLX}'s javadoc for the index of every sibling
 * this facade was split into.
 */
public final class MLXRandom {

  private MLXRandom() {}

  /**
   * Seeds mlx's global RNG state ({@code mlx_random_seed}). No result and no stream: this affects
   * process-wide state.
   */
  public static void seed(long seed) {
    NativeOps.checked("seed", () -> mlx_h.mlx_random_seed(seed));
  }

  /** Creates an explicit UINT32 key with shape {@code [2]} from {@code seed}. */
  public static MLXArray key(MLXScope scope, long seed) {
    Objects.requireNonNull(scope, "scope").checkAccess();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked("key", () -> mlx_h.mlx_random_key(res, seed));
    return new MLXArray(scope, res);
  }

  /**
   * Splits {@code key} into {@code count} independent keys shaped {@code [count, 2]}, allocating
   * the result into {@code target}. The parent key is not mutated.
   */
  public static MLXArray split(MLXArray key, int count, MLXScope target) {
    requireKey(key, "split");
    Objects.requireNonNull(target, "target").checkAccess();
    if (count <= 0) {
      throw new IllegalArgumentException("split: count must be positive");
    }
    if (!key.scope().isAncestorOf(target)) {
      throw new IllegalArgumentException("split: target must be the key scope or a descendant");
    }
    MemorySegment res = mlx_h.mlx_array_new(target);
    NativeOps.checked(
        "split",
        () -> mlx_h.mlx_random_split_num(res, key.handle(), count, NativeOps.DEFAULT_STREAM));
    return new MLXArray(target, res);
  }

  /** Draws one categorical index along {@code axis} with an explicit key, returning INT32. */
  public static MLXArray categorical(MLXArray logits, int axis, MLXArray key) {
    Objects.requireNonNull(logits, "logits");
    requireKey(key, "categorical");
    if (logits.dtype() != DType.FLOAT32) {
      throw new IllegalArgumentException("categorical: logits must have dtype FLOAT32");
    }
    int normalizedAxis = axis < 0 ? axis + logits.ndim() : axis;
    if (normalizedAxis < 0 || normalizedAxis >= logits.ndim()) {
      throw new IllegalArgumentException("categorical: axis out of range: " + axis);
    }
    MLXScope scope = NativeOps.scopeOf("categorical", logits, key);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "categorical",
        () ->
            mlx_h.mlx_random_categorical(
                res, logits.handle(), normalizedAxis, key.handle(), NativeOps.DEFAULT_STREAM));
    return MLX.astype(new MLXArray(scope, res), DType.INT32);
  }

  private static void requireKey(MLXArray key, String operation) {
    Objects.requireNonNull(key, "key");
    if (key.dtype() != DType.UINT32 || !Arrays.equals(key.shape(), new int[] {2})) {
      throw new IllegalArgumentException(operation + ": key must have dtype UINT32 and shape [2]");
    }
  }

  /**
   * Draws a {@code shape}-shaped {@code dtype} array from a normal distribution ({@code
   * mlx_random_normal}) with mean {@code loc} and standard deviation {@code scale}. Unlike {@code
   * uniform}'s {@code low}/{@code high}, {@code loc}/ {@code scale} are genuinely plain {@code
   * float} parameters in the native signature, not {@code mlx_array} -- no scalar-array bridge is
   * needed here. {@code key} is always the RNG's own default (a Java {@code null} through {@link
   * NativeOps#nullableHandle}); this facade does not yet expose explicit keys.
   */
  public static MLXArray normal(MLXScope scope, int[] shape, DType dtype, float loc, float scale) {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
      MemorySegment key = NativeOps.nullableHandle(null, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "normal",
          () ->
              mlx_h.mlx_random_normal(
                  res,
                  nativeShape,
                  shape.length,
                  dtype.nativeValue(),
                  loc,
                  scale,
                  key,
                  NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  /**
   * Builds a single-element {@code dtype}-narrowed scalar {@code mlx_array} from {@code value}, via
   * mlx-c's own statusless {@code mlx_array_new_float32} -- the same null-{@code ctx}-on-failure
   * hazard {@link MLX#array} and {@link MLX#full} already handle. Factored out of {@link #uniform}
   * so the four-line check is not duplicated once for {@code low} and once for {@code high}.
   */
  private static MemorySegment float32Scalar(String opName, float value, Arena tmp) {
    NativeLoader.clearLastNativeError();
    MemorySegment scalar = mlx_h.mlx_array_new_float32(tmp, value);
    if (mlx_array_.ctx(scalar).address() == 0) {
      throw NativeOps.nativeFailure(opName + ": mlx_array_new_float32");
    }
    return scalar;
  }

  /**
   * Draws a {@code shape}-shaped {@code dtype} array from a uniform distribution over {@code [low,
   * high)} ({@code mlx_random_uniform}). Unlike {@code normal}'s {@code loc}/{@code scale}, mlx-c's
   * {@code low}/{@code high} are real {@code mlx_array} parameters passed by value, not {@code
   * float} -- so the Java-facing {@code float low}/ {@code float high} are each first turned into a
   * throwaway scalar array via {@link #float32Scalar}. Those scalars DO need an explicit free,
   * exactly like {@link MLX#full}'s scalar: {@code tmp} (a confined {@link Arena}) owns only the
   * 8-byte {@code mlx_array_} struct that {@code mlx_array_new_float32} writes into; the
   * heap-allocated {@code mlx::core::array} the struct's {@code ctx} points at is owned by mlx-c
   * and freed solely by {@code mlx_array_free} (see {@link MLXArray#toFloatArray()}'s javadoc for
   * the same invariant). {@code mlx_random_uniform} takes {@code low}/{@code high} as {@code const
   * mlx_array}, i.e. it borrows them rather than adopting them, so both scalars are freed in a
   * {@code finally}, mirroring {@link MLX#full}'s pattern. The two frees are nested -- {@code
   * highScalar}'s acquisition and its {@code finally} sit inside {@code lowScalar}'s -- so that if
   * acquiring {@code highScalar} itself throws (e.g. via {@link #float32Scalar}'s null-{@code ctx}
   * check), {@code lowScalar} is still freed rather than leaked. {@code key} is always the RNG's
   * own default, as in {@link #normal}.
   */
  public static MLXArray uniform(MLXScope scope, int[] shape, DType dtype, float low, float high) {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment lowScalar = float32Scalar("uniform", low, tmp);
      try {
        MemorySegment highScalar = float32Scalar("uniform", high, tmp);
        try {
          MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
          MemorySegment key = NativeOps.nullableHandle(null, tmp);
          MemorySegment res = mlx_h.mlx_array_new(scope);
          NativeOps.checked(
              "uniform",
              () ->
                  mlx_h.mlx_random_uniform(
                      res,
                      lowScalar,
                      highScalar,
                      nativeShape,
                      shape.length,
                      dtype.nativeValue(),
                      key,
                      NativeOps.DEFAULT_STREAM));
          return new MLXArray(scope, res);
        } finally {
          mlx_h.mlx_array_free(highScalar);
        }
      } finally {
        mlx_h.mlx_array_free(lowScalar);
      }
    }
  }
}
