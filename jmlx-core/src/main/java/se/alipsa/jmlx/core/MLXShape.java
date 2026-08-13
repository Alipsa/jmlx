package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Shape-manipulating ops: {@code reshape}, {@code broadcastTo}, {@code squeeze}, {@code transpose},
 * {@code slice}. Split out of {@link MLX} as pure motion in req/phase4-plan.md M0a -- see that
 * class's javadoc for the index of every sibling this facade was split into.
 */
public final class MLXShape {

  private MLXShape() {}

  /** Reshapes {@code a} to {@code shape}, which must have the same total element count. */
  public static MLXArray reshape(MLXArray a, int[] shape) {
    long targetSize = 1;
    for (int dim : shape) {
      targetSize *= dim;
    }
    if (targetSize != a.size()) {
      throw new IllegalArgumentException(
          "reshape: shape "
              + Arrays.toString(shape)
              + " (size "
              + targetSize
              + ") does not match array size "
              + a.size());
    }
    MLXScope scope = a.scope();
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "reshape",
          () ->
              mlx_h.mlx_reshape(
                  res, a.handle(), nativeShape, shape.length, NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }

  /** Broadcasts {@code a} to {@code targetShape}, per NumPy's directional broadcasting rules. */
  public static MLXArray broadcastTo(MLXArray a, int[] targetShape) {
    requireBroadcastableTo(a, targetShape);
    return NativeOps.shapeOp("broadcastTo", a, targetShape, mlx_h::mlx_broadcast_to);
  }

  /**
   * Directional broadcast check for {@code broadcastTo(a, targetShape)}: requires {@code
   * broadcast_shapes(a.shape(), targetShape) == targetShape} (upstream {@code ops.cpp:1601-1613}),
   * which is <em>not</em> the symmetric predicate {@link MLXOps}'s broadcast-compatibility check
   * uses -- {@code broadcastTo([3], [1])} satisfies the symmetric rule but native rejects it. Also
   * rejects any negative element of {@code targetShape} outright, independent of the shape
   * arithmetic: {@code broadcastTo([1], [-1])} would otherwise pass here, since {@code 1 == 1}
   * makes the directional check succeed too, and the failure would surface as {@link MLXException}
   * from native instead of {@link IllegalArgumentException} from Java.
   */
  private static void requireBroadcastableTo(MLXArray a, int[] targetShape) {
    for (int t : targetShape) {
      if (t < 0) {
        throw new IllegalArgumentException(
            "broadcastTo: target shape "
                + Arrays.toString(targetShape)
                + " has a negative dimension");
      }
    }
    int[] sa = a.shape();
    if (sa.length > targetShape.length) {
      throw new IllegalArgumentException(
          "broadcastTo: cannot broadcast "
              + Arrays.toString(sa)
              + " to "
              + Arrays.toString(targetShape));
    }
    int diff = targetShape.length - sa.length;
    for (int i = 0; i < sa.length; i++) {
      int ad = sa[i];
      int td = targetShape[i + diff];
      if (ad != td && ad != 1) {
        throw new IllegalArgumentException(
            "broadcastTo: cannot broadcast "
                + Arrays.toString(sa)
                + " to "
                + Arrays.toString(targetShape));
      }
    }
  }

  /** Removes every size-1 axis from {@code a}'s shape. */
  public static MLXArray squeeze(MLXArray a) {
    return NativeOps.unaryOp("squeeze", a, mlx_h::mlx_squeeze);
  }

  /**
   * Removes the given {@code axes} from {@code a}'s shape; native requires each to currently have
   * size 1.
   */
  public static MLXArray squeeze(MLXArray a, int[] axes) {
    return NativeOps.shapeOp("squeeze", a, axes, mlx_h::mlx_squeeze_axes);
  }

  /** Reverses every axis. */
  public static MLXArray transpose(MLXArray a) {
    return NativeOps.unaryOp("transpose", a, mlx_h::mlx_transpose);
  }

  /**
   * Permutes {@code a}'s axes according to {@code axes}, a permutation of {@code 0 .. a.ndim() -
   * 1}.
   */
  public static MLXArray transpose(MLXArray a, int[] axes) {
    return NativeOps.shapeOp("transpose", a, axes, mlx_h::mlx_transpose_axes);
  }

  /**
   * Reverses every axis, allocating the result into {@code target} instead of {@code a.scope()}.
   * See req/phase4-plan.md §2 mitigation 1: lets a weight-derived view computed inside {@code
   * forward()} land in the caller's (step) scope rather than leaking into {@code a}'s own (model)
   * scope once per call.
   */
  public static MLXArray transpose(MLXArray a, MLXScope target) {
    return NativeOps.unaryOp("transpose", a, target, mlx_h::mlx_transpose);
  }

  /** Swaps two axes. */
  public static MLXArray swapaxes(MLXArray a, int axis1, int axis2) {
    return NativeOps.axis2Op("swapaxes", a, axis1, axis2, mlx_h::mlx_swapaxes);
  }

  /**
   * Takes array entries at the given indices, treating the array as flattened regardless of its own
   * shape.
   */
  public static MLXArray take(MLXArray a, MLXArray indices) {
    return NativeOps.binaryOp("take", a, indices, mlx_h::mlx_take);
  }

  /**
   * Takes array slices at the given indices of the specified axis. Result shape is {@code
   * a.shape()[:axis] + indices.shape() + a.shape()[axis+1:]}.
   */
  public static MLXArray takeAxis(MLXArray a, MLXArray indices, int axis) {
    MLXScope scope = NativeOps.scopeOf("takeAxis", a, indices);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "takeAxis",
        () ->
            mlx_h.mlx_take_axis(res, a.handle(), indices.handle(), axis, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  /**
   * Slices {@code a} along every axis using {@code start} (inclusive) and {@code stop} (exclusive),
   * with every axis implicitly strided by 1. Equivalent to {@link #slice(MLXArray, int[], int[],
   * int[])} with an all-ones {@code strides}.
   */
  public static MLXArray slice(MLXArray a, int[] start, int[] stop) {
    requireSliceLengths(a, start, stop, null);
    int[] strides = new int[a.ndim()];
    Arrays.fill(strides, 1);
    return sliceNative(a, start, stop, strides);
  }

  /**
   * Slices {@code a} along every axis using {@code start} (inclusive), {@code stop} (exclusive) and
   * {@code strides}. Mirrors native's own {@code normalize_slice} (upstream {@code
   * ops.cpp:656-696}): negative {@code start}/ {@code stop} are normalized NumPy-style ({@code n +
   * i}); a negative stride reverses that axis; {@code stop} is clamped to the axis length rather
   * than bounds-checked. A zero stride is division-by-zero -- C++ undefined behaviour reachable
   * from this call, silently returning an empty axis on Apple Silicon rather than crashing or
   * erroring -- so it is rejected here with a message phrased after NumPy's own ("slice step cannot
   * be zero") rather than as a jmlx-specific restriction.
   */
  public static MLXArray slice(MLXArray a, int[] start, int[] stop, int[] strides) {
    requireSliceLengths(a, start, stop, strides);
    for (int i = 0; i < strides.length; i++) {
      if (strides[i] == 0) {
        throw new IllegalArgumentException(
            "slice: strides[" + i + "] must not be 0 (slice step cannot be zero)");
      }
    }
    return sliceNative(a, start, stop, strides);
  }

  /**
   * Mirrors upstream's own length check ({@code ops.cpp:757-763}, {@code "[slice] Invalid number of
   * indices or strides for array with dimension N."}) but names which of {@code start}/{@code
   * stop}/{@code strides} disagreed. {@code strides} is {@code null} for the 3-arg {@link
   * #slice(MLXArray, int[], int[])} overload, which synthesizes an all-ones {@code strides} of the
   * correct length itself and so has nothing to check there.
   */
  private static void requireSliceLengths(MLXArray a, int[] start, int[] stop, int[] strides) {
    int nd = a.ndim();
    if (start.length != nd || stop.length != nd || (strides != null && strides.length != nd)) {
      throw new IllegalArgumentException(
          "slice: start (length "
              + start.length
              + "), stop (length "
              + stop.length
              + ")"
              + (strides != null ? ", strides (length " + strides.length + ")" : "")
              + " must all equal a.ndim() ("
              + nd
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
      NativeOps.checked(
          "slice",
          () ->
              mlx_h.mlx_slice(
                  res,
                  a.handle(),
                  nativeStart,
                  start.length,
                  nativeStop,
                  stop.length,
                  nativeStrides,
                  strides.length,
                  NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }
}
