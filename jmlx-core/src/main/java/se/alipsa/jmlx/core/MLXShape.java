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
          () -> mlx_h.mlx_reshape(res, a.handle(), nativeShape, shape.length, scope.stream()));
      return new MLXArray(scope, res);
    }
  }

  /** Broadcasts {@code a} to {@code targetShape}, per NumPy's directional broadcasting rules. */
  public static MLXArray broadcastTo(MLXArray a, int[] targetShape) {
    requireBroadcastableTo(a, targetShape);
    return NativeOps.shapeOp("broadcastTo", a, targetShape, mlx_h::mlx_broadcast_to);
  }

  /** Broadcasts into {@code target}, which must be related to {@code a}'s scope. */
  public static MLXArray broadcastTo(MLXArray a, MLXScope target, int[] targetShape) {
    requireBroadcastableTo(a, targetShape);
    return NativeOps.shapeOp("broadcastTo", a, target, targetShape, mlx_h::mlx_broadcast_to);
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
   * Swaps two axes, allocating the result into {@code target} instead of {@code a.scope()} -- the
   * {@code swapaxes} counterpart of {@link #transpose(MLXArray, MLXScope)}, for a weight-derived
   * view computed inside {@code forward()} that must land in the step scope.
   */
  public static MLXArray swapaxes(MLXArray a, MLXScope target, int axis1, int axis2) {
    return NativeOps.axis2Op("swapaxes", a, target, axis1, axis2, mlx_h::mlx_swapaxes);
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
        () -> mlx_h.mlx_take_axis(res, a.handle(), indices.handle(), axis, scope.stream()));
    return new MLXArray(scope, res);
  }

  /**
   * Takes slices into an explicit destination scope. Callers that need storage independent of
   * {@code a}'s scope must evaluate the result before releasing the source scope.
   */
  public static MLXArray takeAxis(MLXArray a, MLXArray indices, MLXScope destination, int axis) {
    if (destination == null) {
      throw new IllegalArgumentException("takeAxis destination must not be null");
    }
    MemorySegment res = mlx_h.mlx_array_new(destination);
    NativeOps.checked(
        "takeAxis",
        () -> mlx_h.mlx_take_axis(res, a.handle(), indices.handle(), axis, destination.stream()));
    return new MLXArray(destination, res);
  }

  /**
   * Takes values elementwise along {@code axis}. Unlike {@link #takeAxis}, this does not insert the
   * full indices shape: {@code indices} must be broadcast-compatible with {@code a} outside the
   * selected axis.
   */
  public static MLXArray takeAlongAxis(MLXArray a, MLXArray indices, int axis) {
    MLXScope scope = NativeOps.scopeOf("takeAlongAxis", a, indices);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "takeAlongAxis",
        () -> mlx_h.mlx_take_along_axis(res, a.handle(), indices.handle(), axis, scope.stream()));
    return new MLXArray(scope, res);
  }

  /**
   * Returns {@code a} with {@code values} written at elementwise {@code indices} along {@code
   * axis}. This operation is lazy, like {@link #take} and {@link #takeAxis}. The pinned native
   * operation does not bounds-check index values, so callers must establish that every index is
   * valid.
   */
  public static MLXArray putAlongAxis(MLXArray a, MLXArray indices, MLXArray values, int axis) {
    MLXScope scope = NativeOps.scopeOf("putAlongAxis", a, indices, values);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "putAlongAxis",
        () ->
            mlx_h.mlx_put_along_axis(
                res, a.handle(), indices.handle(), values.handle(), axis, scope.stream()));
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
                  scope.stream()));
      return new MLXArray(scope, res);
    }
  }

  /** Concatenates {@code arrays} along {@code axis}; every array must agree on every other axis. */
  public static MLXArray concatenate(MLXArray[] arrays, int axis) {
    if (arrays.length == 0) {
      throw new IllegalArgumentException("concatenate: requires at least one array");
    }
    for (int i = 0; i < arrays.length; i++) {
      if (arrays[i] == null) {
        throw new IllegalArgumentException("concatenate: arrays[" + i + "] must not be null");
      }
    }
    return NativeOps.vectorInOp("concatenate", arrays, axis, mlx_h::mlx_concatenate_axis);
  }

  /** Stacks equal-shaped {@code arrays} along a new axis inserted at {@code axis}. */
  public static MLXArray stack(MLXArray[] arrays, int axis) {
    if (arrays.length == 0) {
      throw new IllegalArgumentException("stack: requires at least one array");
    }
    for (int i = 0; i < arrays.length; i++) {
      if (arrays[i] == null) {
        throw new IllegalArgumentException("stack: arrays[" + i + "] must not be null");
      }
    }
    return NativeOps.vectorInOp("stack", arrays, axis, mlx_h::mlx_stack_axis);
  }

  /** Splits {@code a} into {@code numSplits} equal-size parts along {@code axis}, in order. */
  public static MLXArray[] split(MLXArray a, int numSplits, int axis) {
    return NativeOps.vectorOutOp(
        "split",
        a.scope(),
        (vec, stream) -> mlx_h.mlx_split(vec, a.handle(), numSplits, axis, stream));
  }

  /** Inserts a new size-1 axis at position {@code axis}. */
  public static MLXArray expandDims(MLXArray a, int axis) {
    return NativeOps.axisOp("expandDims", a, axis, mlx_h::mlx_expand_dims);
  }

  /** Inserts a size-1 axis, allocating the result into {@code target}. */
  public static MLXArray expandDims(MLXArray a, MLXScope target, int axis) {
    return NativeOps.axisOp("expandDims", a, target, axis, mlx_h::mlx_expand_dims);
  }

  /**
   * Merges the axes from {@code startAxis} to {@code endAxis} (inclusive) into a single axis. Fits
   * the existing {@link NativeOps#axis2Op} exactly -- same {@code (res, a, int, int, stream)} shape
   * as {@code mlx_swapaxes} -- so no new helper is needed for this op.
   */
  public static MLXArray flatten(MLXArray a, int startAxis, int endAxis) {
    return NativeOps.axis2Op("flatten", a, startAxis, endAxis, mlx_h::mlx_flatten);
  }

  /** Merges axes, allocating the result into {@code target}. */
  public static MLXArray flatten(MLXArray a, MLXScope target, int startAxis, int endAxis) {
    return NativeOps.axis2Op("flatten", a, target, startAxis, endAxis, mlx_h::mlx_flatten);
  }

  /**
   * Zeroes every element strictly above the {@code k}-th diagonal (the lower triangle kept extends
   * {@code k} diagonals above the main one; {@code k=0} keeps the main diagonal).
   */
  public static MLXArray tril(MLXArray a, int k) {
    return NativeOps.axisOp("tril", a, k, mlx_h::mlx_tril);
  }

  /**
   * Zeroes every element strictly below the {@code k}-th diagonal -- the complement of {@link
   * #tril}. {@code triu(ones(shape, BOOL), 1)} is the standard strictly-upper causal mask.
   */
  public static MLXArray triu(MLXArray a, int k) {
    return NativeOps.axisOp("triu", a, k, mlx_h::mlx_triu);
  }

  /** Confirmed native padding modes. Symmetric widths are independent of the mode. */
  public enum PadMode {
    CONSTANT,
    EDGE
  }

  /** Pads selected axes with scalar value, using constant or edge padding. */
  public static MLXArray pad(
      MLXArray a, int[] axes, int[] low, int[] high, MLXArray value, PadMode mode) {
    if (axes.length != low.length || axes.length != high.length) {
      throw new IllegalArgumentException("pad: axes and widths must have equal lengths");
    }
    if (value.size() != 1) {
      throw new IllegalArgumentException("pad: value must be scalar");
    }
    MLXScope scope = NativeOps.scopeOf("pad", a, value);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment na = tmp.allocateFrom(ValueLayout.JAVA_INT, axes);
      MemorySegment nl = tmp.allocateFrom(ValueLayout.JAVA_INT, low);
      MemorySegment nh = tmp.allocateFrom(ValueLayout.JAVA_INT, high);
      MemorySegment nm = tmp.allocateFrom(mode.name().toLowerCase(java.util.Locale.ROOT));
      MemorySegment result = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "pad",
          () ->
              mlx_h.mlx_pad(
                  result,
                  a.handle(),
                  na,
                  axes.length,
                  nl,
                  low.length,
                  nh,
                  high.length,
                  value.handle(),
                  nm,
                  scope.stream()));
      return new MLXArray(scope, result);
    }
  }

  /** Equal padding widths on every axis; this does not mean reflection padding. */
  public static MLXArray padSymmetric(MLXArray a, int width, MLXArray value, PadMode mode) {
    MLXScope scope = NativeOps.scopeOf("padSymmetric", a, value);
    if (value.size() != 1) {
      throw new IllegalArgumentException("pad: value must be scalar");
    }
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment nm = tmp.allocateFrom(mode.name().toLowerCase(java.util.Locale.ROOT));
      MemorySegment result = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "padSymmetric",
          () ->
              mlx_h.mlx_pad_symmetric(
                  result, a.handle(), width, value.handle(), nm, scope.stream()));
      return new MLXArray(scope, result);
    }
  }

  /** Tiles with leading-axis alignment and optional rank extension; zero repetitions are legal. */
  public static MLXArray tile(MLXArray a, int[] repetitions) {
    for (int n : repetitions) {
      if (n < 0) {
        throw new IllegalArgumentException("tile: negative repetition");
      }
    }
    return NativeOps.shapeOp("tile", a, repetitions, mlx_h::mlx_tile);
  }

  /** Repeats along an axis; zero repeats produces an empty axis. */
  public static MLXArray repeatAxis(MLXArray a, int repeats, int axis) {
    if (repeats < 0) {
      throw new IllegalArgumentException("repeatAxis: negative repeats");
    }
    int normalized = axis < 0 ? axis + a.ndim() : axis;
    if (normalized < 0 || normalized >= a.ndim()) {
      throw new IllegalArgumentException("repeatAxis: axis out of range: " + axis);
    }
    return NativeOps.axis2Op("repeatAxis", a, repeats, normalized, mlx_h::mlx_repeat_axis);
  }

  /**
   * Checked view in element strides of a row-contiguous normalization of the input. Offset zero
   * addresses the logical start, including sliced inputs. Negative and zero strides are supported.
   * No reachable address may lie outside the normalized input; arithmetic overflow is rejected.
   */
  public static MLXArray asStrided(MLXArray a, int[] shape, long[] strides, long offset) {
    if (shape.length != strides.length || offset < 0) {
      throw new IllegalArgumentException("asStrided: rank mismatch or negative offset");
    }
    boolean empty = false;
    for (int n : shape) {
      if (n < 0) {
        throw new IllegalArgumentException("asStrided: negative extent");
      }
      empty |= n == 0;
    }
    try {
      long low = offset;
      long high = offset;
      if (!empty) {
        for (int i = 0; i < shape.length; i++) {
          long reach = Math.multiplyExact(shape[i] - 1L, strides[i]);
          low = Math.addExact(low, Math.min(0, reach));
          high = Math.addExact(high, Math.max(0, reach));
        }
        if (low < 0 || high >= a.size()) {
          throw new IllegalArgumentException("asStrided: reachable addresses outside input");
        }
      }
    } catch (ArithmeticException error) {
      throw new IllegalArgumentException("asStrided: address arithmetic overflow", error);
    }
    MLXScope scope = a.scope();
    MLXArray normalized =
        NativeOps.unaryOp(
            "contiguous",
            a,
            (res, input, stream) -> mlx_h.mlx_contiguous(res, input, false, stream));
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment ns = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
      MemorySegment nt = tmp.allocateFrom(ValueLayout.JAVA_LONG, strides);
      MemorySegment result = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "asStrided",
          () ->
              mlx_h.mlx_as_strided(
                  result,
                  normalized.handle(),
                  ns,
                  shape.length,
                  nt,
                  strides.length,
                  offset,
                  scope.stream()));
      return new MLXArray(scope, result);
    }
  }
}
