package se.alipsa.jmlx.core;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Elementwise unary/binary ops, comparisons, reductions, {@code matmul}, {@code inner}/{@code
 * outer}. Split out of {@link MLX} as pure motion in req/phase4-plan.md M0a -- see that class's
 * javadoc for the index of every sibling this facade was split into.
 */
public final class MLXOps {

  private MLXOps() {}

  /**
   * Elementwise sum of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if they
   * differ.
   */
  public static MLXArray add(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "add");
    return addUnchecked(a, b);
  }

  /**
   * Elementwise difference of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules
   * if they differ.
   */
  public static MLXArray subtract(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "subtract");
    return NativeOps.binaryOp("subtract", a, b, mlx_h::mlx_subtract);
  }

  /**
   * Elementwise product of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if
   * they differ.
   */
  public static MLXArray multiply(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "multiply");
    return NativeOps.binaryOp("multiply", a, b, mlx_h::mlx_multiply);
  }

  /**
   * Elementwise quotient of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if
   * they differ.
   */
  public static MLXArray divide(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "divide");
    return NativeOps.binaryOp("divide", a, b, mlx_h::mlx_divide);
  }

  /**
   * Elementwise maximum of {@code a} and {@code b}, broadcasting their shapes per NumPy's rules if
   * they differ.
   */
  public static MLXArray maximum(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "maximum");
    return NativeOps.binaryOp("maximum", a, b, mlx_h::mlx_maximum);
  }

  /**
   * Elementwise power (a raised to the power b), broadcasting their shapes per NumPy's rules if
   * they differ.
   */
  public static MLXArray power(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "power");
    return NativeOps.binaryOp("power", a, b, mlx_h::mlx_power);
  }

  /**
   * Matrix product of {@code a} and {@code b}. Either may be rank-1 (promoted internally, matching
   * mlx's own vector-matrix / matrix-vector rules); rank &ge; 2 operands batch-broadcast over every
   * axis but the last two. Rank-0 operands are rejected, as mlx itself rejects them. Both operands
   * must have an inexact dtype (see {@link DType#isInexact()}); see {@link
   * #requireMatmulCompatible} for how that check relates to native's rule.
   */
  public static MLXArray matmul(MLXArray a, MLXArray b) {
    requireMatmulCompatible(a, b);
    return NativeOps.binaryOp("matmul", a, b, mlx_h::mlx_matmul);
  }

  /**
   * NumPy broadcast compatibility between {@code a} and {@code b}, right-aligned: for every
   * dimension pair, one of {@code d1 == d2}, {@code d1 == 1} or {@code d2 == 1} must hold (upstream
   * {@code mlx/utils.cpp:136-167}, {@code broadcast_shapes}). Deliberately written as {@code d1 ==
   * 1}/{@code d2 == 1}, not {@code d1 <= 1}/ {@code d2 <= 1}: the latter would wrongly accept a
   * {@code 0} dimension against a {@code 3}, which native rejects.
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
            op + ": incompatible shapes " + Arrays.toString(sa) + " and " + Arrays.toString(sb));
      }
    }
  }

  /**
   * Mirrors {@code matmul}'s real native rules (upstream {@code ops.cpp:3192-3267}), in evaluation
   * order: reject rank-0 on either operand; promote a rank-1 {@code a} to {@code [1, a.shape(0)]}
   * and a rank-1 {@code b} to {@code [b.shape(0), 1]} (mlx's own {@code expand_dims}); compare the
   * inner dimension on the <em>promoted</em> shapes, not the raw ones -- indexing {@code
   * b.shape(-2)} on a raw rank-1 {@code b} would be out of bounds; require {@link
   * DType#isInexact()} on both operands; then require the batch dimensions (every axis but the last
   * two of the promoted shapes) to be broadcast-compatible.
   *
   * <p>The dtype check requires <em>at least one</em> of {@code a}/{@code b} to be inexact -- not
   * each independently, and not {@code issubdtype(out_type, inexact)} on their actual promoted
   * type, which this facade has no promotion lattice to compute (see the rejected alternative
   * below). Both-exact is a provable <em>subset</em> of native's rejects: {@code
   * promote_types(exact, exact)} is always exact, so this can only under-reject relative to native
   * (for pairs this facade cannot even construct), never over-reject -- and over-rejecting, not
   * under-rejecting, is the failure mode a Java-side guard exists to avoid. req/phase4-plan.md §4's
   * {@code astype} is what makes a mixed exact/inexact pair (e.g. float32 x int32) reachable at
   * all; before it, nothing in this facade could construct an {@code INT32} array, so this method
   * could only ever be called with operands already both inexact or both exact.
   */
  private static void requireMatmulCompatible(MLXArray a, MLXArray b) {
    int[] sa = a.shape();
    int[] sb = b.shape();
    if (sa.length == 0 || sb.length == 0) {
      throw new IllegalArgumentException(
          "matmul: rank-0 operand not supported (shapes "
              + Arrays.toString(sa)
              + " and "
              + Arrays.toString(sb)
              + ")");
    }
    int[] pa = sa.length == 1 ? new int[] {1, sa[0]} : sa;
    int[] pb = sb.length == 1 ? new int[] {sb[0], 1} : sb;
    if (pa[pa.length - 1] != pb[pb.length - 2]) {
      throw new IllegalArgumentException(
          "matmul: incompatible shapes " + Arrays.toString(sa) + " and " + Arrays.toString(sb));
    }
    if (!a.dtype().isInexact() && !b.dtype().isInexact()) {
      throw new IllegalArgumentException(
          "matmul: requires at least one inexact dtype, got " + a.dtype() + " and " + b.dtype());
    }
    int batchA = pa.length - 2;
    int batchB = pb.length - 2;
    int batchDims = Math.max(batchA, batchB);
    for (int i = 0; i < batchDims; i++) {
      int da = i < batchDims - batchA ? 1 : pa[i - (batchDims - batchA)];
      int db = i < batchDims - batchB ? 1 : pb[i - (batchDims - batchB)];
      if (da != db && da != 1 && db != 1) {
        throw new IllegalArgumentException(
            "matmul: incompatible batch dimensions "
                + Arrays.toString(sa)
                + " and "
                + Arrays.toString(sb));
      }
    }
  }

  /**
   * Skips the Java-side shape check {@link #add} otherwise applies before ever reaching native.
   * Deliberate bypass (req/initial-plan.md, Testing approach) so a test can exercise a genuine
   * native error path and prove it surfaces as {@link MLXException} rather than a process abort.
   */
  static MLXArray addUnchecked(MLXArray a, MLXArray b) {
    return NativeOps.binaryOp("add", a, b, mlx_h::mlx_add);
  }

  /**
   * Blocks gradient flow through {@code a}: the forward value is unchanged, but any traced backward
   * pass treats {@code a} as a constant rather than differentiating through whatever produced it.
   * Fits {@code unaryOp}'s exact {@code (res, a, stream)} shape (upstream {@code
   * mlx_stop_gradient}), so it needs no hand-rolled body.
   */
  public static MLXArray stopGradient(MLXArray a) {
    return NativeOps.unaryOp("stopGradient", a, mlx_h::mlx_stop_gradient);
  }

  /** Elementwise natural exponential. */
  public static MLXArray exp(MLXArray a) {
    return NativeOps.unaryOp("exp", a, mlx_h::mlx_exp);
  }

  /** Elementwise natural logarithm. */
  public static MLXArray log(MLXArray a) {
    return NativeOps.unaryOp("log", a, mlx_h::mlx_log);
  }

  /** Elementwise sine. */
  public static MLXArray sin(MLXArray a) {
    return NativeOps.unaryOp("sin", a, mlx_h::mlx_sin);
  }

  /** Elementwise cosine. */
  public static MLXArray cos(MLXArray a) {
    return NativeOps.unaryOp("cos", a, mlx_h::mlx_cos);
  }

  /** Elementwise sigmoid. */
  public static MLXArray sigmoid(MLXArray a) {
    return NativeOps.unaryOp("sigmoid", a, mlx_h::mlx_sigmoid);
  }

  /** Elementwise error function. */
  public static MLXArray erf(MLXArray a) {
    return NativeOps.unaryOp("erf", a, mlx_h::mlx_erf);
  }

  /** Elementwise hyperbolic tangent. */
  public static MLXArray tanh(MLXArray a) {
    return NativeOps.unaryOp("tanh", a, mlx_h::mlx_tanh);
  }

  /** Elementwise square root. */
  public static MLXArray sqrt(MLXArray a) {
    return NativeOps.unaryOp("sqrt", a, mlx_h::mlx_sqrt);
  }

  /** Elementwise reciprocal square root. */
  public static MLXArray rsqrt(MLXArray a) {
    return NativeOps.unaryOp("rsqrt", a, mlx_h::mlx_rsqrt);
  }

  /** Elementwise square. */
  public static MLXArray square(MLXArray a) {
    return NativeOps.unaryOp("square", a, mlx_h::mlx_square);
  }

  /** Elementwise negation. */
  public static MLXArray negative(MLXArray a) {
    return NativeOps.unaryOp("negative", a, mlx_h::mlx_negative);
  }

  /** Sums every element to a rank-0 scalar array. */
  public static MLXArray sum(MLXArray a) {
    // mlx_sum(res, a, keepdims, s) carries an extra bool beyond unaryOp's
    // (res, a, stream) shape, so it does not fit that helper -- kept as its
    // own body rather than forcing it through a shape unaryOp doesn't have.
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked("sum", () -> mlx_h.mlx_sum(res, a.handle(), false, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  /**
   * Sum reduction along specified axes. {@code keepdims} controls whether reduced axes are kept as
   * singleton dimensions (true) or removed (false).
   */
  public static MLXArray sum(MLXArray a, int[] axes, boolean keepdims) {
    return NativeOps.reduceOp("sum", a, axes, keepdims, mlx_h::mlx_sum_axes);
  }

  /**
   * Index of the largest value along {@code axis}. The result has dtype {@link DType#INT32}; set
   * {@code keepdims} to retain the reduced axis as a singleton dimension. This is intentionally a
   * native reduction rather than a Java-side scan of {@link MLXArray#toFloatArray()}, so greedy
   * decoding can select its next token without copying an entire vocabulary's logits back from the
   * GPU.
   */
  public static MLXArray argmaxAxis(MLXArray a, int axis, boolean keepdims) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "argmaxAxis",
        () -> mlx_h.mlx_argmax_axis(res, a.handle(), axis, keepdims, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  /**
   * Mean reduction along specified axes. {@code keepdims} controls whether reduced axes are kept as
   * singleton dimensions (true) or removed (false).
   */
  public static MLXArray mean(MLXArray a, int[] axes, boolean keepdims) {
    return NativeOps.reduceOp("mean", a, axes, keepdims, mlx_h::mlx_mean_axes);
  }

  /**
   * Generalized inner product of {@code a} and {@code b}: contracts their last axes, matching
   * {@code mlx_inner} (and NumPy's {@code inner}). Requires {@code a.shape()[-1] == b.shape()[-1]}
   * unless either operand is rank-0, in which case native treats the call as a scalar multiply and
   * no last axis exists on that side to compare.
   */
  public static MLXArray inner(MLXArray a, MLXArray b) {
    int[] sa = a.shape();
    int[] sb = b.shape();
    if (sa.length > 0 && sb.length > 0 && sa[sa.length - 1] != sb[sb.length - 1]) {
      throw new IllegalArgumentException(
          "inner: last dimensions disagree: "
              + Arrays.toString(sa)
              + " and "
              + Arrays.toString(sb));
    }
    return NativeOps.binaryOp("inner", a, b, mlx_h::mlx_inner);
  }

  /**
   * Outer product of {@code a} and {@code b}: every pairwise product of their flattened elements,
   * matching {@code mlx_outer}. Guards {@code a.size() <= Integer.MAX_VALUE} because native
   * reshapes {@code a} via a {@code static_cast<int>(a.size())} with no bounds check of its own
   * (upstream {@code ops.cpp:5448-5451}); {@code b} goes through {@code flatten} instead, which
   * needs no such cast, so the guard is deliberately one-sided.
   */
  public static MLXArray outer(MLXArray a, MLXArray b) {
    if (a.size() > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "outer: a has " + a.size() + " elements, exceeding Integer.MAX_VALUE");
    }
    return NativeOps.binaryOp("outer", a, b, mlx_h::mlx_outer);
  }

  /** Elementwise {@code a < b}, broadcasting per NumPy's rules. Result dtype is {@code BOOL}. */
  public static MLXArray less(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "less");
    return NativeOps.binaryOp("less", a, b, mlx_h::mlx_less);
  }

  /** Elementwise {@code a <= b}. Result dtype is {@code BOOL}. */
  public static MLXArray lessEqual(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "lessEqual");
    return NativeOps.binaryOp("lessEqual", a, b, mlx_h::mlx_less_equal);
  }

  /** Elementwise {@code a > b}. Result dtype is {@code BOOL}. */
  public static MLXArray greater(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "greater");
    return NativeOps.binaryOp("greater", a, b, mlx_h::mlx_greater);
  }

  /** Elementwise {@code a >= b}. Result dtype is {@code BOOL}. */
  public static MLXArray greaterEqual(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "greaterEqual");
    return NativeOps.binaryOp("greaterEqual", a, b, mlx_h::mlx_greater_equal);
  }

  /** Elementwise {@code a == b}. Result dtype is {@code BOOL}. */
  public static MLXArray equal(MLXArray a, MLXArray b) {
    requireBroadcastCompatible(a, b, "equal");
    return NativeOps.binaryOp("equal", a, b, mlx_h::mlx_equal);
  }

  /**
   * Elementwise select: {@code x} where {@code condition} is nonzero, {@code y} otherwise. Three
   * array operands, none nullable -- resolves its target via {@link NativeOps#scopeOf} across all
   * three, same shape as this class's other hand-rolled bodies ({@link #inner}, {@link #outer}).
   */
  public static MLXArray where(MLXArray condition, MLXArray x, MLXArray y) {
    requireBroadcastCompatible(condition, x, "where");
    requireBroadcastCompatible(x, y, "where");
    requireBroadcastCompatible(condition, y, "where");
    MLXScope scope = NativeOps.scopeOf("where", condition, x, y);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "where",
        () ->
            mlx_h.mlx_where(
                res, condition.handle(), x.handle(), y.handle(), NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }

  /**
   * Softmax along {@code axis}. {@code precise} requests float32 accumulation regardless of {@code
   * a}'s dtype (mlx-c's own {@code precise} flag) -- carries an extra {@code bool} beyond {@link
   * NativeOps#axisOp}'s {@code (res, a, int, stream)} shape, so it is hand-rolled rather than
   * forced through that helper, the same reason {@link #sum(MLXArray)} is hand-rolled beyond {@code
   * unaryOp}.
   */
  public static MLXArray softmaxAxis(MLXArray a, int axis, boolean precise) {
    MLXScope scope = a.scope();
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "softmaxAxis",
        () -> mlx_h.mlx_softmax_axis(res, a.handle(), axis, precise, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}
