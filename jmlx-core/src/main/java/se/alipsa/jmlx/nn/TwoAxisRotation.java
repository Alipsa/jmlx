package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;

/**
 * Per-image rotation state one {@link TwoAxisRope#prepare} produces and every vision block of that
 * image shares: the float32 {@code cos}/{@code sin} tables of shape {@code [sequence, 1, 2 *
 * rotaryDims]} (broadcastable over the heads axis) plus the per-row frequency width. Threading
 * exactly like {@link MultiAxisRotation}: computed once per image, handed to each rotary attention,
 * never cached between images.
 *
 * <p>{@link #rotate} applies the frozen composition of transformers 4.57.6 {@code
 * apply_rotary_pos_emb_vision} (verbatim in req/plans/phase7-3b-reference-findings.md §11.1) with
 * that function's exact compute order: the INPUT is upcast to float32 ({@code q.float()}) and the
 * whole rotation {@code out = (x * cos) + (rotate_half(x) * sin)} runs in float32 against the
 * float32 tables, and only then is the result cast back to {@code x.dtype()} ({@code
 * q_embed.to(orig_q_dtype)}). This is the OPPOSITE cast order of {@link MultiAxisRotation#rotate},
 * which casts the float32 cos/sin tables DOWN to the input dtype and computes in the input dtype
 * (findings §11.2 "Vision RoPE"): the two orders agree bit-for-bit for float32 inputs but diverge
 * for half-precision ones, and each component is frozen to its own reference function rather than
 * unified.
 */
public final class TwoAxisRotation {

  private final MLXArray cos;
  private final MLXArray sin;
  private final int rotaryDims;

  TwoAxisRotation(MLXArray cos, MLXArray sin, int rotaryDims) {
    this.cos = cos;
    this.sin = sin;
    this.rotaryDims = rotaryDims;
  }

  /** Returns the float32 cosine table, {@code [sequence, 1, 2 * rotaryDims]}. */
  public MLXArray cos() {
    return cos;
  }

  /** Returns the float32 sine table, {@code [sequence, 1, 2 * rotaryDims]}. */
  public MLXArray sin() {
    return sin;
  }

  /**
   * Returns the per-row frequency width (the prepared head dimension is {@code 2 * rotaryDims}).
   */
  public int rotaryDims() {
    return rotaryDims;
  }

  /**
   * The rotation-shape preflight the attention seam runs before any array is written: the sequence
   * axis of the heads to be rotated must match the prepared rotation, and the head dimension must
   * be exactly {@code 2 * rotaryDims} (the reference's doubled table width -- unlike {@link
   * MultiAxisRotation#validate}, whose rotated prefix may be narrower than the head). All
   * validation is {@link IllegalArgumentException} and nothing is allocated. {@link #rotate}
   * delegates its shape checks here.
   */
  public void validate(int sequence, int headDim) {
    int[] cosShape = cos.shape();
    if (sequence != cosShape[0]) {
      throw new IllegalArgumentException(
          "rotation sequence must match the prepared rotation ("
              + sequence
              + " against "
              + cosShape[0]
              + ")");
    }
    if (headDim != 2 * rotaryDims) {
      throw new IllegalArgumentException(
          "headDim " + headDim + " must equal 2 * rotaryDims (" + (2 * rotaryDims) + ")");
    }
  }

  /**
   * Rotates a {@code [sequence, heads, 2 * rotaryDims]} tensor. All validation is {@link
   * IllegalArgumentException} and runs before any array is written: {@code x} is non-null with rank
   * 3, an inexact dtype (an exact dtype cast back would silently truncate, the same reason {@code
   * toFloatArray()} rejects exact dtypes), and its sequence and head dimensions match the prepared
   * rotation. Then the vision compute order (see this class's javadoc) runs: {@code xf = x} or
   * {@code astype(x, FLOAT32)}; {@code rotate_half(xf) = concat((−xf[…, n:], xf[…, :n]), -1)} with
   * {@code n = headDim / 2}; {@code out = (xf * cos) + (rotate_half(xf) * sin)} in float32, the
   * {@code [sequence, 1, ...]} tables broadcasting over the heads axis; finally {@code out} cast
   * back to {@code x.dtype()} (a no-op for float32). Every built array allocates per the usual
   * {@code scopeOf} rule (the innermost related scope of its operands).
   */
  public MLXArray rotate(MLXArray x) {
    if (x == null) {
      throw new IllegalArgumentException("x must not be null");
    }
    int[] shape = x.shape();
    if (shape.length != 3) {
      throw new IllegalArgumentException(
          "rotate requires [sequence, heads, headDim], got shape " + Arrays.toString(shape));
    }
    validate(shape[0], shape[2]);
    DType dtype = x.dtype();
    if (!dtype.isInexact()) {
      throw new IllegalArgumentException("rotate requires an inexact dtype, got " + dtype);
    }
    MLXArray xf = dtype == DType.FLOAT32 ? x : MLX.astype(x, DType.FLOAT32);
    int half = shape[2] / 2;
    int[] fullStop = shape.clone();
    int[] frontStop = shape.clone();
    frontStop[2] = half;
    MLXArray front = MLXShape.slice(xf, new int[] {0, 0, 0}, frontStop);
    int[] backStart = new int[] {0, 0, half};
    MLXArray back = MLXShape.slice(xf, backStart, fullStop);
    MLXArray rotated = MLXShape.concatenate(new MLXArray[] {MLXOps.negative(back), front}, 2);
    MLXArray out = MLXOps.add(MLXOps.multiply(xf, cos), MLXOps.multiply(rotated, sin));
    return dtype == DType.FLOAT32 ? out : MLX.astype(out, dtype);
  }
}
