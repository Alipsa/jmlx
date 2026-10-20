package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;

/**
 * Per-step rotation state one {@link MultiAxisRope#prepare} produces and every layer of that step
 * shares: the float32 {@code cos}/{@code sin} tables of shape {@code [batch, 1, sequence,
 * rotaryDims]} (broadcastable over the heads axis) plus the width of the rotated head prefix.
 * Threading exactly like the existing per-step {@code stepFrequencies}: computed once per step,
 * handed to each attention, never cached between steps.
 *
 * <p>{@link #rotate} applies the frozen composition of req/plans/phase7-3b-reference-findings.md
 * §6.9 in the input array's dtype: the cos/sin tables are cast to {@code x.dtype()} (a no-op for
 * the FLOAT32 model) and only the {@code rotaryDims} prefix of each head is rotated with the NeoX
 * half-split {@code rotate_half = cat((−x[…, n:], x[…, :n]), −1)} -- the same layout the fast
 * kernels use with {@code traditional=false} -- while the tail {@code [rotaryDims, headDim)} passes
 * through bit-identical (partial rotary).
 */
public final class MultiAxisRotation {

  private final MLXArray cos;
  private final MLXArray sin;
  private final int rotaryDims;

  MultiAxisRotation(MLXArray cos, MLXArray sin, int rotaryDims) {
    this.cos = cos;
    this.sin = sin;
    this.rotaryDims = rotaryDims;
  }

  /** Returns the float32 cosine table, {@code [batch, 1, sequence, rotaryDims]}. */
  public MLXArray cos() {
    return cos;
  }

  /** Returns the float32 sine table, {@code [batch, 1, sequence, rotaryDims]}. */
  public MLXArray sin() {
    return sin;
  }

  /** Returns the width of the rotated head prefix. */
  public int rotaryDims() {
    return rotaryDims;
  }

  /**
   * The rotation-shape preflight the decoder seam runs before any array is written: the batch and
   * sequence axes of the heads to be rotated must match the prepared rotation, and the rotated
   * prefix must fit within the head width. All validation is {@link IllegalArgumentException} and
   * nothing is allocated. {@link #rotate} delegates its shape checks here, and the {@code
   * DecoderAttention}/{@code DecoderModel} rotation entries call {@link #validateAgainst(int, int,
   * int, int)} -- which additionally requires the rotation's own width to equal the decoder's
   * configured rotary width -- before any projection, mask or embedding builds a graph node, so a
   * mismatched rotation is rejected before any array is written.
   */
  public void validate(int batch, int sequence, int headDim) {
    int[] cosShape = cos.shape();
    if (batch != cosShape[0] || sequence != cosShape[2]) {
      throw new IllegalArgumentException(
          "rotation batch and sequence must match the prepared rotation ("
              + batch
              + ", "
              + sequence
              + " against "
              + cosShape[0]
              + ", "
              + cosShape[2]
              + ")");
    }
    if (rotaryDims > headDim) {
      throw new IllegalArgumentException("rotaryDims exceeds head dimension");
    }
  }

  /**
   * The width-aware rotation preflight the {@code DecoderAttention}/{@code DecoderModel} rotation
   * entries run at their earliest point: {@link #validate(int, int, int)} plus the requirement that
   * the rotation's own width equals the decoder's configured rotary width (the partial-rotary width
   * from the rope spec or model descriptor) -- a rotation prepared from a different mrope-section
   * sum would otherwise pass the prefix-fit check and silently rotate a different head prefix. All
   * validation is {@link IllegalArgumentException} and nothing is allocated.
   */
  public void validateAgainst(int batch, int sequence, int headDim, int configuredRotaryDims) {
    validate(batch, sequence, headDim);
    if (rotaryDims != configuredRotaryDims) {
      throw new IllegalArgumentException(
          "rotation rotaryDims "
              + rotaryDims
              + " does not match the configured rotary width "
              + configuredRotaryDims);
    }
  }

  /**
   * Rotates a {@code [batch, heads, sequence, headDim]} tensor. All validation is {@link
   * IllegalArgumentException} and runs before any array is written: {@code x} is non-null with rank
   * 4, its batch and sequence axes match the prepared rotation, and {@code rotaryDims} does not
   * exceed the head dimension. Then the cos/sin tables are cast to {@code x.dtype()} (frozen step
   * 5) and, in the input dtype, {@code out = x * cos + rotate_half(x) * sin} on the {@code
   * rotaryDims} prefix with the untouched tail concatenated back (frozen step 6). Every built array
   * allocates per the usual {@code scopeOf} rule (the innermost related scope of its operands).
   */
  public MLXArray rotate(MLXArray x) {
    Objects.requireNonNull(x, "x");
    int[] shape = x.shape();
    if (shape.length != 4) {
      throw new IllegalArgumentException(
          "rotate requires [batch, heads, sequence, headDim], got shape "
              + java.util.Arrays.toString(shape));
    }
    validate(shape[0], shape[2], shape[3]);
    DType dtype = x.dtype();
    MLXArray cosInput = cos.dtype() == dtype ? cos : MLX.astype(cos, dtype);
    MLXArray sinInput = sin.dtype() == dtype ? sin : MLX.astype(sin, dtype);
    int[] prefixStop = shape.clone();
    prefixStop[3] = rotaryDims;
    MLXArray prefix = rotaryDims == shape[3] ? x : MLXShape.slice(x, new int[4], prefixStop);
    int half = rotaryDims / 2;
    int[] halfStop = shape.clone();
    halfStop[3] = half;
    MLXArray front = MLXShape.slice(prefix, new int[4], halfStop);
    MLXArray back = MLXShape.slice(prefix, new int[] {0, 0, 0, half}, prefixStop);
    MLXArray rotated = MLXShape.concatenate(new MLXArray[] {MLXOps.negative(back), front}, 3);
    MLXArray out =
        MLXOps.add(MLXOps.multiply(prefix, cosInput), MLXOps.multiply(rotated, sinInput));
    if (rotaryDims == shape[3]) {
      return out;
    }
    int[] tailStart = new int[4];
    tailStart[3] = rotaryDims;
    return MLXShape.concatenate(
        new MLXArray[] {out, MLXShape.slice(x, tailStart, shape.clone())}, 3);
  }
}
