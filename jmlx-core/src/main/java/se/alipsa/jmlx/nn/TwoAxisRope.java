package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Immutable two-axis (vision) rotary-position specification for Hugging Face's Qwen3-VL vision
 * tower, frozen for transformers 4.57.6 {@code Qwen3VLVisionRotaryEmbedding} (provenance in {@code
 * tools/hf-reference/provenance.json}; the verbatim transcription and the design decisions are
 * req/plans/phase7-3b-reference-findings.md §11). The vision tower rotates queries and keys with
 * the patch's actual (row, col) grid coordinates: one axis per spatial direction, each owning
 * {@code rotaryDims / 2} frequencies, in the reference's {@code dim = head_dim // 2} geometry (16
 * frequencies per axis for the 2B's head_dim 64).
 *
 * <p>Deliberately not a {@link Module}: no parameters, no children, no owned scope. Every MLX array
 * {@link #prepare} builds allocates into the caller's scope, and every array {@link
 * TwoAxisRotation#rotate} builds allocates per the usual {@code scopeOf} rule (the innermost
 * related scope of its operands). The rotation arrives per-image from the caller, mirroring how
 * {@link MultiAxisRotation} is handed into the text-decoder attentions.
 *
 * <p>{@code rotaryDims} is the per-row frequency width -- the reference's {@code dim}, i.e. {@code
 * head_dim // 2} -- and must be even (each axis owns {@code rotaryDims / 2} frequencies) and at
 * least 2 (at least one frequency per axis). The full head dimension a rotation can be applied to
 * is {@code 2 * rotaryDims}: {@code prepare} doubles it via {@code emb = cat((freqs, freqs), -1)}
 * exactly as the reference's model {@code forward} does, so the two halves of a head share
 * identical cos/sin and rotate as two independent {@code rotaryDims}-wide half-rotations -- the
 * first on the row frequency, the second on the col frequency (findings §11.2 "Vision RoPE").
 */
public final class TwoAxisRope {

  private final float theta;
  private final int rotaryDims;
  private final float[] invFreq;

  /**
   * Validates {@code theta} and {@code rotaryDims} and derives the per-frequency inverse
   * frequencies. All validation is {@link IllegalArgumentException} and runs before any MLX array
   * is written: {@code theta} must be finite and positive (the {@code RopeSpec} convention); {@code
   * rotaryDims} must be even and at least 2.
   */
  public TwoAxisRope(float theta, int rotaryDims) {
    if (!Float.isFinite(theta) || theta <= 0) {
      throw new IllegalArgumentException("theta must be positive and finite");
    }
    if (rotaryDims < 2 || rotaryDims % 2 != 0) {
      throw new IllegalArgumentException(
          "rotaryDims must be even and at least 2, got " + rotaryDims);
    }
    this.theta = theta;
    this.rotaryDims = rotaryDims;
    this.invFreq = deriveInvFrequencies(theta, rotaryDims);
  }

  /** Returns the frequency base {@code theta}. */
  public float theta() {
    return theta;
  }

  /** Returns the per-row frequency width (the reference's {@code dim}, {@code head_dim // 2}). */
  public int rotaryDims() {
    return rotaryDims;
  }

  /**
   * The derived float32 inverse frequencies, {@code invFreq[j] = 1 / theta^(2j/rotaryDims)} for
   * {@code j} in {@code [0, rotaryDims/2)}. Package-private so the same-package bit-exact test
   * ({@code TwoAxisRopeTest}) can compare every value against the pinned torch bits; not part of
   * the public API.
   */
  float[] invFreq() {
    return invFreq.clone();
  }

  /**
   * Builds the per-image {@link TwoAxisRotation} for {@code sequence} patch rows. All validation is
   * {@link IllegalArgumentException} and runs before any MLX array is written: {@code sequence} is
   * at least 1, {@code axis0}/{@code axis1} are non-null with length exactly {@code sequence} and
   * non-negative values (they are the patches' (row, col) grid coordinates in merge-block order, as
   * the tower's {@code rotaryCoordinates} helper produces them), and the doubled table size {@code
   * sequence * 2 * rotaryDims} fits an {@code int} array (checked in {@code long}). The reference
   * compute order then runs: the per-row frequency matrix is the float32 outer products {@code
   * [outer(axis0, invFreq) | outer(axis1, invFreq)]} of shape {@code [sequence, rotaryDims]} (each
   * axis value is exactly int-to-float-cast on the CPU first, then one float32 multiply per element
   * -- the same two operands torch's {@code torch.outer} multiplies, bit-verified against the
   * pinned golden's {@code cos_sin_sample} products); the rows are doubled to {@code emb =
   * cat((freqs, freqs), -1)} of shape {@code [sequence, 2 * rotaryDims]}; {@code MLXOps.cos}/{@code
   * sin} in float32; finally both tables are reshaped to {@code [sequence, 1, 2 * rotaryDims]} so
   * they broadcast over the heads axis of a {@code [sequence, heads, 2 * rotaryDims]} projection.
   */
  public TwoAxisRotation prepare(MLXScope scope, int sequence, int[] axis0, int[] axis1) {
    if (sequence < 1) {
      throw new IllegalArgumentException("sequence must be at least 1");
    }
    if (axis0 == null) {
      throw new IllegalArgumentException("axis0 must not be null");
    }
    if (axis1 == null) {
      throw new IllegalArgumentException("axis1 must not be null");
    }
    if (axis0.length != sequence) {
      throw new IllegalArgumentException(
          "axis0 must have length sequence = " + sequence + ", got " + axis0.length);
    }
    if (axis1.length != sequence) {
      throw new IllegalArgumentException(
          "axis1 must have length sequence = " + sequence + ", got " + axis1.length);
    }
    for (int coordinate : axis0) {
      if (coordinate < 0) {
        throw new IllegalArgumentException("axis0 must be non-negative");
      }
    }
    for (int coordinate : axis1) {
      if (coordinate < 0) {
        throw new IllegalArgumentException("axis1 must be non-negative");
      }
    }
    int doubled = rotaryDims * 2;
    if ((long) sequence * doubled > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("sequence * 2 * rotaryDims must fit an int array");
    }
    int half = rotaryDims / 2;
    MLXArray freqs =
        MLX.array(scope, outerRows(axis0, axis1, sequence, half), new int[] {sequence, rotaryDims});
    MLXArray emb = MLXShape.concatenate(new MLXArray[] {freqs, freqs}, 1);
    MLXArray cos = MLXShape.reshape(MLXOps.cos(emb), new int[] {sequence, 1, doubled});
    MLXArray sin = MLXShape.reshape(MLXOps.sin(emb), new int[] {sequence, 1, doubled});
    return new TwoAxisRotation(cos, sin, rotaryDims);
  }

  /**
   * The per-row frequency matrix in row-major {@code [sequence, rotaryDims]} order: row {@code i}
   * is {@code [i*axis0 * invFreq (half) | i*axis1 * invFreq (half)]}, each element the float32
   * multiply of the exact int-to-float cast of the axis coordinate by the derived {@code invFreq}
   * -- the same two float32 operands the reference's {@code torch.outer(arange, inv_freq)} lookup
   * multiplies per element.
   */
  private float[] outerRows(int[] axis0, int[] axis1, int sequence, int half) {
    float[] out = new float[sequence * (half * 2)];
    for (int i = 0; i < sequence; i++) {
      float row = (float) axis0[i];
      float col = (float) axis1[i];
      for (int j = 0; j < half; j++) {
        out[i * (half * 2) + j] = row * invFreq[j];
        out[i * (half * 2) + half + j] = col * invFreq[j];
      }
    }
    return out;
  }

  /**
   * Derives {@code invFreq[j] = 1 / theta^(2j/rotaryDims)} once, in the pinned reference's float32
   * arithmetic order: the exponent is the float32 division of the exact float32 numerator {@code
   * 2j} by the float32 {@code rotaryDims}, the power is a float32 cast of the double power, and the
   * reciprocal is a float32 division. The pinned transformers 4.57.6 {@code
   * Qwen3VLVisionRotaryEmbedding} builds exactly {@code 1.0 / (theta ** (arange(0, dim, 2, float32)
   * / dim))} in float32, and this order reproduces all 16 of the pinned 2B frequencies' bits (theta
   * 10000, dim 32; the WP2 golden's {@code probes.inv_freq} hex bits, asserted bit-for-bit in
   * {@code TwoAxisRopeTest}).
   */
  private static float[] deriveInvFrequencies(float theta, int rotaryDims) {
    int half = rotaryDims / 2;
    float[] invFreq = new float[half];
    for (int j = 0; j < half; j++) {
      float exponent = (2f * j) / (float) rotaryDims;
      float v = (float) Math.pow(theta, (double) exponent);
      invFreq[j] = 1.0f / v;
    }
    return invFreq;
  }
}
