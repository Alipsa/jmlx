package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Immutable multi-axis (MRoPE) rotary-position specification for Hugging Face's interleaved layout
 * ({@code mrope_interleaved: true}, {@code rope_type: default}), frozen for the Qwen3-VL text
 * decoder in req/plans/phase7-3b-reference-findings.md §6.9. This is a dedicated component, not a
 * new {@link RopeSpec} variant: the sealed {@code RopeSpec} hierarchy gains no permitted variant,
 * and the ordinary decoders keep the unchanged {@code mlx_fast_rope} fast kernels via {@link
 * RopeSpec#apply}. The fast kernels' offset-plus-arange contract cannot express unequal multimodal
 * coordinates, which is why the Qwen3-VL path composes this component from existing core facades
 * instead: no new FFI call.
 *
 * <p>Deliberately not a {@link Module}: no parameters, no children, no owned scope. Every MLX array
 * {@link #prepare} builds allocates into the caller's scope, and every array {@link
 * MultiAxisRotation#rotate} builds allocates per the usual {@code scopeOf} rule (the innermost
 * related scope of its operands). Positions and frequencies are recomputed on every {@code prepare}
 * call: no position-derived state is cached.
 *
 * <p>{@code positions} is {@code int[3][batch * sequence]}: three rows of length {@code B*T}
 * (row-major flattened {@code [B,T]}) where row 0 is the T (time) axis, row 1 the H (height) axis
 * and row 2 the W (width) axis -- the reference's {@code [3, B*T]} {@code position_ids} layout as
 * one validated object, in the {@code mrope_section} order. Coordinates are non-negative (decode
 * steps pass {@code cachePosition + delta}, where the delta can be negative but the coordinate it
 * produces must not be).
 */
public final class MultiAxisRope {

  private final float theta;
  private final int[] mropeSection;
  private final int rotaryDims;
  private final float[] invFreq;
  private final float[] ownerT;
  private final float[] ownerH;
  private final float[] ownerW;

  /**
   * Validates the base and the sections and derives the per-frequency inverse and the static T/H/W
   * owner masks. All validation is {@link IllegalArgumentException} and runs before any MLX array
   * is written: {@code theta} must be finite and positive (the {@code RopeSpec} convention); {@code
   * mropeSection} must be a non-null 3-entry array, every element non-negative, with a sum of at
   * least 1 (so {@code rotaryDims = 2 * sum} is even by construction) and small enough to fit an
   * {@code int} rotary dimension.
   */
  public MultiAxisRope(float theta, int[] mropeSection) {
    if (!Float.isFinite(theta) || theta <= 0) {
      throw new IllegalArgumentException("theta must be positive and finite");
    }
    if (mropeSection == null) {
      throw new IllegalArgumentException("mropeSection must not be null");
    }
    if (mropeSection.length != 3) {
      throw new IllegalArgumentException("mropeSection must have exactly 3 entries (T, H, W)");
    }
    long sum = 0;
    for (int i = 0; i < 3; i++) {
      if (mropeSection[i] < 0) {
        throw new IllegalArgumentException("mropeSection[" + i + "] must be non-negative");
      }
      sum += mropeSection[i];
    }
    if (sum < 1) {
      throw new IllegalArgumentException("mropeSection sum must be at least 1");
    }
    if (sum > Integer.MAX_VALUE / 2) {
      throw new IllegalArgumentException("mropeSection sum must fit an int rotary dimension");
    }
    int half = (int) sum;
    this.theta = theta;
    this.mropeSection = mropeSection.clone();
    this.rotaryDims = 2 * half;
    this.invFreq = deriveInvFrequencies(theta, half);
    this.ownerT = new float[half];
    this.ownerH = new float[half];
    this.ownerW = new float[half];
    for (int f = 0; f < half; f++) {
      // The pinned reference (transformers 4.57.6 apply_interleaved_mrope) starts from the T row
      // and overwrites in place: for dim 1 (H) idx = slice(1, mrope_section[1] * 3, 3) -- H owns
      // 1 + 3j for j < mrope_section[1]; for dim 2 (W) idx = slice(2, mrope_section[2] * 3, 3) --
      // W owns 2 + 3j for j < mrope_section[2]; everything else stays T. The masks realize that
      // selection as a static 0/1 weight vector per axis.
      if (f % 3 == 1 && f < 1L + 3L * mropeSection[1]) {
        ownerH[f] = 1f;
      } else if (f % 3 == 2 && f < 2L + 3L * mropeSection[2]) {
        ownerW[f] = 1f;
      } else {
        ownerT[f] = 1f;
      }
    }
  }

  /** Returns the frequency base {@code theta}. */
  public float theta() {
    return theta;
  }

  /** Returns a defensive copy of the mrope sections (T, H, W). */
  public int[] mropeSection() {
    return mropeSection.clone();
  }

  /** Returns {@code 2 * sum(mropeSection)}, the width of the rotated head prefix. */
  public int rotaryDims() {
    return rotaryDims;
  }

  /**
   * Builds the per-step {@link MultiAxisRotation} for {@code batch} rows of {@code sequence}
   * tokens. All validation is {@link IllegalArgumentException} and runs before any MLX array is
   * written: {@code batch} and {@code sequence} are at least 1, {@code positions} has exactly 3
   * non-null rows, each row has length {@code batch * sequence} (computed in {@code long} so the
   * product cannot overflow) and no coordinate is negative. The frozen compute order then runs: the
   * static owner masks select, per (token, frequency) on the CPU before any array is written, the
   * owned row's position, giving one float32 position matrix of shape {@code [B*T, rotaryDims/2]}
   * (the exact int-to-float casts) that is broadcast-multiplied once against the {@code [1,
   * rotaryDims/2]} {@code invFreq} array in {@code scope} -- a single float32 multiply per element
   * with the same two operands the frozen mask expression {@code freqs_t = freqsT * wT + freqsH *
   * wH + freqsW * wW} copies, which is a bit-identical copy for non-negative finite frequencies
   * (findings §6.9, step 2); {@code emb = cat((freqs_t, freqs_t), -1)}; {@code MLXOps.cos}/{@code
   * sin} in float32, with the attention scaling of 1.0 (the pinned {@code rope_type: default})
   * omitted as the identity on finite values; finally the two arrays are reshaped to {@code [batch,
   * 1, sequence, rotaryDims]}, broadcastable over the heads axis.
   */
  public MultiAxisRotation prepare(MLXScope scope, int batch, int sequence, int[][] positions) {
    if (batch < 1) {
      throw new IllegalArgumentException("batch must be at least 1");
    }
    if (sequence < 1) {
      throw new IllegalArgumentException("sequence must be at least 1");
    }
    if (positions == null) {
      throw new IllegalArgumentException("positions must not be null");
    }
    if (positions.length != 3) {
      throw new IllegalArgumentException("positions must have exactly 3 rows (T, H, W)");
    }
    long tokens = (long) batch * sequence;
    for (int axis = 0; axis < 3; axis++) {
      if (positions[axis] == null) {
        throw new IllegalArgumentException("positions[" + axis + "] must not be null");
      }
      if ((long) positions[axis].length != tokens) {
        throw new IllegalArgumentException(
            "positions[" + axis + "] must have length batch * sequence = " + tokens);
      }
    }
    for (int axis = 0; axis < 3; axis++) {
      for (int coordinate : positions[axis]) {
        if (coordinate < 0) {
          throw new IllegalArgumentException("positions must be non-negative");
        }
      }
    }
    int flat = (int) tokens;
    int half = rotaryDims / 2;
    MLXArray invFreqArray = MLX.array(scope, invFreq, new int[] {1, half});
    MLXArray owned =
        MLX.array(scope, ownedPositions(positions, flat, half), new int[] {flat, half});
    MLXArray freqs = MLXOps.multiply(owned, invFreqArray);
    MLXArray emb = MLXShape.concatenate(new MLXArray[] {freqs, freqs}, 1);
    MLXArray cos = MLXShape.reshape(MLXOps.cos(emb), new int[] {batch, 1, sequence, rotaryDims});
    MLXArray sin = MLXShape.reshape(MLXOps.sin(emb), new int[] {batch, 1, sequence, rotaryDims});
    return new MultiAxisRotation(cos, sin, rotaryDims);
  }

  /**
   * The per-token owned-position matrix in row-major {@code [flat, half]} order: element {@code (i,
   * f)} is the exact float32 cast of {@code positions[owner(f)][i]}, selected on the CPU with the
   * static owner masks before any array is written. This is the bit-identical copy the frozen mask
   * expression {@code freqsT * wT + freqsH * wH + freqsW * wW} makes for non-negative finite
   * frequencies (multiplying by the 0/1 masks and adding in float32 is exact there), so multiplying
   * this matrix once by {@code invFreq} performs the same per-element float32 multiply of the same
   * two operands as the frozen per-axis broadcast-multiplies (findings §6.9, step 2).
   */
  private float[] ownedPositions(int[][] positions, int flat, int half) {
    float[] out = new float[flat * half];
    for (int i = 0; i < flat; i++) {
      for (int f = 0; f < half; f++) {
        int coordinate =
            ownerT[f] != 0f ? positions[0][i] : ownerH[f] != 0f ? positions[1][i] : positions[2][i];
        out[i * half + f] = (float) coordinate;
      }
    }
    return out;
  }

  /**
   * Derives {@code invFreq[i] = 1 / theta^(2i/rotaryDims)} once, in the pinned reference's float32
   * arithmetic order: the exponent is the float32 division of the exact float32 numerator {@code
   * 2i} by the float32 {@code rotaryDims}, the power is a float32 cast of the double power, and the
   * reciprocal is a float32 division. The pinned transformers 4.57.6 {@code
   * _compute_default_rope_parameters} builds exactly {@code 1.0 / (base ** (arange(0, head_dim,
   * 2).float() / head_dim))} in float32, and this order reproduces it bit for bit on 63 of the 64
   * pinned 2B frequencies (theta 5,000,000, head_dim 128; the remaining one within 1e-9). The
   * RopeSpec-style alternative -- double power, then cast -- deviates from the pinned reference by
   * about 1e-5 in the rotate-half output already at position 89, inside the pinned 2B prefill
   * envelope, and cannot meet the frozen 1e-5 cross-check threshold, so the float32 order stands;
   * the pinned-venv cross-check (tools/hf-reference/qwen3vl_rope_interleaved_check) is the arbiter
   * for this subtlety (findings §6.9).
   */
  private static float[] deriveInvFrequencies(float theta, int half) {
    int rotaryDims = 2 * half;
    float[] invFreq = new float[half];
    for (int i = 0; i < half; i++) {
      float exponent = (float) (2 * i) / (float) rotaryDims;
      invFreq[i] = 1.0f / (float) Math.pow(theta, exponent);
    }
    return invFreq;
  }
}
