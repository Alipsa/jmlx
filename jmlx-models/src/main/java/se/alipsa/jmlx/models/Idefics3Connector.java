package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;

/**
 * The Idefics3 image-feature connector: the {@code pixel_shuffle} token reduction followed by the
 * bias-free {@code modality_projection}, ported from the pinned Hugging Face Transformers 4.57.6
 * {@code modeling_idefics3.py} (see {@code tools/hf-reference/provenance.json}) — {@code
 * Idefics3Connector} (lines 390-410) and {@code Idefics3SimpleMLP} (274-282).
 *
 * <p>The child {@code modalityProjection} is the checkpoint's {@code
 * connector.modality_projection.proj}. Note that HF's {@code pixel_shuffle} is a token
 * <em>reduction</em> — {@code P x P} tokens of {@code d} channels become {@code (P/s) x (P/s)}
 * tokens of {@code d * s^2} channels for scale {@code s} — not the inverse expansion that
 * rearrange-style "pixel shuffle" formulas usually describe; the gather permutation here is the one
 * derived from the pinned view/permute chain and verified bit-exact against it.
 */
public final class Idefics3Connector extends Module {

  private final int scaleFactor;
  private final Linear modalityProjection;

  /**
   * Builds the connector around the bias-free modality projection.
   *
   * @param scope the scope the parameter is allocated into
   * @param scaleFactor the pixel-shuffle scale {@code s} (the checkpoint's {@code scale_factor})
   * @param projectionWeight the {@code [textHidden, visionHidden * s^2]} projection weight
   * @throws IllegalArgumentException if the scale factor is not positive or the weight is not a
   *     rank-2 matrix
   */
  public Idefics3Connector(MLXScope scope, int scaleFactor, MLXArray projectionWeight) {
    super(scope);
    Objects.requireNonNull(projectionWeight, "projectionWeight");
    if (scaleFactor <= 0) {
      throw new IllegalArgumentException("the scale factor must be positive");
    }
    if (projectionWeight.ndim() != 2) {
      throw new IllegalArgumentException(
          "the modality projection weight must be rank 2 [out, in], got shape "
              + Arrays.toString(projectionWeight.shape()));
    }
    this.scaleFactor = scaleFactor;
    this.modalityProjection =
        child("modalityProjection", new Linear(scope, projectionWeight, null));
    train(false);
  }

  /**
   * Runs the pixel-shuffle reduction on the tower's {@code [N, seq, d]} output (with {@code seq} a
   * perfect square token grid and {@code d * s^2} the projection's input width), then the bias-free
   * projection.
   *
   * <p>Every activation is allocated into {@code x.scope()}, so that scope must be this module's
   * own scope or a descendant of it — the same contract as {@link Idefics3VisionTower#forward}.
   *
   * @return the projected features, {@code [N, seq / s^2, textHidden]}
   * @throws IllegalArgumentException if the input shape is not a square token grid compatible with
   *     the scale factor
   */
  public MLXArray forward(MLXArray x) {
    Objects.requireNonNull(x, "x");
    int[] shape = x.shape();
    if (shape.length != 3) {
      throw new IllegalArgumentException(
          "the connector input must be [N, seq, d], got " + Arrays.toString(shape));
    }
    int seq = shape[1];
    int side = (int) Math.sqrt(seq);
    if (side * side != seq) {
      throw new IllegalArgumentException(
          "the token grid of "
              + seq
              + " tokens is not square; the reference pixel shuffle "
              + "assumes a square grid");
    }
    int channels = shape[2];
    int[] permutation = pixelShuffleIndices(side, side, scaleFactor, channels);
    MLXScope step = x.scope();
    // The permutation is flat over the token-channel rows, so gather per row on the flattened
    // [N, seq*d] view: takeAxis inserts the indices' shape, giving [N, 1, seq*d].
    int flatLength = Math.multiplyExact(seq, channels);
    MLXArray flat = MLXShape.reshape(x, new int[] {shape[0], flatLength});
    MLXArray gathered =
        MLXShape.takeAxis(flat, MLX.array(step, permutation, new int[] {1, flatLength}), 1);
    int scaleSquared = Math.multiplyExact(scaleFactor, scaleFactor);
    MLXArray reduced =
        MLXShape.reshape(
            gathered, new int[] {shape[0], seq / scaleSquared, channels * scaleSquared});
    return modalityProjection.forward(reduced);
  }

  /**
   * The flat gather permutation of the pinned {@code Idefics3Connector.pixel_shuffle} view/permute
   * chain (lines 396-405), generalized to a rectangular {@code ph x pw} token grid: {@code
   * perm[outIndex] = inputIndex} over the {@code ph * pw * channels} elements viewed as tokens of
   * {@code channels} channels. Token {@code (blockRow * s + subRow, blockCol * s + subCol)}'s
   * channel block {@code subRow * (d * s) + subCol * d + k} of the output block {@code (blockRow,
   * blockCol)} lands at output element {@code (blockRow * (pw/s) + blockCol) * (d * s^2) + subRow *
   * (d * s) + subCol * d + k}.
   *
   * <p>For the pinned fixture geometry ({@code ph = pw = 4}, {@code s = 2}, {@code d = 64}) the
   * permutation is a true permutation of {@code 0..1023} and reproduces the reference's flat output
   * entry for entry, verified against the reference venv.
   *
   * @param ph the token-grid rows
   * @param pw the token-grid columns
   * @param scale the pixel-shuffle scale {@code s}
   * @param channels the per-token width {@code d}
   * @return the {@code ph * pw * channels} gather indices
   * @throws IllegalArgumentException if a dimension is not positive or a divisibility constraint is
   *     violated
   */
  public static int[] pixelShuffleIndices(int ph, int pw, int scale, int channels) {
    if (ph <= 0 || pw <= 0 || scale <= 0 || channels <= 0) {
      throw new IllegalArgumentException("all pixel-shuffle dimensions must be positive");
    }
    if (ph % scale != 0) {
      throw new IllegalArgumentException(
          "the token-grid rows " + ph + " are not a multiple of the scale " + scale);
    }
    if (pw % scale != 0) {
      throw new IllegalArgumentException(
          "the token-grid columns " + pw + " are not a multiple of the scale " + scale);
    }
    int scaleSquared = Math.multiplyExact(scale, scale);
    if (channels % scaleSquared != 0) {
      throw new IllegalArgumentException(
          "the channels " + channels + " are not a multiple of scale^2 = " + scaleSquared);
    }
    int block = Math.multiplyExact(channels, scaleSquared);
    int[] perm = new int[Math.multiplyExact(Math.multiplyExact(ph, pw), channels)];
    for (int blockRow = 0; blockRow < ph / scale; blockRow++) {
      for (int blockCol = 0; blockCol < pw / scale; blockCol++) {
        for (int n = 0; n < block; n++) {
          int subRow = n / (channels * scale);
          int subCol = (n % (channels * scale)) / channels;
          int k = n % channels;
          perm[(blockRow * (pw / scale) + blockCol) * block + n] =
              ((blockRow * scale + subRow) * pw + (blockCol * scale + subCol)) * channels + k;
        }
      }
    }
    return perm;
  }
}
