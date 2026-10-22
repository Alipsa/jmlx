package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Rotary bidirectional (noncausal) self-attention for the Qwen3-VL vision tower, frozen for
 * transformers 4.57.6 {@code Qwen3VLVisionAttention} (verbatim in
 * req/plans/phase7-3b-reference-findings.md §11.1; provenance in {@code
 * tools/hf-reference/provenance.json}). One instance is one vision block's {@code attn} submodule:
 * a fused {@code qkv = Linear(hidden -> 3*hidden, bias)} projection, the two-axis rotation of q and
 * k, and a {@code proj = Linear(hidden -> hidden, bias)} output projection, with the reference's
 * own scaling {@code head_dim ** -0.5}.
 *
 * <p>The {@link TwoAxisRope} is deliberately NOT held here: it is stateless, and the rotation
 * arrives per-image from the caller through {@link #forward}, mirroring how {@link
 * MultiAxisRotation} is passed into the text-decoder attentions. The constructor's {@code theta} is
 * validated and exposed so a caller can build the matching {@link TwoAxisRope}, but the module
 * itself never derives a table.
 *
 * <p>jmlx runs one image at a time, so this module issues exactly one unmasked noncausal {@link
 * MLXFast#scaledDotProductAttention(MLXArray, MLXArray, MLXArray, float, boolean, MLXArray,
 * MLXArray)} call per image ({@code causal=false, maskArr=null, sinks=null}) over that image's
 * rows, instead of the reference's packed rows split per image by {@code cu_seqlens}. The two are
 * numerically identical -- attention mixes nothing across images, and every other op is row-wise
 * (findings §11.2 "Per-image sequential execution") -- and looping bounds the concurrent attention
 * workspace to a single image.
 */
public final class TwoAxisRotaryAttention extends Module {

  private final int heads;
  private final int headDim;
  private final float theta;
  private final float scale;
  private final Linear qkv;
  private final Linear proj;

  /**
   * Registers the {@code qkv} and {@code proj} children. All validation is {@link
   * IllegalArgumentException} and runs before any child is registered: {@code heads} and {@code
   * headDim} are positive, {@code theta} is positive and finite (it must match the {@link
   * TwoAxisRope} the caller prepares rotations with), and the weight/bias shapes are exactly {@code
   * qkvWeight [3*hidden, hidden]}, {@code qkvBias [3*hidden]}, {@code projWeight [hidden, hidden]}
   * and {@code projBias [hidden]} where {@code hidden = heads * headDim} (computed in {@code
   * long}-safe exact math, so an overflowing geometry is an arithmetic failure, not a wrapped
   * shape). The {@code scale} is the reference's {@code head_dim ** -0.5}, derived, not passed.
   */
  public TwoAxisRotaryAttention(
      MLXScope scope,
      int heads,
      int headDim,
      float theta,
      MLXArray qkvWeight,
      MLXArray qkvBias,
      MLXArray projWeight,
      MLXArray projBias) {
    super(scope);
    if (heads <= 0) {
      throw new IllegalArgumentException("heads must be positive");
    }
    if (headDim <= 0) {
      throw new IllegalArgumentException("headDim must be positive");
    }
    if (!Float.isFinite(theta) || theta <= 0) {
      throw new IllegalArgumentException("theta must be positive and finite");
    }
    int hidden = Math.multiplyExact(heads, headDim);
    int triple = Math.multiplyExact(3, hidden);
    if (qkvWeight == null) {
      throw new IllegalArgumentException("qkvWeight must not be null");
    }
    if (qkvWeight.ndim() != 2 || qkvWeight.shape()[0] != triple || qkvWeight.shape()[1] != hidden) {
      throw new IllegalArgumentException(
          "qkvWeight must be [3*hidden, hidden] = ["
              + triple
              + ", "
              + hidden
              + "], got shape "
              + Arrays.toString(qkvWeight.shape()));
    }
    if (qkvBias == null || qkvBias.ndim() != 1 || qkvBias.shape()[0] != triple) {
      throw new IllegalArgumentException(
          "qkvBias must be rank 1 with length 3*hidden = "
              + triple
              + ", got shape "
              + (qkvBias == null ? "null" : Arrays.toString(qkvBias.shape())));
    }
    if (projWeight == null) {
      throw new IllegalArgumentException("projWeight must not be null");
    }
    if (projWeight.ndim() != 2
        || projWeight.shape()[0] != hidden
        || projWeight.shape()[1] != hidden) {
      throw new IllegalArgumentException(
          "projWeight must be [hidden, hidden] = ["
              + hidden
              + ", "
              + hidden
              + "], got shape "
              + Arrays.toString(projWeight.shape()));
    }
    if (projBias == null || projBias.ndim() != 1 || projBias.shape()[0] != hidden) {
      throw new IllegalArgumentException(
          "projBias must be rank 1 with length hidden = "
              + hidden
              + ", got shape "
              + (projBias == null ? "null" : Arrays.toString(projBias.shape())));
    }
    this.heads = heads;
    this.headDim = headDim;
    this.theta = theta;
    this.scale = (float) Math.pow(headDim, -0.5);
    this.qkv = child("qkv", new Linear(scope, qkvWeight, qkvBias));
    this.proj = child("proj", new Linear(scope, projWeight, projBias));
  }

  /** Returns the attention scaling {@code head_dim ** -0.5}. */
  public float scale() {
    return scale;
  }

  /** Returns the frequency base the prepared rotations must match. */
  public float theta() {
    return theta;
  }

  /**
   * Runs one image's full attention. All validation is {@link IllegalArgumentException} and runs
   * before any graph node is built: {@code input} is non-null rank 2 {@code [N, hidden]} (the
   * image's patch rows, hidden = heads * headDim), {@code rotation} is non-null, and {@link
   * TwoAxisRotation#validate} passes for the row count and head dimension (a rotation prepared for
   * a different image size or a different rotary width is rejected before any projection runs).
   * Then the frozen op sequence: {@code qkv.forward(input)} to {@code [N, 3*hidden]}; reshape to
   * {@code [N, 3, heads, headDim]} and transpose {@code {1,0,2,3}} to {@code [3, N, heads,
   * headDim]}; split into q, k, v on axis 0 (the split keeps the axis, so each part's size-1 axis 0
   * is squeezed back to {@code [N, heads, headDim]}); {@code rotation.rotate} on q and k (v is
   * unrotated, as in the reference); each of the three transposed to {@code [heads, N, headDim]}
   * and reshaped to {@code [1, heads, N, headDim]}; one unmasked noncausal {@link
   * MLXFast#scaledDotProductAttention} with the module's {@link #scale()}; the result reshaped back
   * to {@code [N, hidden]} and passed through {@code proj}.
   */
  public MLXArray forward(MLXArray input, TwoAxisRotation rotation) {
    if (input == null) {
      throw new IllegalArgumentException("input must not be null");
    }
    if (rotation == null) {
      throw new IllegalArgumentException("rotation must not be null");
    }
    int[] shape = input.shape();
    if (shape.length != 2) {
      throw new IllegalArgumentException(
          "forward requires [N, hidden], got shape " + Arrays.toString(shape));
    }
    int hidden = heads * headDim;
    if (shape[1] != hidden) {
      throw new IllegalArgumentException(
          "forward input must have hidden = heads * headDim = " + hidden + ", got " + shape[1]);
    }
    int n = shape[0];
    rotation.validate(n, headDim);
    MLXArray projected =
        MLXShape.transpose(
            MLXShape.reshape(qkv.forward(input), new int[] {n, 3, heads, headDim}),
            new int[] {1, 0, 2, 3});
    MLXArray[] parts = MLXShape.split(projected, 3, 0);
    MLXArray q = headBatch(rotation.rotate(MLXShape.squeeze(parts[0], new int[] {0})), n);
    MLXArray k = headBatch(rotation.rotate(MLXShape.squeeze(parts[1], new int[] {0})), n);
    MLXArray v = headBatch(MLXShape.squeeze(parts[2], new int[] {0}), n);
    MLXArray attended = MLXFast.scaledDotProductAttention(q, k, v, scale, false, null, null);
    MLXArray merged =
        MLXShape.reshape(
            MLXShape.transpose(
                MLXShape.reshape(attended, new int[] {heads, n, headDim}), new int[] {1, 0, 2}),
            new int[] {n, hidden});
    return proj.forward(merged);
  }

  /**
   * The {@code [heads, N, headDim]} single-batch layout {@code scaledDotProductAttention} wants.
   */
  private MLXArray headBatch(MLXArray x, int n) {
    return MLXShape.reshape(
        MLXShape.transpose(x, new int[] {1, 0, 2}), new int[] {1, heads, n, headDim});
  }
}
