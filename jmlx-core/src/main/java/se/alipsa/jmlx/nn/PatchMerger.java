package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * The Qwen3-VL vision patch merger, frozen for transformers 4.57.6 {@code Qwen3VLVisionPatchMerger}
 * (verbatim in req/plans/phase7-3b-reference-findings.md §11.1; provenance in {@code
 * tools/hf-reference/provenance.json}). The main merger ({@code use_postshuffle_norm=False})
 * normalizes the per-patch rows over {@code hiddenSize} BEFORE viewing them into merged rows; the
 * DeepStack mergers ({@code use_postshuffle_norm=True}) view into the merged rows FIRST and
 * normalize over the merged width {@code hiddenSize * merge^2}. Both then run {@code
 * fc2(GELU(fc1(x)))} with the EXACT (erf) GELU -- the reference's {@code nn.GELU()}, NOT the tanh
 * approximation the vision blocks' MLP uses (findings §11.2 "Merger ordering").
 *
 * <p>The normalization {@code eps} is hardwired to {@code 1e-6} exactly as in the 4.57.6 source (an
 * {@code nn.LayerNorm(..., eps=1e-6)} literal, not a config field -- the same hardwiring the block
 * norms use), so the constructor takes no {@code eps}.
 */
public final class PatchMerger extends Module {

  /** The reference's hardcoded LayerNorm eps (4.57.6 literal, not a config field). */
  private static final float EPS = 1e-6f;

  private final int spatialMergeSize;
  private final boolean usePostShuffleNorm;
  private final int hiddenSize;
  private final int outHiddenSize;
  private final LayerNorm norm;
  private final Linear linearFc1;
  private final GELU actFn;
  private final Linear linearFc2;

  /**
   * Registers the {@code norm}, {@code linearFc1}, {@code actFn} and {@code linearFc2} children in
   * that order. All validation is {@link IllegalArgumentException} and runs before any child is
   * registered: {@code spatialMergeSize} is positive; {@code fc1Weight} is square {@code [m, m]}
   * (the reference's {@code hidden_size = config.hidden_size * merge^2}, i.e. the merged width);
   * {@code m % merge^2 == 0} with {@code hiddenSize = m / merge^2}; {@code fc1Bias} has length
   * {@code m}; the norm weight/bias have length exactly {@code usePostShuffleNorm ? m : hiddenSize}
   * (the two placements' defining difference); and {@code fc2Weight} is {@code [outHidden, m]} with
   * a matching {@code fc2Bias} (the offending array is named in each message).
   */
  public PatchMerger(
      MLXScope scope,
      int spatialMergeSize,
      boolean usePostShuffleNorm,
      MLXArray normWeight,
      MLXArray normBias,
      MLXArray fc1Weight,
      MLXArray fc1Bias,
      MLXArray fc2Weight,
      MLXArray fc2Bias) {
    super(scope);
    if (spatialMergeSize < 1) {
      throw new IllegalArgumentException("spatialMergeSize must be positive");
    }
    int mergeSq = Math.multiplyExact(spatialMergeSize, spatialMergeSize);
    if (fc1Weight == null) {
      throw new IllegalArgumentException("fc1Weight must not be null");
    }
    if (fc1Weight.ndim() != 2 || fc1Weight.shape()[0] != fc1Weight.shape()[1]) {
      throw new IllegalArgumentException(
          "fc1Weight must be square [m, m], got shape " + Arrays.toString(fc1Weight.shape()));
    }
    int m = fc1Weight.shape()[0];
    if (m % mergeSq != 0) {
      throw new IllegalArgumentException(
          "fc1Width m = " + m + " must be divisible by merge^2 = " + mergeSq);
    }
    int hidden = m / mergeSq;
    int expectedNormDim = usePostShuffleNorm ? m : hidden;
    if (normWeight == null || normWeight.ndim() != 1 || normWeight.shape()[0] != expectedNormDim) {
      throw new IllegalArgumentException(
          "normWeight must be rank 1 with length "
              + expectedNormDim
              + " (usePostShuffleNorm="
              + usePostShuffleNorm
              + "), got shape "
              + (normWeight == null ? "null" : Arrays.toString(normWeight.shape())));
    }
    if (normBias == null || normBias.ndim() != 1 || normBias.shape()[0] != expectedNormDim) {
      throw new IllegalArgumentException(
          "normBias must be rank 1 with length "
              + expectedNormDim
              + " (usePostShuffleNorm="
              + usePostShuffleNorm
              + "), got shape "
              + (normBias == null ? "null" : Arrays.toString(normBias.shape())));
    }
    if (fc1Bias == null || fc1Bias.ndim() != 1 || fc1Bias.shape()[0] != m) {
      throw new IllegalArgumentException(
          "fc1Bias must be rank 1 with length m = "
              + m
              + ", got shape "
              + (fc1Bias == null ? "null" : Arrays.toString(fc1Bias.shape())));
    }
    if (fc2Weight == null) {
      throw new IllegalArgumentException("fc2Weight must not be null");
    }
    if (fc2Weight.ndim() != 2 || fc2Weight.shape()[1] != m) {
      throw new IllegalArgumentException(
          "fc2Weight must be [outHidden, m] with m = "
              + m
              + ", got shape "
              + Arrays.toString(fc2Weight.shape()));
    }
    int outHidden = fc2Weight.shape()[0];
    if (fc2Bias == null || fc2Bias.ndim() != 1 || fc2Bias.shape()[0] != outHidden) {
      throw new IllegalArgumentException(
          "fc2Bias must be rank 1 with length outHidden = "
              + outHidden
              + ", got shape "
              + (fc2Bias == null ? "null" : Arrays.toString(fc2Bias.shape())));
    }
    this.spatialMergeSize = spatialMergeSize;
    this.usePostShuffleNorm = usePostShuffleNorm;
    this.hiddenSize = hidden;
    this.outHiddenSize = outHidden;
    this.norm = child("norm", new LayerNorm(scope, normWeight, normBias, EPS));
    this.linearFc1 = child("linearFc1", new Linear(scope, fc1Weight, fc1Bias));
    this.actFn = child("actFn", new GELU(scope));
    this.linearFc2 = child("linearFc2", new Linear(scope, fc2Weight, fc2Bias));
  }

  /** Returns the spatial merge size (the merged width is {@code hiddenSize * merge^2}). */
  public int spatialMergeSize() {
    return spatialMergeSize;
  }

  /**
   * Whether the norm runs over the merged rows (DeepStack) instead of the per-patch rows (main).
   */
  public boolean usePostShuffleNorm() {
    return usePostShuffleNorm;
  }

  /** Returns the per-patch hidden size {@code m / merge^2}. */
  public int hiddenSize() {
    return hiddenSize;
  }

  /** Returns the output width of {@code fc2}. */
  public int outHiddenSize() {
    return outHiddenSize;
  }

  /**
   * Merges one image's or block's rows. All validation is {@link IllegalArgumentException} and runs
   * before any graph node is built: {@code x} is non-null rank 2 {@code [N, hiddenSize]} (inexact
   * dtype) with {@code N % merge^2 == 0}. Then the frozen order: {@code y = x} or, for the
   * post-shuffle norm, {@code reshape(x, [N/merge^2, m])} first; {@code norm.forward(y)} over the
   * corresponding width; for the main placement, the post-norm rows are then reshaped to {@code
   * [N/merge^2, m]}; finally {@code linearFc2(actFn(linearFc1(y)))} to {@code [N/merge^2,
   * outHiddenSize]}.
   */
  public MLXArray forward(MLXArray x) {
    if (x == null) {
      throw new IllegalArgumentException("x must not be null");
    }
    int[] shape = x.shape();
    if (shape.length != 2) {
      throw new IllegalArgumentException(
          "forward requires [N, hiddenSize], got shape " + Arrays.toString(shape));
    }
    if (shape[1] != hiddenSize) {
      throw new IllegalArgumentException(
          "forward input must have hiddenSize = " + hiddenSize + ", got " + shape[1]);
    }
    int n = shape[0];
    int mergeSq = spatialMergeSize * spatialMergeSize;
    if (n % mergeSq != 0) {
      throw new IllegalArgumentException("N = " + n + " must be divisible by merge^2 = " + mergeSq);
    }
    int merged = n / mergeSq;
    int m = hiddenSize * mergeSq;
    MLXArray y = usePostShuffleNorm ? MLXShape.reshape(x, new int[] {merged, m}) : x;
    y = norm.forward(y);
    if (!usePostShuffleNorm) {
      y = MLXShape.reshape(y, new int[] {merged, m});
    }
    return linearFc2.forward(actFn.forward(linearFc1.forward(y)));
  }
}
