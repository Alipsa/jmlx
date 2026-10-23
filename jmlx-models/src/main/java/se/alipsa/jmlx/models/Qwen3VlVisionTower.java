package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Conv3d;
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.GELU;
import se.alipsa.jmlx.nn.LayerNorm;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.PatchMerger;
import se.alipsa.jmlx.nn.Sequential;
import se.alipsa.jmlx.nn.TwoAxisRope;
import se.alipsa.jmlx.nn.TwoAxisRotaryAttention;
import se.alipsa.jmlx.nn.TwoAxisRotation;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessorResult;

/**
 * The Qwen3-VL vision tower: the patch-embedding convolution, the bilinear-interpolated position
 * embedding, the two-axis-RoPE pre-norm encoder stack, the main merger and the DeepStack mergers of
 * {@code Qwen3VLVisionModel}, ported from the pinned Hugging Face Transformers 4.57.6 {@code
 * modeling_qwen3_vl.py} (see {@code tools/hf-reference/provenance.json}) — {@code
 * Qwen3VLVisionModel} (including {@code rot_pos_emb} and {@code fast_pos_embed_interpolate}),
 * {@code Qwen3VLVisionBlock}, {@code Qwen3VLVisionAttention}, {@code Qwen3VLVisionMLP}, {@code
 * Qwen3VLVisionPatchMerger}, {@code Qwen3VLVisionPatchEmbed} and {@code
 * Qwen3VLVisionRotaryEmbedding}; the verbatim transcription is findings §11.1. The reference runs
 * in float32 (its default dtype), so every weight is converted to float32 by this constructor; on
 * float32 inputs, {@link #forward(MLXScope, List)} matches the reference's op order within the
 * golden test's 1e-4 tolerance (measured max diff 3.1e-6) — MLX's GPU kernels (matmul, SDPA) are
 * not bit-identical to torch on the CPU.
 *
 * <p>The module child names mirror the HF checkpoint tree through the mapping of {@link
 * #visionKeyToTowerKey(String, int, int[])}: {@code patchEmbedding} is {@code patch_embed.proj}
 * (the weight transposed to jmlx's channels-last {@code [hidden, temporal, height, width,
 * inChannels]} layout by {@link #visionWeights(Map, int, int[])}), {@code positionEmbedding} is
 * {@code pos_embed}, {@code layer0..layerN-1} are {@code blocks.{i}} (each a {@link
 * Qwen3VlVisionBlock} with {@code norm1}/{@code norm2} for {@code norm1/norm2}, {@code
 * attention.qkv}/{@code attention.proj} for {@code attn.qkv/attn.proj} and {@code mlp.0}/{@code
 * mlp.2} for {@code mlp.linear_fc1/linear_fc2}, the tanh-approximation GELU {@code mlp.1} carrying
 * no parameters — the reference's {@code hidden_act = gelu_pytorch_tanh}), {@code merger} is {@code
 * merger} (post-shuffle norm <em>off</em>, norm width {@code hidden}) and {@code
 * deepstackMerger0..} are {@code deepstack_merger_list.{j}} (post- shuffle norm <em>on</em>, norm
 * width {@code hidden * merge^2}). {@link #forward(MLXScope, List)} stops at the mergers; it
 * returns the main-merger rows and, per DeepStack index, that level's rows.
 *
 * <p>Two layout facts worth knowing before reasoning about the math:
 *
 * <ul>
 *   <li>The reference convolves channels-first pixel batches; MLX's convolutions are channels-last,
 *       so both the conv weight (in {@link #visionWeights(Map, int, int[])}) and the per-image
 *       pixel tensor (in {@link #forward(MLXScope, List)}) are transposed {@code {0, 2, 3, 4, 1}}
 *       before the convolution.
 *   <li>The reference builds the bilinear position-embedding rows in per-image row-major patch
 *       order, sums the four weighted corner rows, and only then permutes into merge-block order;
 *       {@link #forward(MLXScope, List)} gathers the corners directly in merge-block order instead.
 *       The sum is per-row, so the two orderings produce identical bits.
 * </ul>
 *
 * <p>Memory: {@link #forward(MLXScope, List)} follows the {@code SmolVlmModel} per-request pattern:
 * every activation of image {@code i} is allocated into a private child scope of the caller's
 * target scope, and each encoder block's activations are further confined to a private per-block
 * child scope of the image scope — a block's output (and its DeepStack rows, when tapped) is
 * evaluated and hoisted out before the block scope closes, so one block's intermediates do not
 * accumulate across blocks and the per-image peak does not scale with depth (findings §11.4). The
 * image's DeepStack and main-merger rows are hoisted into the target scope, and the block and image
 * scopes are closed on return — the tower's own module scope receives no per-forward handle. The
 * returned {@link Qwen3VlVisionFeatures} arrays therefore live in the target scope, which the
 * caller owns and closes (a per-request scope under the model scope, as {@code SmolVlmModel}'s
 * tower runs under its per-request prefill scope). With more than one image, each output level's
 * per-image parts are concatenated on axis 0, the concatenation is evaluated, and the parts are
 * closed, so only the concatenations are retained. If a forward fails partway, the parts already
 * hoisted for earlier images — and, for a failing image, the per-level parts hoisted from its
 * earlier blocks — remain in the target scope (the caller's to release); the failing image's block
 * and image scopes are still closed.
 */
public final class Qwen3VlVisionTower extends Module {

  /**
   * The reference's {@code Qwen3VLVisionRotaryEmbedding} constructor default: {@code theta =
   * 10000.0}, evaluated in float64 and cast to the float32 the module uses.
   */
  public static final float ROPE_THETA = 10000.0f;

  /** The reference's hard-wired vision LayerNorm epsilon: {@code eps = 1e-6}. */
  public static final float LAYER_NORM_EPS = 1e-6f;

  /** The {@code model.visual.} checkpoint prefix. */
  static final String VISUAL_PREFIX = "model.visual.";

  private final int depth;
  private final int numHeads;
  private final int hidden;
  private final int headDim;
  private final int gridSide;
  private final int merge;
  private final int outHiddenSize;
  private final int inChannels;
  private final int temporal;
  private final int patchHeight;
  private final int patchWidth;
  private final int[] deepstackIndexes;
  private final Conv3d patchEmbedding;
  private final Embedding positionEmbedding;
  private final List<Qwen3VlVisionBlock> layers;
  private final PatchMerger merger;
  private final List<PatchMerger> deepstackMergers;
  private final TwoAxisRope rope;

  /**
   * Builds the tower from checkpoint weights keyed by this module's own dotted parameter paths (see
   * the class javadoc for the mapping; {@link #visionWeights(Map, int, int[])} produces such a map
   * from a raw checkpoint). Every weight must be present and shaped consistently with every other
   * weight; any extra or missing key is rejected. Every weight must be an inexact (floating) dtype,
   * and each non-float32 weight is converted to float32 (evaluated and the original closed, the
   * {@code SmolVlmModel} pattern) before any child module is built.
   *
   * <p>The geometry is derived from the weight shapes the way {@code Idefics3VisionTower} derives
   * its own: the patch-embedding weight is rank-5 channels-last {@code [hidden, temporal,
   * patchHeight, patchWidth, inChannels]}, {@code headDim = hidden / numHeads} must be a multiple
   * of 4 (the two-axis RoPE rotates {@code headDim / 2} even frequencies), {@code
   * positionEmbedding.weight} must be a square {@code gridSide x gridSide} grid, the main merger's
   * {@code linearFc1} width {@code m} must be {@code hidden * merge^2} for a square {@code
   * merge^2}, and every DeepStack merger must share that {@code m} and the merger's output width.
   *
   * @param numHeads the attention head count (not derivable from the weights)
   * @param deepstackIndexes the strictly-increasing layer indices (in {@code [0, depth)}) at which
   *     a DeepStack feature is captured; must be non-empty and match the {@code deepstackMerger*}
   *     subtrees present in {@code weights}
   * @throws IllegalArgumentException if a dimension is invalid, a weight is missing or extra, a
   *     weight is not floating, or the weights are mutually inconsistent
   * @throws NullPointerException if {@code weights} or {@code deepstackIndexes} is {@code null}
   */
  public Qwen3VlVisionTower(
      MLXScope scope, int numHeads, Map<String, MLXArray> weights, int[] deepstackIndexes) {
    super(scope);
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(deepstackIndexes, "deepstackIndexes");
    if (numHeads <= 0) {
      throw new IllegalArgumentException("the vision tower needs at least one attention head");
    }
    if (deepstackIndexes.length == 0) {
      throw new IllegalArgumentException("deepstackIndexes must not be empty");
    }
    int depth = 0;
    while (weights.containsKey("layer" + depth + ".norm1.weight")) {
      depth++;
    }
    if (depth == 0) {
      throw new IllegalArgumentException("the vision tower needs at least one layer");
    }
    for (int i = 0; i < deepstackIndexes.length; i++) {
      int index = deepstackIndexes[i];
      if (index < 0 || index >= depth) {
        throw new IllegalArgumentException(
            "deepstack index " + i + " = " + index + " is out of range [0, " + depth + ")");
      }
      if (i > 0 && index <= deepstackIndexes[i - 1]) {
        throw new IllegalArgumentException(
            "deepstack indexes must be strictly increasing; index "
                + i
                + " has "
                + index
                + " after "
                + deepstackIndexes[i - 1]);
      }
    }

    MLXArray convWeight = weight(weights, "patchEmbedding.weight");
    if (convWeight.ndim() != 5) {
      throw new IllegalArgumentException(
          "patchEmbedding.weight must be rank-5 channels-last, got rank " + convWeight.ndim());
    }
    int[] conv = convWeight.shape();
    for (int i = 0; i < conv.length; i++) {
      if (conv[i] <= 0) {
        throw new IllegalArgumentException(
            "patchEmbedding.weight must have positive dimensions, got " + Arrays.toString(conv));
      }
    }
    int hidden = conv[0];
    requireShape(weight(weights, "patchEmbedding.bias"), "patchEmbedding.bias", new int[] {hidden});
    if (hidden % numHeads != 0) {
      throw new IllegalArgumentException(
          "hidden size " + hidden + " is not divisible by " + numHeads + " attention heads");
    }
    int headDim = hidden / numHeads;
    if (headDim < 4 || headDim % 4 != 0) {
      throw new IllegalArgumentException(
          "headDim "
              + headDim
              + " must be at least 4 and a multiple of 4 (the two-axis RoPE rotates headDim/2"
              + " even frequencies)");
    }
    MLXArray positionTable = weight(weights, "positionEmbedding.weight");
    if (positionTable.ndim() != 2) {
      throw new IllegalArgumentException(
          "positionEmbedding.weight must be a matrix, got rank " + positionTable.ndim());
    }
    int numPatches = positionTable.shape()[0];
    int gridSide = (int) Math.sqrt(numPatches);
    if (numPatches <= 0 || gridSide <= 0 || gridSide * gridSide != numPatches) {
      throw new IllegalArgumentException(
          "positionEmbedding.weight must contain a square grid, got " + numPatches + " rows");
    }
    requireShape(positionTable, "positionEmbedding.weight", new int[] {numPatches, hidden});

    int intermediate = 0;
    for (int i = 0; i < depth; i++) {
      String layer = "layer" + i + ".";
      requireShape(
          weight(weights, layer + "attention.qkv.weight"),
          layer + "attention.qkv.weight",
          new int[] {Math.multiplyExact(3, hidden), hidden});
      requireShape(
          weight(weights, layer + "attention.qkv.bias"),
          layer + "attention.qkv.bias",
          new int[] {Math.multiplyExact(3, hidden)});
      requireShape(
          weight(weights, layer + "attention.proj.weight"),
          layer + "attention.proj.weight",
          new int[] {hidden, hidden});
      requireShape(
          weight(weights, layer + "attention.proj.bias"),
          layer + "attention.proj.bias",
          new int[] {hidden});
      for (String norm : new String[] {"norm1", "norm2"}) {
        requireShape(
            weight(weights, layer + norm + ".weight"),
            layer + norm + ".weight",
            new int[] {hidden});
        requireShape(
            weight(weights, layer + norm + ".bias"), layer + norm + ".bias", new int[] {hidden});
      }
      if (i == 0) {
        intermediate = weight(weights, layer + "mlp.0.weight").shape()[0];
        if (intermediate <= 0) {
          throw new IllegalArgumentException("the feed-forward width must be positive");
        }
      }
      requireShape(
          weight(weights, layer + "mlp.0.weight"),
          layer + "mlp.0.weight",
          new int[] {intermediate, hidden});
      requireShape(
          weight(weights, layer + "mlp.0.bias"), layer + "mlp.0.bias", new int[] {intermediate});
      requireShape(
          weight(weights, layer + "mlp.2.weight"),
          layer + "mlp.2.weight",
          new int[] {hidden, intermediate});
      requireShape(weight(weights, layer + "mlp.2.bias"), layer + "mlp.2.bias", new int[] {hidden});
    }

    MLXArray fc1 = weight(weights, "merger.linearFc1.weight");
    if (fc1.ndim() != 2 || fc1.shape()[0] != fc1.shape()[1]) {
      throw new IllegalArgumentException(
          "merger.linearFc1.weight must be square [m, m], got shape "
              + Arrays.toString(fc1.shape()));
    }
    int m = fc1.shape()[0];
    if (m % hidden != 0) {
      throw new IllegalArgumentException(
          "merger width m = " + m + " must be divisible by hidden size " + hidden);
    }
    int mergeSquare = m / hidden;
    int merge = (int) Math.sqrt(mergeSquare);
    if (merge < 1 || merge * merge != mergeSquare) {
      throw new IllegalArgumentException(
          "merger width m = "
              + m
              + " must be the hidden size times a square merge^2, got merge^2 ="
              + " "
              + mergeSquare);
    }
    requireShape(weight(weights, "merger.norm.weight"), "merger.norm.weight", new int[] {hidden});
    requireShape(weight(weights, "merger.norm.bias"), "merger.norm.bias", new int[] {hidden});
    requireShape(weight(weights, "merger.linearFc1.bias"), "merger.linearFc1.bias", new int[] {m});
    MLXArray fc2 = weight(weights, "merger.linearFc2.weight");
    if (fc2.ndim() != 2) {
      throw new IllegalArgumentException(
          "merger.linearFc2.weight must be a matrix, got rank " + fc2.ndim());
    }
    int outHiddenSize = fc2.shape()[0];
    if (outHiddenSize <= 0) {
      throw new IllegalArgumentException("merger.linearFc2.weight must have positive output rows");
    }
    requireShape(fc2, "merger.linearFc2.weight", new int[] {outHiddenSize, m});
    requireShape(
        weight(weights, "merger.linearFc2.bias"),
        "merger.linearFc2.bias",
        new int[] {outHiddenSize});
    for (int j = 0; j < deepstackIndexes.length; j++) {
      String prefix = "deepstackMerger" + j + ".";
      requireShape(weight(weights, prefix + "norm.weight"), prefix + "norm.weight", new int[] {m});
      requireShape(weight(weights, prefix + "norm.bias"), prefix + "norm.bias", new int[] {m});
      MLXArray deepFc1 = weight(weights, prefix + "linearFc1.weight");
      if (deepFc1.ndim() != 2 || deepFc1.shape()[0] != m || deepFc1.shape()[1] != m) {
        throw new IllegalArgumentException(
            prefix
                + "linearFc1.weight must be square [m, m] = "
                + Arrays.toString(new int[] {m, m})
                + ", got shape "
                + Arrays.toString(deepFc1.shape()));
      }
      requireShape(
          weight(weights, prefix + "linearFc1.bias"), prefix + "linearFc1.bias", new int[] {m});
      requireShape(
          weight(weights, prefix + "linearFc2.weight"),
          prefix + "linearFc2.weight",
          new int[] {outHiddenSize, m});
      requireShape(
          weight(weights, prefix + "linearFc2.bias"),
          prefix + "linearFc2.bias",
          new int[] {outHiddenSize});
    }

    for (Map.Entry<String, MLXArray> entry : weights.entrySet()) {
      if (!entry.getValue().dtype().isInexact()) {
        throw new IllegalArgumentException(
            "vision tower weights must be floating: " + entry.getKey());
      }
    }
    rejectUnexpected(weights, expectedKeys(depth, deepstackIndexes.length));

    Map<String, MLXArray> f32 = new LinkedHashMap<>(weights.size());
    for (Map.Entry<String, MLXArray> entry : weights.entrySet()) {
      MLXArray original = entry.getValue();
      MLXArray value =
          original.dtype() == DType.FLOAT32 ? original : MLX.astype(original, DType.FLOAT32);
      if (value != original) {
        MLX.eval(value);
        original.close();
      }
      f32.put(entry.getKey(), value);
    }

    this.depth = depth;
    this.numHeads = numHeads;
    this.hidden = hidden;
    this.headDim = headDim;
    this.gridSide = gridSide;
    this.merge = merge;
    this.outHiddenSize = outHiddenSize;
    this.inChannels = conv[4];
    this.temporal = conv[1];
    this.patchHeight = conv[2];
    this.patchWidth = conv[3];
    this.deepstackIndexes = deepstackIndexes.clone();
    this.patchEmbedding =
        child(
            "patchEmbedding",
            new Conv3d(
                scope,
                weight(f32, "patchEmbedding.weight"),
                weight(f32, "patchEmbedding.bias"),
                new int[] {this.temporal, this.patchHeight, this.patchWidth},
                new int[] {0, 0, 0},
                new int[] {1, 1, 1},
                1));
    this.positionEmbedding =
        child("positionEmbedding", new Embedding(scope, weight(f32, "positionEmbedding.weight")));
    this.layers = new ArrayList<>(depth);
    for (int i = 0; i < depth; i++) {
      String layer = "layer" + i + ".";
      this.layers.add(
          child(
              "layer" + i,
              new Qwen3VlVisionBlock(
                  scope,
                  new LayerNorm(
                      scope,
                      weight(f32, layer + "norm1.weight"),
                      weight(f32, layer + "norm1.bias"),
                      LAYER_NORM_EPS),
                  new LayerNorm(
                      scope,
                      weight(f32, layer + "norm2.weight"),
                      weight(f32, layer + "norm2.bias"),
                      LAYER_NORM_EPS),
                  new TwoAxisRotaryAttention(
                      scope,
                      numHeads,
                      headDim,
                      ROPE_THETA,
                      weight(f32, layer + "attention.qkv.weight"),
                      weight(f32, layer + "attention.qkv.bias"),
                      weight(f32, layer + "attention.proj.weight"),
                      weight(f32, layer + "attention.proj.bias")),
                  new Sequential(
                      scope,
                      new Linear(
                          scope,
                          weight(f32, layer + "mlp.0.weight"),
                          weight(f32, layer + "mlp.0.bias")),
                      new GELU(scope, true),
                      new Linear(
                          scope,
                          weight(f32, layer + "mlp.2.weight"),
                          weight(f32, layer + "mlp.2.bias"))))));
    }
    this.merger =
        child(
            "merger",
            new PatchMerger(
                scope,
                merge,
                false,
                weight(f32, "merger.norm.weight"),
                weight(f32, "merger.norm.bias"),
                weight(f32, "merger.linearFc1.weight"),
                weight(f32, "merger.linearFc1.bias"),
                weight(f32, "merger.linearFc2.weight"),
                weight(f32, "merger.linearFc2.bias")));
    this.deepstackMergers = new ArrayList<>(deepstackIndexes.length);
    for (int j = 0; j < deepstackIndexes.length; j++) {
      String prefix = "deepstackMerger" + j + ".";
      this.deepstackMergers.add(
          child(
              "deepstackMerger" + j,
              new PatchMerger(
                  scope,
                  merge,
                  true,
                  weight(f32, prefix + "norm.weight"),
                  weight(f32, prefix + "norm.bias"),
                  weight(f32, prefix + "linearFc1.weight"),
                  weight(f32, prefix + "linearFc1.bias"),
                  weight(f32, prefix + "linearFc2.weight"),
                  weight(f32, prefix + "linearFc2.bias"))));
    }
    this.rope = new TwoAxisRope(ROPE_THETA, headDim / 2);
    train(false);
  }

  /** The encoder stack depth (the {@code blocks} count). */
  public int depth() {
    return depth;
  }

  /** The attention head count. */
  public int numHeads() {
    return numHeads;
  }

  /** The vision hidden size (the patch-embedding output width). */
  public int hidden() {
    return hidden;
  }

  /** The per-head dimension, {@code hidden / numHeads}. */
  public int headDim() {
    return headDim;
  }

  /** The position-embedding grid side ({@code sqrt(num_position_embeddings)}). */
  public int gridSide() {
    return gridSide;
  }

  /** The spatial merge size ({@code m / hidden} in the merger's {@code m x m} width). */
  public int spatialMergeSize() {
    return merge;
  }

  /** The merger output width (the reference's {@code out_hidden_size}). */
  public int outHiddenSize() {
    return outHiddenSize;
  }

  /** A copy of the DeepStack capture layer indices. */
  public int[] deepstackIndexes() {
    return deepstackIndexes.clone();
  }

  /**
   * Runs the tower on one or more processed still images. For each image, in order: the pixel rows
   * are transposed to the channels-last conv layout and patch-embedded, the bilinear position
   * embedding is gathered directly in merge-block order and added, the two-axis RoPE is prepared
   * for the image's merge-block coordinates, the pre-norm encoder stack runs while capturing the
   * DeepStack features at {@code deepstackIndexes}, and the main merger runs. The image's DeepStack
   * rows and main-merger row are evaluated and hoisted into {@code target}, and the image's private
   * child scope is closed. The per-image results are concatenated on axis 0 across the input images
   * (a single image is returned as-is).
   *
   * <p>Every image must be a still image ({@code gridThw()} {@code [1, h, w]}), with {@code h} and
   * {@code w} positive multiples of {@link #spatialMergeSize()}, {@code h * w} equal to the
   * result's {@code rawPatchRows()}, {@code patchStride()} equal to the tower's {@code inChannels *
   * temporal * patchHeight * patchWidth} and {@code values().length} equal to {@code rawPatchRows()
   * * patchStride()}.
   *
   * <p>Memory contract (the {@code SmolVlmModel} per-request pattern): every image's activations
   * are allocated into a private child scope of {@code target}, and each encoder block's
   * activations into a further per-block child scope of the image scope that closes after the
   * block's evaluated output is hoisted out, so one block's intermediates do not accumulate across
   * blocks and the per-image peak does not scale with depth. This tower's own module scope receives
   * no per-forward handle. The returned arrays live in {@code target}, which the caller owns and
   * closes. When more than one image is given, each level's per-image parts are concatenated, the
   * concatenation is evaluated, and the per-image parts are closed, so only the concatenations are
   * retained in {@code target}. If a forward fails partway, the parts already hoisted for earlier
   * images — and, for the failing image, the per-level parts hoisted from its earlier blocks —
   * remain in {@code target} (the caller's to release), while the failing image's block and image
   * scopes are still closed.
   *
   * @param target the caller's per-request scope; must be this tower's module scope or a descendant
   *     of it (the per-image child scopes mix this tower's weights with the image activations, and
   *     unrelated scopes are rejected by the first such op)
   * @param images the processed images, in the order their rows must be concatenated
   * @return the concatenated main-merger rows and per-level DeepStack rows, in {@code target}
   * @throws NullPointerException if {@code target}, {@code images} or an element is {@code null}
   * @throws IllegalArgumentException if {@code target} is not this tower's module scope or a
   *     descendant of it, or if an image's geometry disagrees with the tower's
   */
  public Qwen3VlVisionFeatures forward(MLXScope target, List<Qwen3VlImageProcessorResult> images) {
    return run(target, images, null).features();
  }

  /**
   * The golden-verification view of {@link #forward(MLXScope, List)} (findings §11.3): the
   * identical computation and scope handling, plus one {@link ImageCaptures} per input image
   * holding the bilinear-interpolated position-embedding rows, the patch-embedding (convolution)
   * output before the position-embedding add, and {@code 1 + deepstackIndexes.length} block outputs
   * — block 0 first, then each DeepStack source block in this tower's {@code deepstackIndexes}
   * order (so when the first index is 0, block 0's output is captured twice) — the reference's
   * hook-based per-image captures. Every captured array is evaluated and hoisted into {@code
   * target} with the final features, so it outlives each image's private scopes (the image scope
   * and the per-block scopes). Package-private: the golden test ({@code
   * Qwen3VlVisionTowerGoldenTest}) is the intended consumer; {@link #forward(MLXScope, List)} stays
   * the production surface and discards the captures.
   *
   * @param target the caller's scope for the features and captures; see {@link #forward(MLXScope,
   *     List)}
   * @param images the processed images; see {@link #forward(MLXScope, List)}
   * @return the final features and the per-image captures, in {@code target}
   */
  ForwardCaptures forwardForTest(MLXScope target, List<Qwen3VlImageProcessorResult> images) {
    List<ImageCaptures> captures = new ArrayList<>(images.size());
    return run(target, images, captures);
  }

  private ForwardCaptures run(
      MLXScope target, List<Qwen3VlImageProcessorResult> images, List<ImageCaptures> captures) {
    Objects.requireNonNull(target, "target");
    if (!scope().isAncestorOf(target)) {
      throw new IllegalArgumentException(
          "target scope must be the tower's module scope or a descendant of it, got " + target);
    }
    Objects.requireNonNull(images, "images");
    if (images.isEmpty()) {
      throw new IllegalArgumentException("at least one image is required");
    }
    int levelCount = deepstackIndexes.length;
    List<MLXArray> mergedParts = new ArrayList<>(images.size());
    List<List<MLXArray>> deepstackParts = new ArrayList<>(levelCount);
    for (int level = 0; level < levelCount; level++) {
      deepstackParts.add(new ArrayList<>(images.size()));
    }
    for (Qwen3VlImageProcessorResult image : images) {
      Objects.requireNonNull(image, "images element");
      runImage(target, image, mergedParts, deepstackParts, captures);
    }
    MLXArray merged = mergedParts.size() == 1 ? mergedParts.get(0) : concatenate(mergedParts);
    List<MLXArray> levels = new ArrayList<>(levelCount);
    for (int level = 0; level < levelCount; level++) {
      List<MLXArray> parts = deepstackParts.get(level);
      levels.add(parts.size() == 1 ? parts.get(0) : concatenate(parts));
    }
    return new ForwardCaptures(
        new Qwen3VlVisionFeatures(merged, levels),
        captures == null ? List.of() : List.copyOf(captures));
  }

  /**
   * Joins the per-image parts of one output level: concatenates them on axis 0 (the parts all live
   * in the caller's target scope, so the concatenation lands there too), evaluates the
   * concatenation, and closes the per-image parts, so only the concatenation's own buffer is
   * retained in the target scope.
   */
  private static MLXArray concatenate(List<MLXArray> parts) {
    MLXArray joined = MLXShape.concatenate(parts.toArray(MLXArray[]::new), 0);
    MLX.eval(joined);
    for (MLXArray part : parts) {
      part.close();
    }
    return joined;
  }

  /**
   * One image of {@link #forward(MLXScope, List)}: the {@code SmolVlmModel} per-request pattern — a
   * private child scope of the caller's target scope holds the image's own activations, and each
   * encoder block runs in a further private child scope of it: the block's output (and its
   * DeepStack rows, when tapped) is evaluated and hoisted out, then the block scope closes, so one
   * block's intermediates never accumulate across blocks and the per-image peak does not scale with
   * depth. The DeepStack and main-merger rows (and, for the golden test, the intermediate captures)
   * are hoisted into the target scope, and both the block and image scopes are closed on return, so
   * this tower's own module scope receives no handle from the image. If the image's forward fails
   * partway, the per-level rows hoisted from its earlier blocks remain in the target scope (the
   * caller's to release); the block and image scopes are still closed.
   */
  private void runImage(
      MLXScope target,
      Qwen3VlImageProcessorResult image,
      List<MLXArray> mergedParts,
      List<List<MLXArray>> deepstackParts,
      List<ImageCaptures> captures) {
    final int levelCount = deepstackIndexes.length;
    int[] grid = image.gridThw();
    if (grid.length != 3 || grid[0] != 1) {
      throw new IllegalArgumentException(
          "the tower accepts still images with gridThw [1, h, w], got " + Arrays.toString(grid));
    }
    int h = grid[1];
    int w = grid[2];
    if (h <= 0 || w <= 0) {
      throw new IllegalArgumentException(
          "the image grid dimensions must be positive, got " + h + "x" + w);
    }
    if (h % merge != 0 || w % merge != 0) {
      throw new IllegalArgumentException(
          "image grid " + h + "x" + w + " must be multiples of the spatial merge size " + merge);
    }
    int n = Math.multiplyExact(h, w);
    if (image.rawPatchRows() != n) {
      throw new IllegalArgumentException(
          "image grid "
              + h
              + "x"
              + w
              + " implies "
              + n
              + " raw patch rows, got "
              + image.rawPatchRows());
    }
    int expectedStride =
        Math.multiplyExact(
            Math.multiplyExact(Math.multiplyExact(inChannels, temporal), patchHeight), patchWidth);
    if (image.patchStride() != expectedStride) {
      throw new IllegalArgumentException(
          "image patch stride "
              + image.patchStride()
              + " does not match the tower's "
              + expectedStride
              + " (inChannels * temporal * patchHeight * patchWidth)");
    }
    if (image.values().length != Math.multiplyExact(n, expectedStride)) {
      throw new IllegalArgumentException(
          "image values must be rawPatchRows * patchStride = "
              + Math.multiplyExact(n, expectedStride)
              + " entries, got "
              + image.values().length);
    }

    try (MLXScope img = target.newChild()) {
      MLXArray pixels =
          MLX.array(
              img, image.values(), new int[] {n, inChannels, temporal, patchHeight, patchWidth});
      // PyTorch [N, C, T, H, W] -> MLX channels-last [N, T, H, W, C].
      MLXArray channelsLast = MLXShape.transpose(pixels, new int[] {0, 2, 3, 4, 1});
      MLXArray patchRows =
          MLXShape.reshape(patchEmbedding.forward(channelsLast), new int[] {n, hidden});

      int[][] coords = rotaryCoordinates(h, w, merge);
      int[] rowCoords = coords[0];
      int[] colCoords = coords[1];
      PositionLookup lookup = positionLookup(h, w, gridSide);
      int[] flatIndices = new int[Math.multiplyExact(4, n)];
      float[] cornerWeights = new float[Math.multiplyExact(4, n)];
      for (int m = 0; m < n; m++) {
        int k = rowCoords[m] * w + colCoords[m];
        for (int c = 0; c < 4; c++) {
          flatIndices[c * n + m] = lookup.cornerIndices()[c][k];
          cornerWeights[c * n + m] = lookup.cornerWeights()[c][k];
        }
      }
      // pos_embed(idx) * weight[:, :, None], then the reference's left-associative 4-way add,
      // gathered directly in merge-block order (per-row, so identical bits to the reference's
      // sum-then-permute).
      MLXArray idxTensor = MLX.array(img, flatIndices, new int[] {4, n});
      MLXArray cornerRows = positionEmbedding.forward(idxTensor);
      MLXArray weightTensor = MLX.array(img, cornerWeights, new int[] {4, n, 1});
      MLXArray[] parts = MLXShape.split(MLXOps.multiply(cornerRows, weightTensor), 4, 0);
      MLXArray pe = MLXShape.squeeze(parts[0], new int[] {0});
      for (int c = 1; c < 4; c++) {
        pe = MLXOps.add(pe, MLXShape.squeeze(parts[c], new int[] {0}));
      }
      MLXArray x = MLXOps.add(patchRows, pe);

      TwoAxisRotation rotation = rope.prepare(img, n, rowCoords, colCoords);
      List<MLXArray> capturedBlocks = captures == null ? null : new ArrayList<>(levelCount + 1);
      int level = 0;
      for (int i = 0; i < depth; i++) {
        try (MLXScope block = img.newChild()) {
          // Pin this block's entry row into the block scope the way T5Model.project does: the
          // same-shape broadcast view is a no-op that allocates a handle into `block`, so every
          // op result of the block resolves into it (the weights and the rotation tables live in
          // ancestor scopes, and GELU's scalar constants follow the input's scope).
          MLXArray in = MLXShape.broadcastTo(x, block, x.shape());
          MLXArray out = layers.get(i).forward(in, rotation);
          boolean tap = level < levelCount && i == deepstackIndexes[level];
          MLXArray ds = tap ? deepstackMergers.get(level).forward(out) : null;
          // Evaluate this block before the next one starts: closing the block scope below frees
          // every intermediate of this block (the peak does not accumulate across blocks), while
          // `out` (and `ds`) are hoisted out first, so the computation is unchanged — uncompiled
          // MLX does not fuse across eval boundaries (verified bit-identical in both TF32 modes
          // against the pre-change recording, findings §11.4).
          MLX.eval(tap ? new MLXArray[] {out, ds} : new MLXArray[] {out});
          // The block-0 capture and the DeepStack capture are independent conditions: when
          // deepstackIndexes[0] == 0 both fire in this order, so blockOutputs keeps the
          // documented 1 + levelCount entries (block 0 first, then the DeepStack source blocks in
          // deepstackIndexes order).
          if (captures != null && i == 0) {
            capturedBlocks.add(MLX.hoist(out, target));
          }
          if (tap) {
            if (captures != null) {
              capturedBlocks.add(MLX.hoist(out, target));
            }
            deepstackParts.get(level).add(MLX.hoist(ds, target));
            level++;
          }
          // Drop the previous residual before the next block's intermediates accumulate: only
          // the hoisted `out` (now `x`) survives this block's scope.
          x.close();
          x = MLX.hoist(out, img);
        }
      }
      MLXArray merged = merger.forward(x);

      List<MLXArray> toEvaluate = new ArrayList<>(2);
      toEvaluate.add(merged);
      if (captures != null) {
        toEvaluate.add(pe);
        toEvaluate.add(patchRows);
        toEvaluate.addAll(capturedBlocks);
      }
      MLX.eval(toEvaluate.toArray(MLXArray[]::new));
      mergedParts.add(MLX.hoist(merged, target));
      if (captures != null) {
        captures.add(
            new ImageCaptures(
                MLX.hoist(pe, target),
                MLX.hoist(patchRows, target),
                capturedBlocks.toArray(MLXArray[]::new)));
      }
    }
  }

  /**
   * The bilinear corner indices and weights of {@code fast_pos_embed_interpolate} for one image
   * grid, in row-major patch order: for patch {@code (ph, pw)} at position {@code k = ph * w + pw},
   * the fractional coordinates are {@code linspaceValue(gridSide, h, ph)} and {@code
   * linspaceValue(gridSide, w, pw)}; the floor index (torch's {@code .int()} truncation, always
   * non-negative here) and its clipped ceiling give the four row-major corner indices {@code
   * [floorH * gridSide + floorW, floorH * gridSide + ceilW, ceilH * gridSide + floorW, ceilH *
   * gridSide + ceilW]}, and the weights are the single-float32-multiply products {@code (1 - dh) *
   * (1 - dw)}, {@code (1 - dh) * dw}, {@code dh * (1 - dw)}, {@code dh * dw} with one float32
   * subtraction each for {@code dh = hv - floorH} and {@code dw} (the reference computes {@code 1 -
   * dh} in float32 before the broadcast multiplication).
   *
   * <p>Consumers that need merge-block order gather {@code [c][row * w + col]} with the {@code
   * (row, col)} pairs of {@link #rotaryCoordinates(int, int, int)} — the per-row sum of the four
   * weighted corner rows is then bit-identical to the reference's sum-then-permute.
   *
   * @throws IllegalArgumentException if a dimension is not positive
   */
  static PositionLookup positionLookup(int h, int w, int gridSide) {
    if (h <= 0 || w <= 0 || gridSide <= 0) {
      throw new IllegalArgumentException("h, w and gridSide must be positive");
    }
    int n = Math.multiplyExact(h, w);
    int[][] cornerIndices = new int[4][n];
    float[][] cornerWeights = new float[4][n];
    for (int ph = 0; ph < h; ph++) {
      float hv = linspaceValue(gridSide, h, ph);
      int floorH = (int) hv;
      int ceilH = Math.min(floorH + 1, gridSide - 1);
      float dh = hv - (float) floorH;
      float oneMinusDh = 1f - dh;
      int baseFloor = floorH * gridSide;
      int baseCeil = ceilH * gridSide;
      for (int pw = 0; pw < w; pw++) {
        float wv = linspaceValue(gridSide, w, pw);
        int floorW = (int) wv;
        int ceilW = Math.min(floorW + 1, gridSide - 1);
        float dw = wv - (float) floorW;
        final float oneMinusDw = 1f - dw;
        int k = ph * w + pw;
        cornerIndices[0][k] = baseFloor + floorW;
        cornerIndices[1][k] = baseFloor + ceilW;
        cornerIndices[2][k] = baseCeil + floorW;
        cornerIndices[3][k] = baseCeil + ceilW;
        cornerWeights[0][k] = oneMinusDh * oneMinusDw;
        cornerWeights[1][k] = oneMinusDh * dw;
        cornerWeights[2][k] = dh * oneMinusDw;
        cornerWeights[3][k] = dh * dw;
      }
    }
    return new PositionLookup(cornerIndices, cornerWeights);
  }

  /**
   * The reference's float32 {@code torch.linspace(0, gridSide - 1, n)} value at index {@code i},
   * ported exactly from the torch 2.6.0 CPU {@code linspace_kernel} ({@code
   * aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp} at tag {@code v2.6.0}): the step is the
   * float32 rounding of the double division {@code (double)(end - start) / (n - 1)}, and each value
   * is the float32 rounding of a double expression — {@code start + (double) step * i} for {@code i
   * < n / 2}, otherwise {@code end - (double) step * (n - i - 1)} (the C++
   * usual-arithmetic-conversion promotions of the kernel's float/int64 operands). The composite
   * {@code linspace_out} ({@code aten/src/ATen/native/RangeFactories.cpp}) fills a one-step result
   * with {@code start}; the kernel itself would produce {@code NaN} there. A naive closed form
   * ({@code start + (end - start) * i / (n - 1)}) diverges by one float32 ulp for {@code n = 4, i =
   * 2} (versus the pinned golden); the pure-float32 split-halfway port diverges at further grid
   * points — the double intermediates are load-bearing.
   *
   * @param side the position-grid side ({@code gridSide})
   * @param n the requested count (the image's patch count along the axis)
   * @param i the output index, in {@code [0, n)}
   * @return the float32 value
   * @throws IllegalArgumentException if {@code side} or {@code n} is not positive or {@code i} is
   *     out of range
   */
  static float linspaceValue(int side, int n, int i) {
    if (side <= 0 || n <= 0) {
      throw new IllegalArgumentException("side and n must be positive");
    }
    if (i < 0 || i >= n) {
      throw new IllegalArgumentException("i = " + i + " is out of range [0, " + n + ")");
    }
    if (n == 1) {
      return 0f; // the composite's fill_(start) guard; start is always 0 here
    }
    float end = (float) (side - 1);
    float step = (float) ((double) (end - 0f) / (double) (n - 1));
    if (i < n / 2) {
      return (float) (0.0d + (double) step * (double) i);
    }
    return (float) ((double) end - (double) step * (double) (n - i - 1));
  }

  /**
   * The merge-block (row, col) full-resolution patch coordinates of {@code rot_pos_emb}, the
   * block-row, block-col, intra-block-row, intra-block-col loop order of the reference's {@code
   * row_idx}/{@code col_idx} expansion: output position {@code m} is the patch whose merge-block
   * position is the {@code m}-th element of {@code (blockRow, blockCol, intraRow, intraCol)}
   * flattened row-major, at full-resolution row {@code blockRow * merge + intraRow} and column
   * {@code blockCol * merge + intraCol}.
   *
   * <p>For the 2-by-2 merge of an 8-by-8 grid the first eight coordinates are {@code (0,0), (0,1),
   * (1,0), (1,1), (0,2), (0,3), (1,2), (1,3)} and the last four are {@code (6,6), (6,7), (7,6),
   * (7,7)}.
   *
   * @return the {@code {rows, cols}} coordinate arrays, both in merge-block order
   * @throws IllegalArgumentException if a dimension is not a positive multiple of {@code merge}
   */
  public static int[][] rotaryCoordinates(int h, int w, int merge) {
    if (merge < 1) {
      throw new IllegalArgumentException("merge must be positive");
    }
    if (h <= 0 || w <= 0) {
      throw new IllegalArgumentException("h and w must be positive");
    }
    if (h % merge != 0 || w % merge != 0) {
      throw new IllegalArgumentException(
          "h = " + h + " and w = " + w + " must be multiples of merge = " + merge);
    }
    int n = Math.multiplyExact(h, w);
    int[] rows = new int[n];
    int[] cols = new int[n];
    int m = 0;
    for (int blockRow = 0; blockRow < h / merge; blockRow++) {
      for (int blockCol = 0; blockCol < w / merge; blockCol++) {
        for (int intraRow = 0; intraRow < merge; intraRow++) {
          for (int intraCol = 0; intraCol < merge; intraCol++) {
            rows[m] = blockRow * merge + intraRow;
            cols[m] = blockCol * merge + intraCol;
            m++;
          }
        }
      }
    }
    return new int[][] {rows, cols};
  }

  /**
   * The checkpoint-key to tower-key rename of the class javadoc, as a pure function: {@code
   * model.visual.patch_embed.proj.*} to {@code patchEmbedding.*}, {@code model.visual.pos_embed.
   * weight} to {@code positionEmbedding.weight}, {@code model.visual.blocks.{i}.norm{1,2}.*} to
   * {@code layer{i}.norm{1,2}.*}, {@code model.visual.blocks.{i}.attn.{qkv,proj}.*} to {@code
   * layer{i}.attention.{qkv,proj}.*}, {@code model.visual.blocks.{i}.mlp.linear_fc{1,2}.*} to
   * {@code layer{i}.mlp.{0,2}.*}, {@code model.visual.merger.{norm,linear_fc1,linear_fc2}.*} to
   * {@code merger.{norm,linearFc1,linearFc2}.*} and {@code model.visual.
   * deepstack_merger_list.{j}.{norm,linear_fc1,linear_fc2}.*} to {@code deepstackMerger{j}.{norm,
   * linearFc1,linearFc2}.*}.
   *
   * @param key the full checkpoint key
   * @param depth the tower depth (blocks {@code {i}} with {@code i >= depth} map to {@code null})
   * @param deepstackIndexes the DeepStack indices; list position {@code j} maps to {@code
   *     deepstackMerger{j}} (any {@code j >= length} maps to {@code null})
   * @return the tower key, or {@code null} for non-visual keys, out-of-scheme indices, or unknown
   *     sub-keys
   * @throws NullPointerException if an argument is {@code null}
   */
  public static String visionKeyToTowerKey(String key, int depth, int[] deepstackIndexes) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(deepstackIndexes, "deepstackIndexes");
    if (!key.startsWith(VISUAL_PREFIX)) {
      return null;
    }
    String rest = key.substring(VISUAL_PREFIX.length());
    if (rest.startsWith("patch_embed.proj.")) {
      String tail = rest.substring("patch_embed.proj.".length());
      if (tail.equals("weight")) {
        return "patchEmbedding.weight";
      }
      if (tail.equals("bias")) {
        return "patchEmbedding.bias";
      }
      return null;
    }
    if (rest.equals("pos_embed.weight")) {
      return "positionEmbedding.weight";
    }
    if (rest.startsWith("blocks.")) {
      String tail = rest.substring("blocks.".length());
      Integer index = visionListIndex(tail);
      if (index == null || index >= depth) {
        return null;
      }
      String sub = tail.substring(tail.indexOf('.') + 1);
      if (sub.equals("norm1.weight")
          || sub.equals("norm1.bias")
          || sub.equals("norm2.weight")
          || sub.equals("norm2.bias")) {
        return "layer" + index + "." + sub;
      }
      if (sub.startsWith("attn.")) {
        String name = sub.substring("attn.".length());
        if (name.equals("qkv.weight")
            || name.equals("qkv.bias")
            || name.equals("proj.weight")
            || name.equals("proj.bias")) {
          return "layer" + index + ".attention." + name;
        }
      }
      if (sub.startsWith("mlp.")) {
        String name = sub.substring("mlp.".length());
        if (name.equals("linear_fc1.weight") || name.equals("linear_fc1.bias")) {
          return "layer" + index + ".mlp.0." + name.substring("linear_fc1.".length());
        }
        if (name.equals("linear_fc2.weight") || name.equals("linear_fc2.bias")) {
          return "layer" + index + ".mlp.2." + name.substring("linear_fc2.".length());
        }
      }
      return null;
    }
    if (rest.startsWith("merger.")) {
      return mergerKey(rest.substring("merger.".length()), "merger");
    }
    if (rest.startsWith("deepstack_merger_list.")) {
      String tail = rest.substring("deepstack_merger_list.".length());
      Integer index = visionListIndex(tail);
      if (index == null || index >= deepstackIndexes.length) {
        return null;
      }
      return mergerKey(tail.substring(tail.indexOf('.') + 1), "deepstackMerger" + index);
    }
    return null;
  }

  /**
   * Extracts the tower weights from a raw checkpoint: every key is run through {@link
   * #visionKeyToTowerKey(String, int, int[])}, non-visual keys are skipped, and the result is a
   * fresh map of tower keys to the checkpoint's arrays (the conv weight is the exception: it is
   * transposed {@code {0, 2, 3, 4, 1}} from PyTorch {@code [out, in, T, H, W]} to jmlx's
   * channels-last {@code [out, T, H, W, in]}, evaluated, with the original closed — the {@code
   * SmolVlmModel} pattern). Every {@code model.visual.*} key that does not map to a known tower key
   * is rejected outright rather than skipped: a block or DeepStack merger index past the configured
   * depth or DeepStack count is rejected by name with the limit that excludes it (dropping it would
   * truncate the tower to the configured depth instead of surfacing the mismatch), and any other
   * unrecognised visual key — an unknown sub-key, an unknown top-level visual key, or a non-numeric
   * or overflowing index — is rejected by name. Only genuinely non-visual checkpoint keys are
   * skipped. Completeness against the tower's expected-key set is the constructor's job ({@code
   * rejectUnexpected}), not this method's.
   *
   * <p>All-or-nothing: the method makes two passes. The first pass maps and validates every key —
   * the duplicate, out-of-range and unrecognised-key rejections above — and does nothing else; the
   * second pass only then materializes the result, transposing the conv weight there. A thrown
   * {@link IllegalArgumentException} therefore leaves the caller's checkpoint map byte-for-byte
   * untouched: every entry is still its original, still-open {@code MLXArray}, and no transposed
   * copy has been allocated. On success the returned map holds the caller's own arrays for every
   * mapped key except {@code patchEmbedding.weight}, which holds the transposed, evaluated copy
   * (allocated into the original's scope); the caller's original conv weight array is closed, and
   * the caller's map itself is never written to.
   *
   * @throws IllegalArgumentException if two checkpoint keys map to the same tower key, if a key
   *     names a block or DeepStack merger index out of the configured range, or if a {@code
   *     model.visual.*} key is not recognised
   * @throws NullPointerException if an argument is {@code null}
   */
  public static Map<String, MLXArray> visionWeights(
      Map<String, MLXArray> checkpoint, int depth, int[] deepstackIndexes) {
    Objects.requireNonNull(checkpoint, "checkpoint");
    Objects.requireNonNull(deepstackIndexes, "deepstackIndexes");
    // Pass 1: map and validate every key before transposing, closing or inserting anything, so a
    // rejection leaves the caller's map untouched.
    Set<String> towerKeys = new HashSet<>();
    for (String key : checkpoint.keySet()) {
      String towerKey = visionKeyToTowerKey(key, depth, deepstackIndexes);
      if (towerKey == null) {
        if (!key.startsWith(VISUAL_PREFIX)) {
          continue;
        }
        String outOfRange = outOfRangeVisionIndex(key, depth, deepstackIndexes);
        if (outOfRange != null) {
          throw new IllegalArgumentException(outOfRange);
        }
        throw new IllegalArgumentException("unrecognized vision checkpoint key \"" + key + "\"");
      }
      if (!towerKeys.add(towerKey)) {
        throw new IllegalArgumentException(
            "duplicate vision tower key \"" + towerKey + "\" from checkpoint key \"" + key + "\"");
      }
    }
    // Pass 2: pass 1 proved every visual key maps to a distinct tower key, so this cannot fail;
    // the transpose happens only here, after the whole map is known valid.
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    for (Map.Entry<String, MLXArray> entry : checkpoint.entrySet()) {
      String towerKey = visionKeyToTowerKey(entry.getKey(), depth, deepstackIndexes);
      if (towerKey == null) {
        continue; // pass 1 proved every unmapped key is non-visual
      }
      MLXArray value = entry.getValue();
      if (towerKey.equals("patchEmbedding.weight")) {
        MLXArray transposed = MLXShape.transpose(value, new int[] {0, 2, 3, 4, 1});
        MLX.eval(transposed);
        value.close();
        value = transposed;
      }
      weights.put(towerKey, value);
    }
    return weights;
  }

  /**
   * The out-of-range rejection of {@link #visionWeights(Map, int, int[])}: a {@code
   * model.visual.blocks.{i}} or {@code model.visual.deepstack_merger_list.{j}} key whose numeric
   * index is out of the configured depth (respectively the configured DeepStack count) is a
   * well-formed visual key that {@link #visionKeyToTowerKey(String, int, int[])} rejected only
   * because of its index — the signature of a checkpoint deeper, or with more DeepStack mergers,
   * than the configuration, which must fail loudly instead of truncating the tower silently.
   * Returns the rejection message, or {@code null} for every other key — the non-visual keys {@code
   * visionWeights} skips, and the non-numeric, overflowing or missing indices it rejects as
   * unrecognised.
   */
  private static String outOfRangeVisionIndex(String key, int depth, int[] deepstackIndexes) {
    if (!key.startsWith(VISUAL_PREFIX)) {
      return null;
    }
    String rest = key.substring(VISUAL_PREFIX.length());
    if (rest.startsWith("blocks.")) {
      Integer index = visionListIndex(rest.substring("blocks.".length()));
      if (index != null && index >= depth) {
        return "checkpoint key \""
            + key
            + "\" names block "
            + index
            + ", out of range [0, "
            + depth
            + ") for the configured depth";
      }
      return null;
    }
    if (rest.startsWith("deepstack_merger_list.")) {
      Integer index = visionListIndex(rest.substring("deepstack_merger_list.".length()));
      if (index != null && index >= deepstackIndexes.length) {
        return "checkpoint key \""
            + key
            + "\" names DeepStack merger "
            + index
            + ", out of range [0, "
            + deepstackIndexes.length
            + ") for the configured DeepStack count";
      }
    }
    return null;
  }

  /** The {@code {norm,linearFc1,linear_fc2}} rename shared by the main and DeepStack mergers. */
  private static String mergerKey(String sub, String prefix) {
    if (sub.equals("norm.weight") || sub.equals("norm.bias")) {
      return prefix + "." + sub;
    }
    if (sub.equals("linear_fc1.weight") || sub.equals("linear_fc1.bias")) {
      return prefix + ".linearFc1." + sub.substring("linear_fc1.".length());
    }
    if (sub.equals("linear_fc2.weight") || sub.equals("linear_fc2.bias")) {
      return prefix + ".linearFc2." + sub.substring("linear_fc2.".length());
    }
    return null;
  }

  /**
   * The numeric list index before the tail's first dot (the {@code i} of {@code blocks.{i}.} or the
   * {@code j} of {@code deepstack_merger_list.{j}.}), or {@code null} for a dot-less tail or a
   * non-numeric or overflowing digit run.
   */
  private static Integer visionListIndex(String tail) {
    int dot = tail.indexOf('.');
    if (dot < 1) {
      return null;
    }
    return parseNonNegativeInt(tail.substring(0, dot));
  }

  /** The non-negative decimal parse, or {@code null} for a non-numeric or overflowing digit run. */
  private static Integer parseNonNegativeInt(String digits) {
    if (digits.isEmpty()) {
      return null;
    }
    for (int i = 0; i < digits.length(); i++) {
      if (digits.charAt(i) < '0' || digits.charAt(i) > '9') {
        return null;
      }
    }
    try {
      return Integer.parseInt(digits);
    } catch (NumberFormatException e) {
      return null;
    }
  }

  /**
   * The bilinear corner data of {@link #positionLookup(int, int, int)}: both arrays are {@code
   * [4][n]} with {@code n = h * w} and patch position {@code k = ph * w + pw} row-major; corner
   * order is {@code (floorH, floorW), (floorH, ceilW), (ceilH, floorW), (ceilH, ceilW)}.
   */
  static final record PositionLookup(int[][] cornerIndices, float[][] cornerWeights) {}

  /**
   * One image's golden-verification captures of {@link #forwardForTest(MLXScope, List)} (findings
   * §11.3): the interpolated position-embedding rows {@code [n, hidden]}, the patch-embedding
   * (convolution) output before the position-embedding add {@code [n, hidden]}, and the block
   * outputs — exactly {@code 1 + deepstackIndexes.length} entries: block 0 first, then the
   * DeepStack source blocks in this tower's {@code deepstackIndexes} order — each {@code [n,
   * hidden]}. When {@code deepstackIndexes[0]} is 0, block 0's output appears twice: once as its
   * own entry 0 and once as the DeepStack level-0 source at entry 1. Every array lives in the
   * target scope passed to {@link #forwardForTest(MLXScope, List)}.
   */
  record ImageCaptures(
      MLXArray posEmbedInterpolated, MLXArray patchEmbed, MLXArray[] blockOutputs) {
    ImageCaptures {
      Objects.requireNonNull(posEmbedInterpolated, "posEmbedInterpolated");
      Objects.requireNonNull(patchEmbed, "patchEmbed");
      Objects.requireNonNull(blockOutputs, "blockOutputs");
      blockOutputs = blockOutputs.clone();
    }
  }

  /**
   * The final features of {@link #forwardForTest(MLXScope, List)} plus the per-image {@link
   * ImageCaptures}, one per input image in input order (empty for {@link #forward(MLXScope,
   * List)}).
   */
  record ForwardCaptures(Qwen3VlVisionFeatures features, List<ImageCaptures> images) {
    ForwardCaptures {
      Objects.requireNonNull(features, "features");
      images = List.copyOf(Objects.requireNonNull(images, "images"));
    }
  }

  private static MLXArray weight(Map<String, MLXArray> weights, String name) {
    MLXArray value = weights.get(name);
    if (value == null) {
      throw new IllegalArgumentException("missing vision tower weight \"" + name + "\"");
    }
    return value;
  }

  private static void requireShape(MLXArray array, String name, int[] expected) {
    if (array.ndim() != expected.length) {
      throw new IllegalArgumentException(
          name + " must be rank " + expected.length + ", got " + array.ndim());
    }
    for (int i = 0; i < expected.length; i++) {
      if (array.shape()[i] != expected[i]) {
        throw new IllegalArgumentException(
            name
                + " must be "
                + Arrays.toString(expected)
                + ", got "
                + Arrays.toString(array.shape()));
      }
    }
  }

  private static Set<String> expectedKeys(int depth, int deepstackCount) {
    Set<String> expected = new TreeSet<>();
    expected.add("patchEmbedding.weight");
    expected.add("patchEmbedding.bias");
    expected.add("positionEmbedding.weight");
    for (int i = 0; i < depth; i++) {
      String layer = "layer" + i + ".";
      for (String name : new String[] {"qkv", "proj"}) {
        expected.add(layer + "attention." + name + ".weight");
        expected.add(layer + "attention." + name + ".bias");
      }
      for (String norm : new String[] {"norm1", "norm2"}) {
        expected.add(layer + norm + ".weight");
        expected.add(layer + norm + ".bias");
      }
      expected.add(layer + "mlp.0.weight");
      expected.add(layer + "mlp.0.bias");
      expected.add(layer + "mlp.2.weight");
      expected.add(layer + "mlp.2.bias");
    }
    expected.add("merger.norm.weight");
    expected.add("merger.norm.bias");
    expected.add("merger.linearFc1.weight");
    expected.add("merger.linearFc1.bias");
    expected.add("merger.linearFc2.weight");
    expected.add("merger.linearFc2.bias");
    for (int j = 0; j < deepstackCount; j++) {
      String prefix = "deepstackMerger" + j + ".";
      expected.add(prefix + "norm.weight");
      expected.add(prefix + "norm.bias");
      expected.add(prefix + "linearFc1.weight");
      expected.add(prefix + "linearFc1.bias");
      expected.add(prefix + "linearFc2.weight");
      expected.add(prefix + "linearFc2.bias");
    }
    return expected;
  }

  private static void rejectUnexpected(Map<String, MLXArray> weights, Set<String> expected) {
    Set<String> extra = new TreeSet<>(weights.keySet());
    extra.removeAll(expected);
    if (!extra.isEmpty()) {
      throw new IllegalArgumentException("unexpected vision tower weights: " + extra);
    }
  }
}
