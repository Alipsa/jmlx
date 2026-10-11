package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.AttentionMask;
import se.alipsa.jmlx.nn.BidirectionalAttention;
import se.alipsa.jmlx.nn.Conv2d;
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.EncoderBlock;
import se.alipsa.jmlx.nn.GELU;
import se.alipsa.jmlx.nn.LayerNorm;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.Sequential;

/**
 * The Idefics3 vision tower: the SigLIP-style patch embedding, the pre-norm encoder stack and the
 * post LayerNorm of {@code Idefics3VisionTransformer}, ported from the pinned Hugging Face
 * Transformers 4.57.6 {@code modeling_idefics3.py} (see {@code tools/hf-reference/provenance.json})
 * — {@code Idefics3VisionTransformer} (lines 449-516), {@code Idefics3VisionEmbeddings} (104-165),
 * {@code Idefics3VisionAttention} (193-255), {@code Idefics3VisionMLP} (259-271), {@code
 * Idefics3EncoderLayer} (286-318) and {@code Idefics3Encoder} (322-353). HF's additive float
 * attention mask ({@code _prepare_4d_attention_mask}) is replaced by jmlx's BOOL mask convention:
 * {@link AttentionMask#bidirectional} built from the same binary patch validity, filled with {@code
 * -Float.MAX_VALUE} in scores, which is bit-identical to HF's {@code finfo.min} fill in float32; an
 * explicitly all-true mask is numerically identical to HF passing {@code None}.
 *
 * <p>The module child names mirror the HF checkpoint tree: {@code patchEmbedding} is {@code
 * embeddings.patch_embedding} (weight transposed to jmlx's channels-last {@code [hidden, patch,
 * patch, channels]} layout), {@code positionEmbedding} is {@code embeddings.position_embedding},
 * {@code layer0..layerN-1} are {@code encoder.layers.{i}} (with {@code
 * attention.{query,key,value,output}} for {@code self_attn.{q,k,v,out}_proj}, {@code
 * attentionNorm}/{@code outputNorm} for {@code layer_norm1}/{@code layer_norm2} and {@code
 * feedForward.0/.2} for {@code mlp.fc1/fc2}, the tanh GELU {@code feedForward.1} carrying no
 * parameters), and {@code postLayernorm} is {@code post_layernorm}. The tower also registers the
 * model-supplied {@link Idefics3Connector} as the child {@code connector} (the checkpoint's {@code
 * connector.*} subtree): {@link #forward} deliberately stops after the post LayerNorm, and the
 * model reaches the connector through {@link #connector()} to project the tower's output into the
 * text hidden space.
 */
public final class Idefics3VisionTower extends Module {

  /**
   * The {@code (1 - 1e-6)} multiplier of the fractional position coordinates, in the exact float32
   * the pinned reference uses: torch evaluates the scalar in float64 and casts, and every
   * float32-rounding candidate ({@code 1f - 1e-6f}, {@code (float)(1.0 - 1e-6)}, the literal {@code
   * 0.999999f}) lands on the same bits, {@code 0x3f7fffef}.
   */
  private static final float FRACTIONAL_MULTIPLIER = 0.999999f;

  private final int patchSize;
  private final int hidden;
  private final int positionSide;
  private final int numChannels;
  private final Conv2d patchEmbedding;
  private final Embedding positionEmbedding;
  private final List<EncoderBlock> layers = new ArrayList<>();
  private final LayerNorm postLayernorm;
  private final Idefics3Connector connector;

  /**
   * Builds the tower from checkpoint weights keyed by this module's own dotted parameter paths (see
   * the class javadoc for the HF key mapping). Every weight must be present and shaped consistently
   * with every other weight; any extra or missing key is rejected.
   *
   * @param connector the image-feature connector, registered as the tower's {@code connector}
   *     child; {@link #forward} returns the post-LayerNorm states, and the model applies {@code
   *     connector.forward} to them (via {@link #connector()}) to get the final text-space features
   * @throws IllegalArgumentException if a dimension is invalid, a weight is missing or extra, or
   *     the weights are mutually inconsistent
   * @throws NullPointerException if {@code connector} is {@code null}
   */
  public Idefics3VisionTower(
      MLXScope scope,
      int numLayers,
      int numHeads,
      float layerNormEps,
      Map<String, MLXArray> weights,
      Idefics3Connector connector) {
    super(scope);
    Objects.requireNonNull(weights, "weights");
    if (numLayers <= 0) {
      throw new IllegalArgumentException("the vision tower needs at least one encoder layer");
    }
    if (numHeads <= 0) {
      throw new IllegalArgumentException("the vision tower needs at least one attention head");
    }
    if (!Float.isFinite(layerNormEps) || layerNormEps <= 0) {
      throw new IllegalArgumentException("layerNormEps must be a positive finite value");
    }

    MLXArray convBias = weight(weights, "patchEmbedding.bias");
    int hiddenSize = convBias.shape()[0];
    MLXArray convWeight = weight(weights, "patchEmbedding.weight");
    int[] conv = convWeight.shape();
    if (conv.length != 4
        || conv[0] != hiddenSize
        || conv[1] <= 0
        || conv[1] != conv[2]
        || conv[3] <= 0
        || convBias.shape().length != 1) {
      throw new IllegalArgumentException(
          "patchEmbedding weights must be consistent [hidden, patch, patch, channels] channels"
              + "-last with a [hidden] bias, got "
              + java.util.Arrays.toString(conv)
              + " and "
              + java.util.Arrays.toString(convBias.shape()));
    }
    this.patchSize = conv[1];
    this.hidden = hiddenSize;
    this.numChannels = conv[3];
    MLXArray positionTable = weight(weights, "positionEmbedding.weight");
    if (positionTable.ndim() != 2) {
      throw new IllegalArgumentException("positionEmbedding.weight must be a matrix");
    }
    int numPatches = positionTable.shape()[0];
    positionSide = (int) Math.sqrt(numPatches);
    if (positionSide <= 0 || positionSide * positionSide != numPatches) {
      throw new IllegalArgumentException("positionEmbedding.weight must contain a square grid");
    }
    requireShape(positionTable, "positionEmbedding.weight", new int[] {numPatches, hiddenSize});
    for (int i = 0; i < numLayers; i++) {
      String layer = "layer" + i + ".";
      for (String name : new String[] {"query", "key", "value", "output"}) {
        requireShape(
            weight(weights, layer + "attention." + name + ".weight"),
            layer + "attention." + name + ".weight",
            new int[] {hiddenSize, hiddenSize});
        requireShape(
            weight(weights, layer + "attention." + name + ".bias"),
            layer + "attention." + name + ".bias",
            new int[] {hiddenSize});
      }
      for (String norm : new String[] {"attentionNorm", "outputNorm"}) {
        requireShape(
            weight(weights, layer + norm + ".weight"),
            layer + norm + ".weight",
            new int[] {hiddenSize});
        requireShape(
            weight(weights, layer + norm + ".bias"),
            layer + norm + ".bias",
            new int[] {hiddenSize});
      }
      int intermediate = weight(weights, layer + "feedForward.0.weight").shape()[0];
      if (intermediate <= 0) {
        throw new IllegalArgumentException("the feed-forward width must be positive");
      }
      requireShape(
          weight(weights, layer + "feedForward.0.weight"),
          layer + "feedForward.0.weight",
          new int[] {intermediate, hiddenSize});
      requireShape(
          weight(weights, layer + "feedForward.0.bias"),
          layer + "feedForward.0.bias",
          new int[] {intermediate});
      requireShape(
          weight(weights, layer + "feedForward.2.weight"),
          layer + "feedForward.2.weight",
          new int[] {hiddenSize, intermediate});
      requireShape(
          weight(weights, layer + "feedForward.2.bias"),
          layer + "feedForward.2.bias",
          new int[] {hiddenSize});
    }
    requireShape(
        weight(weights, "postLayernorm.weight"), "postLayernorm.weight", new int[] {hiddenSize});
    requireShape(
        weight(weights, "postLayernorm.bias"), "postLayernorm.bias", new int[] {hiddenSize});
    if (hiddenSize % numHeads != 0) {
      throw new IllegalArgumentException(
          "hidden size " + hiddenSize + " is not divisible by " + numHeads + " attention heads");
    }
    rejectUnexpected(weights, expectedKeys(numLayers));

    this.patchEmbedding =
        child(
            "patchEmbedding",
            new Conv2d(
                scope,
                convWeight,
                convBias,
                new int[] {patchSize, patchSize},
                new int[] {0, 0},
                new int[] {1, 1},
                1));
    this.positionEmbedding = child("positionEmbedding", new Embedding(scope, positionTable));
    int headDim = hiddenSize / numHeads;
    float scale = (float) Math.pow(headDim, -0.5);
    for (int i = 0; i < numLayers; i++) {
      String layer = "layer" + i + ".";
      layers.add(
          child(
              "layer" + i,
              new EncoderBlock(
                  scope,
                  new BidirectionalAttention(
                      scope,
                      numHeads,
                      headDim,
                      scale,
                      new Linear(
                          scope,
                          weight(weights, layer + "attention.query.weight"),
                          weight(weights, layer + "attention.query.bias")),
                      new Linear(
                          scope,
                          weight(weights, layer + "attention.key.weight"),
                          weight(weights, layer + "attention.key.bias")),
                      new Linear(
                          scope,
                          weight(weights, layer + "attention.value.weight"),
                          weight(weights, layer + "attention.value.bias")),
                      new Linear(
                          scope,
                          weight(weights, layer + "attention.output.weight"),
                          weight(weights, layer + "attention.output.bias"))),
                  new LayerNorm(
                      scope,
                      weight(weights, layer + "attentionNorm.weight"),
                      weight(weights, layer + "attentionNorm.bias"),
                      layerNormEps),
                  new Sequential(
                      scope,
                      new Linear(
                          scope,
                          weight(weights, layer + "feedForward.0.weight"),
                          weight(weights, layer + "feedForward.0.bias")),
                      new GELU(scope, true),
                      new Linear(
                          scope,
                          weight(weights, layer + "feedForward.2.weight"),
                          weight(weights, layer + "feedForward.2.bias"))),
                  new LayerNorm(
                      scope,
                      weight(weights, layer + "outputNorm.weight"),
                      weight(weights, layer + "outputNorm.bias"),
                      layerNormEps),
                  true)));
    }
    this.postLayernorm =
        child(
            "postLayernorm",
            new LayerNorm(
                scope,
                weight(weights, "postLayernorm.weight"),
                weight(weights, "postLayernorm.bias"),
                layerNormEps));
    this.connector = child("connector", connector);
    train(false);
  }

  /** The pixel-shuffle projection child; the model applies it to {@link #forward} output. */
  Idefics3Connector connector() {
    return connector;
  }

  /**
   * Runs the tower on one batch of tiles: patch convolution, position embedding, the pre-norm
   * encoder stack under the tile's binary patch-attention mask, and the post LayerNorm.
   *
   * <p>{@code pixels} is {@code [N, H, W, C]} (channels-last float) with {@code H} and {@code W}
   * multiples of the patch size and {@code C} the tower's input channel count; {@code patchMask} is
   * the shared P-by-P row-major 0/1 validity of the whole batch ({@link #patchMaskFromPixelMask}
   * produces it). Every activation is allocated into {@code pixels.scope()}, so that scope must be
   * this module's own scope or a descendant of it — the same contract as {@link T5Model}. A fully
   * padded tile must not reach this method: its all-zero mask row is rejected by the attention-mask
   * builder, and callers drop such tiles with {@link #isZeroTile} first.
   *
   * <p>This method deliberately stops at the post LayerNorm: it does not apply the {@code
   * connector} child, and the model applies {@code connector.forward(towerForwardOutput)} (see
   * {@link #connector()}) to the result to get the final {@code [N, (H/(patch*s)) * (W/(patch*s)),
   * textHidden]} text-space features.
   *
   * @return the post-LayerNorm hidden states, {@code [N, (H/patch) * (W/patch), hidden]}
   * @throws IllegalArgumentException if the pixel shape, dtype or patch mask is invalid
   */
  public MLXArray forward(MLXArray pixels, int[] patchMask) {
    Objects.requireNonNull(pixels, "pixels");
    Objects.requireNonNull(patchMask, "patchMask");
    if (!pixels.dtype().isInexact()) {
      throw new IllegalArgumentException("pixel values must be a floating tensor");
    }
    int[] shape = pixels.shape();
    if (shape.length != 4 || shape[3] != numChannels) {
      throw new IllegalArgumentException(
          "pixel values must be [N, H, W, "
              + numChannels
              + "] channels-last, got "
              + java.util.Arrays.toString(shape));
    }
    int ph = Math.floorDiv(shape[1], patchSize);
    int pw = Math.floorDiv(shape[2], patchSize);
    if (shape[1] % patchSize != 0 || shape[2] % patchSize != 0 || ph == 0 || pw == 0) {
      throw new IllegalArgumentException(
          "pixel extents "
              + shape[1]
              + "x"
              + shape[2]
              + " are not multiples of the patch size "
              + patchSize);
    }
    int numPatches = Math.multiplyExact(ph, pw);
    if (patchMask.length != numPatches) {
      throw new IllegalArgumentException(
          "the patch mask needs "
              + numPatches
              + " entries for a "
              + ph
              + "x"
              + pw
              + " patch grid, got "
              + patchMask.length);
    }
    for (int value : patchMask) {
      if (value != 0 && value != 1) {
        throw new IllegalArgumentException("the patch mask must be binary");
      }
    }

    MLXScope step = pixels.scope();
    // HF applies its channels-first convolution then flatten(2).transpose(1, 2); for the
    // contiguous channels-last layout the reshape alone yields the same [N, ph*pw, hidden]
    // row-major patch order.
    MLXArray tokens =
        MLXShape.reshape(patchEmbedding.forward(pixels), new int[] {shape[0], numPatches, hidden});
    int[] ids = positionIds(patchMask, ph, pw, positionSide);
    int[] flatIds = new int[Math.multiplyExact(shape[0], numPatches)];
    int[][] padding = new int[shape[0]][numPatches];
    for (int row = 0; row < shape[0]; row++) {
      System.arraycopy(ids, 0, flatIds, row * numPatches, numPatches);
      System.arraycopy(patchMask, 0, padding[row], 0, numPatches);
    }
    MLXArray positions =
        positionEmbedding.forward(MLX.array(step, flatIds, new int[] {shape[0], numPatches}));
    MLXArray x = MLXOps.add(tokens, positions);
    MLXArray mask = AttentionMask.bidirectional(step, padding, numPatches);
    for (EncoderBlock layer : layers) {
      x = layer.forward(x, mask, null);
    }
    return postLayernorm.forward(x);
  }

  MLXArray patchTokens(MLXArray pixels) {
    int[] shape = pixels.shape();
    return MLXShape.reshape(
        patchEmbedding.forward(pixels),
        new int[] {shape[0], shape[1] / patchSize * (shape[2] / patchSize), hidden});
  }

  /**
   * The variable-resolution patch position ids of {@code Idefics3VisionEmbeddings.forward} (pinned
   * lines 141-162): the valid patches' fractional coordinates {@code i / nb * (1 - 1e-6)} (float32
   * division first, then the float32 multiplication by {@code 0x3f7fffef}) are {@code
   * bucketize(right=True)}-ed against the boundaries {@code k / n, k = 1..n-1}, and patch {@code
   * (r, c)} maps to {@code bucketH * pw + bucketW}. Invalid patches keep the fill value 0.
   *
   * <p>Two deliberate deviations from a literal port, both pinned against the reference venv
   * (transformers 4.57.6 / torch 2.6.0 CPU, float32):
   *
   * <ul>
   *   <li>HF derives the boundary count and stride from the square config's {@code
   *       num_patches_per_side} while {@code nb_patches_h}/{@code nb_patches_w} come from the tile;
   *       this method generalizes both per axis to {@code ph} and {@code pw} (stride {@code pw}).
   *       Identical to HF whenever {@code ph == pw}, which the pinned fixtures use.
   *   <li>torch's CPU float32 {@code arange} runs through the SIMD {@code
   *       Vectorized<float>::arange} kernel, so no closed form reproduces its boundary bits
   *       exactly. The boundary used here is {@code (float)(1.0 / n + i * (1.0 / n))}
   *       (double-precision accumulation, float32 cast): bit-identical to torch for every dyadic
   *       {@code n} (including the pinned fixtures' {@code P = 4}) and off by one float32 ulp at 5
   *       of 35 boundaries for {@code P = 36} — a distance far below the {@code i / n * 1e-6} gap
   *       that keeps every fractional coordinate strictly inside its bucket, so no bucket
   *       assignment ever flips.
   * </ul>
   *
   * <p>Where HF counts valid patches per axis from column 0 and row 0, this method does the same; a
   * mask whose column 0 or row 0 is entirely padded would make HF's {@code position_ids[mask] =
   * pos_ids} assignment a shape mismatch, so this method degenerates to all zeros instead of
   * throwing. The per-({@code r}, {@code c}) evaluation likewise generalizes non-rectangular
   * validity masks, which HF likewise mishandles.
   *
   * @param patchMask the P-by-P row-major 0/1 patch validity
   * @param ph the number of patch rows of the tile
   * @param pw the number of patch columns of the tile
   * @return the {@code ph * pw} position ids, row-major, 0 at invalid patches
   * @throws IllegalArgumentException if the dimensions or mask are invalid
   */
  public static int[] positionIds(int[] patchMask, int ph, int pw) {
    return positionIds(patchMask, ph, pw, ph, pw);
  }

  static int[] positionIds(int[] patchMask, int ph, int pw, int positionSide) {
    return positionIds(patchMask, ph, pw, positionSide, positionSide);
  }

  private static int[] positionIds(
      int[] patchMask, int ph, int pw, int positionRows, int positionCols) {
    Objects.requireNonNull(patchMask, "patchMask");
    if (ph <= 0 || pw <= 0) {
      throw new IllegalArgumentException("the patch grid dimensions must be positive");
    }
    int numPatches = Math.multiplyExact(ph, pw);
    if (patchMask.length != numPatches) {
      throw new IllegalArgumentException(
          "the patch mask needs "
              + numPatches
              + " entries for a "
              + ph
              + "x"
              + pw
              + " patch grid, got "
              + patchMask.length);
    }
    for (int value : patchMask) {
      if (value != 0 && value != 1) {
        throw new IllegalArgumentException("the patch mask must be binary");
      }
    }

    int nbH = 0;
    int nbW = 0;
    for (int r = 0; r < ph; r++) {
      nbH += patchMask[r * pw];
    }
    for (int c = 0; c < pw; c++) {
      nbW += patchMask[c];
    }
    int[] rowBucket = new int[ph];
    int[] colBucket = new int[pw];
    if (nbH > 0 && nbW > 0) {
      for (int r = 0; r < ph; r++) {
        rowBucket[r] = bucketOf(r, nbH, positionRows);
      }
      for (int c = 0; c < pw; c++) {
        colBucket[c] = bucketOf(c, nbW, positionCols);
      }
    }
    int[] ids = new int[numPatches];
    for (int r = 0; r < ph; r++) {
      for (int c = 0; c < pw; c++) {
        ids[r * pw + c] =
            patchMask[r * pw + c] == 1 ? rowBucket[r] * positionCols + colBucket[c] : 0;
      }
    }
    return ids;
  }

  /** The {@code bucketize(right=True)} bucket of coordinate {@code i / nb} among {@code n}. */
  private static int bucketOf(int i, int nb, int n) {
    // HF evaluates h / nb * (1 - 1e-6) left to right: float32 division, then float32
    // multiplication by the float64-rounded-then-cast scalar (all three float32 candidates
    // for 1 - 1e-6 share the bits 0x3f7fffef; verified against the reference venv).
    float fraction = (float) i / (float) nb * FRACTIONAL_MULTIPLIER;
    int bucket = 0;
    for (int b = 0; b < n - 1; b++) {
      // torch.bucketize(x, boundaries, right=True) counts the boundaries <= x; no fraction ever
      // equals a boundary (the multiplier keeps each i / nb strictly below it), so the strict
      // count is identical.
      if (boundary(b, n) < fraction) {
        bucket++;
      }
    }
    return bucket;
  }

  /** The {@code (b + 1) / n} boundary, computed as the best closed form for torch's arange. */
  private static float boundary(int b, int n) {
    return (float) (1.0 / n + (double) b * (1.0 / n));
  }

  /**
   * The patch-attention mask of {@code Idefics3Model.get_image_features} (pinned lines 643-645):
   * the pixel mask is unfolded into non-overlapping {@code patch}-by-{@code patch} subgrids and a
   * patch is valid when <em>any</em> of its pixels is non-zero ({@code sum > 0}), not all.
   *
   * @param pixelMask the {@code h * w} row-major 0/1 pixel validity
   * @param h the pixel rows
   * @param w the pixel columns
   * @param patch the square patch size
   * @return the {@code (h / patch) * (w / patch)} row-major 0/1 patch validity
   * @throws IllegalArgumentException if the extents are not patch multiples or the mask is invalid
   */
  public static int[] patchMaskFromPixelMask(int[] pixelMask, int h, int w, int patch) {
    Objects.requireNonNull(pixelMask, "pixelMask");
    if (patch <= 0 || h <= 0 || w <= 0) {
      throw new IllegalArgumentException("the pixel extents and patch size must be positive");
    }
    if (h % patch != 0 || w % patch != 0) {
      throw new IllegalArgumentException(
          "the pixel extents " + h + "x" + w + " are not multiples of the patch size " + patch);
    }
    if (pixelMask.length != Math.multiplyExact(h, w)) {
      throw new IllegalArgumentException(
          "the pixel mask needs "
              + Math.multiplyExact(h, w)
              + " entries for "
              + h
              + "x"
              + w
              + " pixels, got "
              + pixelMask.length);
    }
    for (int value : pixelMask) {
      if (value != 0 && value != 1) {
        throw new IllegalArgumentException("the pixel mask must be binary");
      }
    }
    int ph = h / patch;
    int pw = w / patch;
    int[] mask = new int[Math.multiplyExact(ph, pw)];
    for (int r = 0; r < ph; r++) {
      for (int c = 0; c < pw; c++) {
        boolean any = false;
        for (int dr = 0; dr < patch && !any; dr++) {
          int base = (r * patch + dr) * w + c * patch;
          for (int dc = 0; dc < patch; dc++) {
            if (pixelMask[base + dc] != 0) {
              any = true;
              break;
            }
          }
        }
        mask[r * pw + c] = any ? 1 : 0;
      }
    }
    return mask;
  }

  /**
   * Whether a tile is the all-zero padding image that {@code Idefics3Model.get_image_features}
   * drops before the vision encoder (pinned lines 625-628): padding iff every pixel value is
   * exactly {@code 0.0f} ({@code -0.0f} counts as zero, as in the reference; a {@code NaN} pixel
   * makes the tile real, matching {@code == 0.0}). An empty tile is padding.
   *
   * @param values the {@code H * W * C} flattened tile pixel values
   * @return true iff every value is exactly {@code 0.0f}
   */
  public static boolean isZeroTile(float[] values) {
    Objects.requireNonNull(values, "values");
    for (float value : values) {
      if (value != 0.0f) {
        return false;
      }
    }
    return true;
  }

  /**
   * The gather indices that merge image features into token embeddings exactly as {@code
   * Idefics3Model.inputs_merger} (pinned lines 604-606) does with {@code
   * inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)}.
   *
   * <p>{@code masked_scatter} is a scatter, not an insertion: feature row {@code f} overwrites
   * exactly the {@code f}-th image-token position and every other position keeps its own embedding.
   * The returned {@code expandedLength}-index array therefore gathers, from a source whose first
   * {@code expandedLength} rows are the original embeddings followed by the {@code
   * featureDestinations.size()} feature rows, the merged sequence: position {@code i} reads {@code
   * expandedLength + f} when {@code i} is the {@code f}-th destination and {@code i} otherwise.
   *
   * @param expandedLength the merged sequence length
   * @param featureDestinations the image-token positions, strictly increasing in {@code [0,
   *     expandedLength)}
   * @return the {@code expandedLength} gather indices into {@code [0, expandedLength + F)}
   * @throws IllegalArgumentException if a destination is out of range or not strictly increasing
   */
  public static int[] mergeGatherIndices(int expandedLength, List<Integer> featureDestinations) {
    Objects.requireNonNull(featureDestinations, "featureDestinations");
    if (expandedLength < 0) {
      throw new IllegalArgumentException("expandedLength must be non-negative");
    }
    int features = featureDestinations.size();
    int previous = -1;
    for (int f = 0; f < features; f++) {
      int destination = featureDestinations.get(f);
      if (destination < 0 || destination >= expandedLength) {
        throw new IllegalArgumentException(
            "feature destination " + destination + " is out of range [0, " + expandedLength + ")");
      }
      if (destination <= previous) {
        throw new IllegalArgumentException(
            "feature destinations must be strictly increasing; index "
                + f
                + " has "
                + destination
                + " after "
                + previous);
      }
      previous = destination;
    }
    Math.addExact(expandedLength, features);
    int[] indices = new int[expandedLength];
    int next = 0;
    for (int i = 0; i < expandedLength; i++) {
      if (next < features && featureDestinations.get(next) == i) {
        indices[i] = expandedLength + next;
        next++;
      } else {
        indices[i] = i;
      }
    }
    return indices;
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
                + java.util.Arrays.toString(expected)
                + ", got "
                + java.util.Arrays.toString(array.shape()));
      }
    }
  }

  private static java.util.Set<String> expectedKeys(int numLayers) {
    java.util.Set<String> expected = new TreeSet<>();
    expected.add("patchEmbedding.weight");
    expected.add("patchEmbedding.bias");
    expected.add("positionEmbedding.weight");
    expected.add("postLayernorm.weight");
    expected.add("postLayernorm.bias");
    for (int i = 0; i < numLayers; i++) {
      String layer = "layer" + i + ".";
      for (String name : new String[] {"query", "key", "value", "output"}) {
        expected.add(layer + "attention." + name + ".weight");
        expected.add(layer + "attention." + name + ".bias");
      }
      expected.add(layer + "attentionNorm.weight");
      expected.add(layer + "attentionNorm.bias");
      expected.add(layer + "outputNorm.weight");
      expected.add(layer + "outputNorm.bias");
      expected.add(layer + "feedForward.0.weight");
      expected.add(layer + "feedForward.0.bias");
      expected.add(layer + "feedForward.2.weight");
      expected.add(layer + "feedForward.2.bias");
    }
    return expected;
  }

  private static void rejectUnexpected(
      Map<String, MLXArray> weights, java.util.Set<String> expected) {
    java.util.Set<String> extra = new TreeSet<>(weights.keySet());
    extra.removeAll(expected);
    if (!extra.isEmpty()) {
      throw new IllegalArgumentException("unexpected vision tower weights: " + extra);
    }
  }
}
