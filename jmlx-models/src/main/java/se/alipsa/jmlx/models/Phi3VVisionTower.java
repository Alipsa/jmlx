package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.BidirectionalAttention;
import se.alipsa.jmlx.nn.Conv2d;
import se.alipsa.jmlx.nn.EncoderBlock;
import se.alipsa.jmlx.nn.GELU;
import se.alipsa.jmlx.nn.LayerNorm;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.QuickGELU;
import se.alipsa.jmlx.nn.Sequential;

/**
 * The Phi-3.5-Vision CLIP vision tower: the bias-free patch convolution, the class token, the
 * single 577-row learned position table, the pre-encoder LayerNorm (checkpoint key {@code
 * pre_layrnorm} — HF's typo) and the pre-LN encoder blocks of the pinned {@code
 * openai/clip-vit-large-patch14-336} CLIP configuration (see {@code
 * tools/hf-reference/provenance.json}), ported from the bundled {@code modeling_phi3_v.py} {@code
 * Phi3ImageEmbedding.get_img_features} (pinned lines 208-219) and the pinned Hugging Face
 * Transformers 4.57.6 {@code modeling_clip.py} — {@code CLIPVisionEmbeddings} (131-211), {@code
 * CLIPEncoderLayer} (368-415) and {@code CLIPEncoder} (485-568). The CLIP constants (hidden 1024,
 * 16 heads, head dim 64, patch 14, 336 pixels, 577 positions, {@code layer_norm_eps = 1e-05}) are
 * the pinned config's, not fields of {@code config.json}.
 *
 * <p>The module child names mirror the HF checkpoint tree: {@code patchEmbedding} is {@code
 * embeddings.patch_embedding} (weight transposed to jmlx's channels-last {@code [hidden, patch,
 * patch, channels]} layout at load time), {@code positionTable} is {@code embeddings.
 * position_embedding} (a plain table, broadcast-added over all 577 positions — identical to HF's
 * {@code position_ids} lookup over its full table), {@code layer0..layerN-2} are {@code
 * encoder.layers.{i}} (with {@code attention.{query,key,value,output}} for {@code
 * self_attn.{q,k,v,out}_proj}, {@code attentionNorm}/{@code outputNorm} for {@code
 * layer_norm1}/{@code layer_norm2} and {@code feedForward.0/.2} for {@code mlp.fc1/fc2} with the
 * QuickGELU activation), {@code preLayernorm} is {@code pre_layrnorm}, and {@code projector} is the
 * {@code img_projection} MLP (exact GELU). The last checkpoint layer (its output is {@code
 * hidden_states[-1]}) and the pooled-output {@code post_layernorm} are loaded but never executed:
 * with {@code output_hidden_states=True} and {@code layer_idx = -2}, the features are the state
 * after block {@code N-2}, and the class row is dropped by {@code type_feature = patch}.
 *
 * <p>{@link #forward} returns the per-slot patch features {@code [S, 576, 1024]} (class row
 * removed); {@link #hdRows} then runs the reference's high-definition transform — the global slot's
 * 2x2 merge and newline, the sub-crop 2x2 merge and newline, and the per-image {@code [sub,
 * separator, global]} row order with the learnable {@code glb_GN} separator — into the one {@code
 * [numImgTokens, 4096]} block the projector consumes. Each forward stage (the embedding stage, then
 * one per CLIP block) is evaluated and released as it completes; only the stage's surviving
 * activation is hoisted into the input's scope ({@code pixels.scope()}).
 */
final class Phi3VVisionTower extends Module {

  private static final int HIDDEN = 1024;
  private static final int HEADS = 16;
  private static final int HEAD_DIM = 64;
  private static final int PATCH = 14;
  private static final int IMAGE = 336;
  private static final int POSITIONS = (IMAGE / PATCH) * (IMAGE / PATCH) + 1;
  private static final int PATCHES = POSITIONS - 1;
  private static final float LAYERNORM_EPS = 1e-5f;
  private static final int MERGED = 4 * HIDDEN;

  private final Conv2d patchEmbedding;
  private final LayerNorm preLayernorm;
  private final List<EncoderBlock> layers = new ArrayList<>();
  private final Sequential projector;

  /**
   * Builds the tower from checkpoint weights keyed by this module's own dotted parameter paths (see
   * the class javadoc for the HF key mapping). {@code clipLayers} is the checkpoint's CLIP layer
   * count; block {@code clipLayers - 1} is plan-optional and never extracted here. Every weight
   * must be present and shaped consistently; any extra or missing key is rejected.
   *
   * @param clipLayers the checkpoint's CLIP encoder layer count, at least one
   * @param textHidden the decoder's hidden size (the projector's output width)
   * @param intermediate the CLIP MLP width, from the checkpoint's first {@code mlp.fc1.weight}
   * @throws IllegalArgumentException if a dimension is invalid, a weight is missing or extra, or
   *     the weights are mutually inconsistent
   * @throws NullPointerException if {@code weights} is {@code null}
   */
  Phi3VVisionTower(
      MLXScope scope,
      int clipLayers,
      int textHidden,
      int intermediate,
      Map<String, MLXArray> weights) {
    super(scope);
    Objects.requireNonNull(weights, "weights");
    if (clipLayers < 1) {
      throw new IllegalArgumentException("the vision tower needs at least one CLIP layer");
    }
    if (textHidden <= 0) {
      throw new IllegalArgumentException("textHidden must be positive");
    }
    if (intermediate <= 0) {
      throw new IllegalArgumentException("intermediate must be positive");
    }
    requireShape(weight(weights, "classToken"), "classToken", new int[] {HIDDEN});
    requireShape(
        weight(weights, "patchEmbedding.weight"),
        "patchEmbedding.weight",
        new int[] {HIDDEN, PATCH, PATCH, 3});
    requireShape(weight(weights, "positionTable"), "positionTable", new int[] {POSITIONS, HIDDEN});
    requireShape(weight(weights, "preLayernorm.weight"), "preLayernorm.weight", new int[] {HIDDEN});
    requireShape(weight(weights, "preLayernorm.bias"), "preLayernorm.bias", new int[] {HIDDEN});
    for (int i = 0; i < clipLayers - 1; i++) {
      String layer = "layer" + i + ".";
      for (String name : new String[] {"query", "key", "value", "output"}) {
        requireShape(
            weight(weights, layer + "attention." + name + ".weight"),
            layer + "attention." + name + ".weight",
            new int[] {HIDDEN, HIDDEN});
        requireShape(
            weight(weights, layer + "attention." + name + ".bias"),
            layer + "attention." + name + ".bias",
            new int[] {HIDDEN});
      }
      for (String norm : new String[] {"attentionNorm", "outputNorm"}) {
        requireShape(
            weight(weights, layer + norm + ".weight"),
            layer + norm + ".weight",
            new int[] {HIDDEN});
        requireShape(
            weight(weights, layer + norm + ".bias"), layer + norm + ".bias", new int[] {HIDDEN});
      }
      requireShape(
          weight(weights, layer + "feedForward.0.weight"),
          layer + "feedForward.0.weight",
          new int[] {intermediate, HIDDEN});
      requireShape(
          weight(weights, layer + "feedForward.0.bias"),
          layer + "feedForward.0.bias",
          new int[] {intermediate});
      requireShape(
          weight(weights, layer + "feedForward.2.weight"),
          layer + "feedForward.2.weight",
          new int[] {HIDDEN, intermediate});
      requireShape(
          weight(weights, layer + "feedForward.2.bias"),
          layer + "feedForward.2.bias",
          new int[] {HIDDEN});
    }
    requireShape(weight(weights, "glbGN"), "glbGN", new int[] {1, 1, MERGED});
    requireShape(weight(weights, "subGN"), "subGN", new int[] {1, 1, 1, MERGED});
    requireShape(
        weight(weights, "projector.0.weight"),
        "projector.0.weight",
        new int[] {textHidden, MERGED});
    requireShape(weight(weights, "projector.0.bias"), "projector.0.bias", new int[] {textHidden});
    requireShape(
        weight(weights, "projector.2.weight"),
        "projector.2.weight",
        new int[] {textHidden, textHidden});
    requireShape(weight(weights, "projector.2.bias"), "projector.2.bias", new int[] {textHidden});
    rejectUnexpected(weights, expectedKeys(clipLayers));

    this.patchEmbedding =
        child(
            "patchEmbedding",
            new Conv2d(
                scope,
                weight(weights, "patchEmbedding.weight"),
                null,
                new int[] {PATCH, PATCH},
                new int[] {0, 0},
                new int[] {1, 1},
                1));
    this.preLayernorm =
        child(
            "preLayernorm",
            new LayerNorm(
                scope,
                weight(weights, "preLayernorm.weight"),
                weight(weights, "preLayernorm.bias"),
                LAYERNORM_EPS));
    float scale = (float) Math.pow(HEAD_DIM, -0.5);
    for (int i = 0; i < clipLayers - 1; i++) {
      String layer = "layer" + i + ".";
      layers.add(
          child(
              "layer" + i,
              new EncoderBlock(
                  scope,
                  new BidirectionalAttention(
                      scope,
                      HEADS,
                      HEAD_DIM,
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
                      LAYERNORM_EPS),
                  new Sequential(
                      scope,
                      new Linear(
                          scope,
                          weight(weights, layer + "feedForward.0.weight"),
                          weight(weights, layer + "feedForward.0.bias")),
                      new QuickGELU(scope),
                      new Linear(
                          scope,
                          weight(weights, layer + "feedForward.2.weight"),
                          weight(weights, layer + "feedForward.2.bias"))),
                  new LayerNorm(
                      scope,
                      weight(weights, layer + "outputNorm.weight"),
                      weight(weights, layer + "outputNorm.bias"),
                      LAYERNORM_EPS),
                  true)));
    }
    this.projector =
        child(
            "projector",
            new Sequential(
                scope,
                new Linear(
                    scope,
                    weight(weights, "projector.0.weight"),
                    weight(weights, "projector.0.bias")),
                new GELU(scope),
                new Linear(
                    scope,
                    weight(weights, "projector.2.weight"),
                    weight(weights, "projector.2.bias"))));
    param("classToken", weight(weights, "classToken"));
    param("positionTable", weight(weights, "positionTable"));
    param("glbGN", weight(weights, "glbGN"));
    param("subGN", weight(weights, "subGN"));
    train(false);
  }

  /** The {@code img_projection} MLP child; the model applies it to {@link #hdRows} output. */
  Sequential projector() {
    return projector;
  }

  /**
   * Runs the CLIP tower over one image's real slots: patch convolution, class token, the
   * broadcast-added position table, the pre-encoder LayerNorm and the executed pre-LN blocks under
   * the all-true attention mask (every slot's 577 positions are valid).
   *
   * <p>{@code pixels} is {@code [S, 336, 336, 3]} channels-last float (slot 0 the global image,
   * slots 1..S-1 the real sub-crops — zero-padded slots are never computed, since the reference
   * computes them but slices their rows away); each stage's intermediates are released as its scope
   * closes, and only the stage's surviving activation is hoisted into {@code pixels.scope()}.
   *
   * @return the per-slot patch features {@code [S, 576, 1024]}, the class row removed
   * @throws IllegalArgumentException if the pixel shape or dtype is invalid
   */
  MLXArray forward(MLXArray pixels) {
    Objects.requireNonNull(pixels, "pixels");
    if (!pixels.dtype().isInexact()) {
      throw new IllegalArgumentException("pixel values must be a floating tensor");
    }
    int[] shape = pixels.shape();
    if (shape.length != 4 || shape[1] != IMAGE || shape[2] != IMAGE || shape[3] != 3) {
      throw new IllegalArgumentException(
          "pixel values must be [S, 336, 336, 3] channels-last, got "
              + java.util.Arrays.toString(shape));
    }
    int slots = shape[0];
    MLXScope scope = pixels.scope();
    // The embedding stage (conv -> class concatenation -> position add -> pre-encoder norm) runs
    // in its own child scope: only x is needed after it, so evaluate it and hoist it to the
    // caller's scope before the stage's scope closes, releasing every other stage intermediate
    // before block 0 starts (the 7.3b WP5 per-stage evaluate-and-release pattern, findings 7.3b
    // §11.4).
    MLXArray x;
    try (MLXScope embed = scope.newChild()) {
      // HF applies its channels-first convolution then flatten(2).transpose(1, 2); for the
      // contiguous channels-last layout the reshape alone yields the same [S, 576, hidden]
      // row-major patch order.
      MLXArray patches =
          MLXShape.reshape(patchEmbedding.forward(pixels), new int[] {slots, PATCHES, HIDDEN});
      MLXArray cls = MLXShape.broadcastTo(param("classToken"), new int[] {slots, 1, HIDDEN});
      x =
          MLXOps.add(
              MLXShape.concatenate(new MLXArray[] {cls, patches}, 1),
              MLXShape.broadcastTo(param("positionTable"), new int[] {slots, POSITIONS, HIDDEN}));
      x = preLayernorm.forward(x);
      MLX.eval(x);
      x = MLX.hoist(x, scope);
    }
    // An explicitly all-true mask is numerically identical to HF passing None.
    MLXArray mask = MLX.full(scope, new int[] {slots, 1, POSITIONS, POSITIONS}, 1f, DType.BOOL);
    for (EncoderBlock layer : layers) {
      try (MLXScope block = scope.newChild()) {
        // Pin this block's entry row into the block scope the way Qwen3VlVisionTower does: the
        // same-shape broadcast view is a no-op that allocates a handle into `block`, so every
        // op result of the block resolves into it (the weights live in an ancestor scope).
        MLXArray in = MLXShape.broadcastTo(x, block, x.shape());
        MLXArray out = layer.forward(in, mask, null);
        // Evaluate this block before the next one starts: closing the block scope below frees
        // every intermediate of this block (the peak does not accumulate across blocks), while
        // `out` is hoisted out first, so the computation is unchanged (uncompiled MLX does not
        // fuse across eval boundaries; findings 7.3b §11.4).
        MLX.eval(out);
        x.close();
        x = MLX.hoist(out, scope);
      }
    }
    // hidden_states[-2] is the state after the last executed block; type_feature "patch" drops
    // the class row.
    return MLXShape.slice(x, new int[] {0, 1, 0}, new int[] {slots, POSITIONS, HIDDEN});
  }

  /**
   * The reference's {@code hd_feature_transform} (pinned lines 250-299) for one image: the global
   * slot's 2x2 merge and newline, the sub-crop 2x2 merge and newline, and the per-image {@code [sub
   * rows, glb_GN separator, global rows]} row order.
   *
   * @param features this image's tower output, {@code [cropBlocks + 1, 576, 1024]} (slot 0 the
   *     global image)
   * @param gridH the image's crop grid rows ({@code paddedHeight / 336})
   * @param gridW the image's crop grid columns ({@code paddedWidth / 336})
   * @return the {@code (cropBlocks + 1) * 144 + 1 + (gridH + 1) * 12} rows of width 4096, in the
   *     reference's concatenated order
   * @throws IllegalArgumentException if the feature shape disagrees with the grid
   */
  MLXArray hdRows(MLXArray features, int gridH, int gridW) {
    Objects.requireNonNull(features, "features");
    if (gridH <= 0 || gridW <= 0) {
      throw new IllegalArgumentException("the crop grid must be positive: " + gridH + "x" + gridW);
    }
    int[] shape = features.shape();
    int cropBlocks = gridH * gridW;
    if (shape.length != 3
        || shape[0] != cropBlocks + 1
        || shape[1] != PATCHES
        || shape[2] != HIDDEN) {
      throw new IllegalArgumentException(
          "features must be [cropBlocks + 1, 576, 1024] for a "
              + gridH
              + "x"
              + gridW
              + " grid, got "
              + java.util.Arrays.toString(shape));
    }
    MLXArray global = MLXShape.slice(features, new int[] {0, 0, 0}, new int[] {1, PATCHES, HIDDEN});
    MLXArray globalNewline = addNewline(reshapeHd2x2Merge(global, 1, 1));
    MLXArray sub =
        MLXShape.slice(features, new int[] {1, 0, 0}, new int[] {cropBlocks + 1, PATCHES, HIDDEN});
    MLXArray subNewline = addNewline(reshapeHd2x2Merge(sub, gridH, gridW));
    // [sub features, separator, global features], exactly the reference's extend order — all
    // three blocks 2D row matrices, the reference's glb_GN.squeeze(0) and per-image index.
    return MLXShape.concatenate(
        new MLXArray[] {
          MLXShape.squeeze(subNewline, new int[] {0}),
          MLXShape.squeeze(param("glbGN"), new int[] {0}),
          MLXShape.squeeze(globalNewline, new int[] {0})
        },
        0);
  }

  /**
   * The reference's {@code reshape_hd_patches_2x2merge} (pinned lines 301-345): pure reshapes and
   * transposes merging each 24x24 patch grid's 2x2 spatial neighborhood into the channel dimension,
   * {@code [N, 576, 1024] -> [N / (h*w), h * 12, w * 12, 4096]}.
   */
  private static MLXArray reshapeHd2x2Merge(MLXArray features, int rows, int cols) {
    int[] shape = features.shape();
    int n = shape[0];
    if (n % (rows * cols) != 0) {
      throw new IllegalArgumentException(
          "the " + n + " feature slots do not form a whole " + rows + "x" + cols + " crop grid");
    }
    int images = n / (rows * cols);
    MLXArray merged =
        MLXShape.transpose(
            MLXShape.reshape(
                MLXShape.reshape(features, new int[] {n, 24, 24, HIDDEN}),
                new int[] {n, 12, 2, 12, 2, HIDDEN}),
            new int[] {0, 1, 3, 2, 4, 5});
    merged = MLXShape.reshape(merged, new int[] {images, rows, cols, 12, 12, MERGED});
    merged = MLXShape.transpose(merged, new int[] {0, 1, 3, 2, 4, 5});
    return MLXShape.reshape(merged, new int[] {images, rows * 12, cols * 12, MERGED});
  }

  /**
   * The reference's {@code add_image_newline} (pinned lines 347-358): the learnable {@code sub_GN}
   * row appended after every HD row, {@code [n, h, w, 4096] -> [n, h * (w + 1), 4096]}.
   */
  private MLXArray addNewline(MLXArray features) {
    int[] shape = features.shape();
    MLXArray newline =
        MLXShape.broadcastTo(param("subGN"), new int[] {shape[0], shape[1], 1, MERGED});
    return MLXShape.reshape(
        MLXShape.concatenate(new MLXArray[] {features, newline}, 2),
        new int[] {shape[0], shape[1] * (shape[2] + 1), MERGED});
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

  private static TreeSet<String> expectedKeys(int clipLayers) {
    TreeSet<String> expected = new TreeSet<>();
    expected.add("classToken");
    expected.add("patchEmbedding.weight");
    expected.add("positionTable");
    expected.add("preLayernorm.weight");
    expected.add("preLayernorm.bias");
    for (int i = 0; i < clipLayers - 1; i++) {
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
    expected.add("glbGN");
    expected.add("subGN");
    expected.add("projector.0.weight");
    expected.add("projector.0.bias");
    expected.add("projector.2.weight");
    expected.add("projector.2.bias");
    return expected;
  }

  private static void rejectUnexpected(Map<String, MLXArray> weights, TreeSet<String> expected) {
    TreeSet<String> extra = new TreeSet<>(weights.keySet());
    extra.removeAll(expected);
    if (!extra.isEmpty()) {
      throw new IllegalArgumentException("unexpected vision tower weights: " + extra);
    }
  }
}
