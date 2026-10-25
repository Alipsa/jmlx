package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.nn.MultiAxisRope;
import se.alipsa.jmlx.nn.MultiAxisRotation;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.TokenizerException;
import se.alipsa.jmlx.vision.Qwen3VlGeometryPlan;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessor;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;

/**
 * Local Qwen3-VL image-and-text generation. The HF float layout loads every weight up-cast to
 * FLOAT32 (a BF16 checkpoint therefore occupies roughly twice its source weight memory after
 * loading); the MLX affine-quantized layout loads the packed U32 text projections straight into the
 * packed decoder path, keeping their BF16/FP16 {@code .scales}/{@code .biases} sidecars up-cast to
 * FLOAT32, up-casts every other text weight (embedding, norm) to FLOAT32, and up-casts the
 * all-float vision tower to FLOAT32 (its conv weight already arrives channels-last and is never
 * transposed). Images are processed once during prefill and run through the vision tower there;
 * each DeepStack level is injected after its configured text block in the same prefill, and the
 * cached decode retains only the self-attention KV plus the per-step multi-axis rotation derived
 * from the request's expansion plan.
 *
 * <p>Checkpoint provenance is selected from the shard headers' {@code __metadata__} blocks and the
 * config before any tensor is loaded (see {@link Qwen3VlProvenance}): the HF float layout and the
 * MLX affine-quantized layout are both recognized, and every other combination is rejected by name.
 * The load then parses the config strictly ({@link Qwen3VlDescriptor}), cross-checks the image
 * processor config against the vision config, cross-checks the tokenizer's sentinel spellings
 * against the config's declared token IDs, and only then runs the combined tensor preflight and
 * load, the FLOAT32 up-cast, the text/vision split and the module construction.
 *
 * <p>The caller owns the loading scope. Requests are batch one and follow core's single platform
 * thread rule. Full and capacity-bounded full caches are supported; sliding retention is rejected.
 * Synthetic correctness does not imply Tier-B support.
 */
public final class Qwen3VlModel implements TextGenerationModel {
  private static final String LANGUAGE_PREFIX = "model.language_model.";

  /** The text prefix the real MLX-converted artifact carries on every text tensor key. */
  static final String AFFINE_TEXT_PREFIX = "language_model.";

  private final Qwen3VlDescriptor descriptor;
  private final TextDecoder decoder;
  private final Qwen3VlVisionTower tower;
  private final Qwen3VlImageProcessor processor;
  private final Qwen3VlPromptTokens tokens;
  private final MultiAxisRope mrope;
  private final ModelMetadata metadata;
  private Consumer<String> observer = ignored -> {};

  /**
   * The text-side decoder: a private {@link DecoderModel} built from the strict text descriptor and
   * the rekeyed {@code model.*} text weights. It is deliberately not exposed: {@link Qwen3VlModel}
   * composes it with the vision tower, and the composite (not the bare decoder) is what callers and
   * the scheduler see.
   */
  private static final class TextDecoder extends DecoderModel {
    TextDecoder(MLXScope scope, ArchitectureDescriptor descriptor, Map<String, MLXArray> weights) {
      super(scope, descriptor, weights);
    }
  }

  private Qwen3VlModel(
      Qwen3VlDescriptor descriptor,
      TextDecoder decoder,
      Qwen3VlVisionTower tower,
      Qwen3VlImageProcessor processor,
      Qwen3VlPromptTokens tokens) {
    this.descriptor = descriptor;
    this.decoder = decoder;
    this.tower = tower;
    this.processor = processor;
    this.tokens = tokens;
    this.mrope = descriptor.multiAxisRope();
    this.metadata =
        new VisionLanguageMetadata(
            "qwen3_vl",
            decoder.config().vocabSize(),
            decoder.config().numHiddenLayers(),
            descriptor.depth());
  }

  /** Loads the checkpoint, tokenizer sentinels and image processor from one directory. */
  public static Qwen3VlModel load(MLXScope scope, Path directory) throws IOException {
    return load(scope, directory, JsonFiles.read(directory.resolve("config.json")));
  }

  /**
   * The package entry the dispatch uses, passing the {@code config.json} tree it already read so
   * the file is parsed once.
   *
   * <p>Load order: provenance selection (headers plus the quantization declarations), strict config
   * parse (the affine parse carries the packed plan's group size and bit width), processor
   * cross-check, tokenizer sentinel resolution, combined tensor preflight and load, the dtype
   * up-cast (the affine layout keeps its packed U32 weights as-is), the text/vision split with
   * shape and alias validation, and finally the module construction on the caller's scope.
   *
   * @throws IOException if a file cannot be read
   * @throws IllegalArgumentException when the provenance, config, processor config, tokenizer,
   *     tensors or shapes disagree
   */
  static Qwen3VlModel load(MLXScope scope, Path directory, JsonNode root) throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(root, "root");
    // Provenance selection reads the raw config tree (the quantization declaration is an outer
    // inert key the strict descriptor parse strips, so it must be read before the parse).
    Qwen3VlProvenance.Resolved provenance = Qwen3VlProvenance.resolve(directory, root);
    boolean affine = provenance.kind() == Qwen3VlProvenance.Kind.MLX_AFFINE;
    Qwen3VlDescriptor d =
        affine
            ? Qwen3VlDescriptor.parseAffine(root, provenance.groupSize(), provenance.bits())
            : Qwen3VlDescriptor.parse(root);
    Qwen3VlProcessorConfig processorConfig =
        Qwen3VlProcessorConfig.fromJson(
            Files.readAllBytes(directory.resolve("preprocessor_config.json")));
    if (processorConfig.patchSize() != d.patchSize()) {
      throw new IllegalArgumentException(
          "preprocessor_config.json patch_size "
              + processorConfig.patchSize()
              + " must equal vision_config patch_size "
              + d.patchSize());
    }
    if (processorConfig.temporalPatchSize() != d.temporalPatchSize()) {
      throw new IllegalArgumentException(
          "preprocessor_config.json temporal_patch_size "
              + processorConfig.temporalPatchSize()
              + " must equal vision_config temporal_patch_size "
              + d.temporalPatchSize());
    }
    if (processorConfig.mergeSize() != d.mergeSize()) {
      throw new IllegalArgumentException(
          "preprocessor_config.json merge_size "
              + processorConfig.mergeSize()
              + " must equal vision_config spatial_merge_size "
              + d.mergeSize());
    }
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    int vocab = d.text().dimensions().vocabSize();
    if (tokenizer.vocabSize() > vocab) {
      throw new IllegalArgumentException(
          "tokenizer vocabulary size "
              + tokenizer.vocabSize()
              + " exceeds checkpoint vocabulary "
              + vocab);
    }
    Qwen3VlPromptTokens resolved = resolveTokens(tokenizer, d.tokens());
    if (!resolved.equals(d.tokens())) {
      throw new IllegalArgumentException(
          "tokenizer sentinel spellings do not match the config's declared token IDs");
    }
    TensorPlan plan =
        affine ? d.affineTensorPlan(provenance.groupSize(), provenance.bits()) : d.tensorPlan();
    Map<String, MLXArray> loaded = CheckpointLoader.load(scope, directory, plan);
    Map<String, MLXArray> weights = affine ? upcastFloatsKeepPacked(loaded) : upcastAll(loaded);
    Map<String, MLXArray> textWeights = new LinkedHashMap<>();
    if (affine) {
      for (String key : weights.keySet()) {
        if (key.startsWith(AFFINE_TEXT_PREFIX)) {
          textWeights.put(key.substring(AFFINE_TEXT_PREFIX.length()), weights.get(key));
        }
      }
      ArchitectureMappings.tensorPlan(d.text()).validate(textWeights.keySet());
      validateQuantizedTextShapes(d, textWeights, provenance.groupSize(), provenance.bits());
    } else {
      for (String key : weights.keySet()) {
        if (key.startsWith(LANGUAGE_PREFIX)) {
          textWeights.put("model." + key.substring(LANGUAGE_PREFIX.length()), weights.get(key));
        } else if (key.equals("lm_head.weight")) {
          textWeights.put(key, weights.get(key));
        }
      }
      ArchitectureMappings.tensorPlan(d.text()).validate(textWeights.keySet());
      validateTextShapes(d, textWeights);
    }
    checkTiedLmHeadAlias(d, textWeights, affine);
    Map<String, MLXArray> vision =
        affine
            ? affineVisionWeights(weights, d)
            : Qwen3VlVisionTower.visionWeights(weights, d.depth(), d.deepstackIndexes());
    validateVisionShapes(d, vision);
    return new Qwen3VlModel(
        d,
        new TextDecoder(scope, d.text(), textWeights),
        new Qwen3VlVisionTower(scope, d.numHeads(), vision, d.deepstackIndexes()),
        new Qwen3VlImageProcessor(processorConfig),
        resolved);
  }

  private static Qwen3VlPromptTokens resolveTokens(
      HfTokenizer tokenizer, Qwen3VlPromptTokens declared) {
    try {
      return Qwen3VlPromptTokens.resolve(
          tokenizer,
          declared.imageTokenId(),
          declared.videoTokenId(),
          declared.visionStartTokenId(),
          declared.visionEndTokenId());
    } catch (TokenizerException | IllegalArgumentException failure) {
      throw new IllegalArgumentException(
          "checkpoint tokenizer does not match the config's prompt sentinels", failure);
    }
  }

  /**
   * Every stored weight must be floating, and every non-FLOAT32 storage dtype is up-cast to FLOAT32
   * in place (evaluated, the original closed), so the modules the loader builds operate on one
   * dtype. The up-cast doubles the byte cost of a BF16/FP16 stored weight.
   */
  private static Map<String, MLXArray> upcastAll(Map<String, MLXArray> loaded) {
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    for (var entry : loaded.entrySet()) {
      MLXArray original = entry.getValue();
      if (!original.dtype().isInexact()) {
        throw new IllegalArgumentException("Qwen3-VL weights must be floating: " + entry.getKey());
      }
      MLXArray value =
          original.dtype() == DType.FLOAT32 ? original : MLX.astype(original, DType.FLOAT32);
      if (value != original) {
        MLX.eval(value);
        original.close();
      }
      weights.put(entry.getKey(), value);
    }
    return weights;
  }

  /**
   * The affine-quantized layout's dtype pass: the packed U32 weight tensors are kept as-is (they
   * are not floating storage and cannot be up-cast), every other non-FLOAT32 float tensor is
   * up-cast to FLOAT32 in place (evaluated, the original closed), and a non-inexact, non-U32 dtype
   * is a storage the layout does not recognize and is rejected by name.
   */
  private static Map<String, MLXArray> upcastFloatsKeepPacked(Map<String, MLXArray> loaded) {
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    for (var entry : loaded.entrySet()) {
      MLXArray original = entry.getValue();
      DType dtype = original.dtype();
      if (!dtype.isInexact()) {
        if (dtype != DType.UINT32) {
          throw new IllegalArgumentException(
              "Qwen3-VL weights must be floating or packed U32: " + entry.getKey());
        }
        weights.put(entry.getKey(), original);
        continue;
      }
      MLXArray value = dtype == DType.FLOAT32 ? original : MLX.astype(original, DType.FLOAT32);
      if (value != original) {
        MLX.eval(value);
        original.close();
      }
      weights.put(entry.getKey(), value);
    }
    return weights;
  }

  private static void validateTextShapes(Qwen3VlDescriptor d, Map<String, MLXArray> weights) {
    DecoderConfig c = d.text().dimensions();
    int hidden = c.hiddenSize();
    int qkv = Math.multiplyExact(c.numAttentionHeads(), d.text().headDim());
    int kv = Math.multiplyExact(c.numKeyValueHeads(), d.text().headDim());
    for (var entry : weights.entrySet()) {
      String key = entry.getKey();
      MLXArray value = entry.getValue();
      if (key.equals("lm_head.weight") || key.equals("model.embed_tokens.weight")) {
        requireShape(value, c.vocabSize(), hidden);
      } else if (key.endsWith("norm.weight")) {
        // q_norm/k_norm are per-head-dim, every other norm is per-hidden.
        requireShape(
            value,
            key.endsWith("q_norm.weight") || key.endsWith("k_norm.weight")
                ? d.text().headDim()
                : hidden);
      } else if (key.endsWith("k_proj.weight") || key.endsWith("v_proj.weight")) {
        requireShape(value, kv, hidden);
      } else if (key.endsWith("q_proj.weight")) {
        requireShape(value, qkv, hidden);
      } else if (key.endsWith("o_proj.weight")) {
        requireShape(value, hidden, qkv);
      } else if (key.endsWith("gate_proj.weight") || key.endsWith("up_proj.weight")) {
        requireShape(value, c.intermediateSize(), hidden);
      } else if (key.endsWith("down_proj.weight")) {
        requireShape(value, hidden, c.intermediateSize());
      } else {
        throw new IllegalArgumentException("unexpected text checkpoint key '" + key + "'");
      }
    }
  }

  /**
   * The affine-quantized layout's text shape validation: every key must be a {@code .weight} of a
   * known projection, the embedding or the head (plus, for packed weights, its {@code
   * .scales}/{@code .biases} sidecars), and each shape must match the config's dimensions for the
   * tensor's packing — a packed U32 weight is {@code [out, in/8]} (4-bit groups of 8) with its
   * sidecars {@code [out, in/group_size]}, a floating weight {@code [out, in]} and a norm weight
   * 1-D. A packed tensor whose input width does not divide the config's group size is rejected by
   * name, since its groups cannot be reconstructed, as is a packed weight missing either sidecar or
   * a floating weight that carries one. The names the tensor plan already validated, so this pass
   * adds the shapes and the group division, not a second name check.
   */
  private static void validateQuantizedTextShapes(
      Qwen3VlDescriptor d, Map<String, MLXArray> weights, int group, int bits) {
    for (var entry : weights.entrySet()) {
      String key = entry.getKey();
      MLXArray value = entry.getValue();
      int dot = key.lastIndexOf('.');
      String stem = dot < 0 ? key : key.substring(0, dot);
      String suffix = dot < 0 ? "" : key.substring(dot + 1);
      if (suffix.equals("scales") || suffix.equals("biases")) {
        MLXArray weight = weights.get(stem + ".weight");
        if (weight == null || weight.dtype() != DType.UINT32) {
          throw new IllegalArgumentException(
              "checkpoint tensor '"
                  + key
                  + "' is a quantization sidecar without its packed weight");
        }
        requireShape(value, textOutputWidth(d, stem), textInputWidth(d, stem) / group);
        continue;
      }
      if (!suffix.equals("weight")) {
        throw new IllegalArgumentException("unexpected text checkpoint key '" + key + "'");
      }
      if (stem.endsWith("norm")) {
        // q_norm/k_norm and the layer norms are float-only; the plan already bars their sidecars.
        requireShape(value, textOutputWidth(d, stem));
        if (weights.containsKey(stem + ".scales")) {
          throw new IllegalArgumentException(
              "checkpoint tensor '" + key + "' is floating but carries quantization sidecars");
        }
        continue;
      }
      int out = textOutputWidth(d, stem);
      int input = textInputWidth(d, stem);
      if (value.dtype() == DType.UINT32) {
        if (input % group != 0) {
          throw new IllegalArgumentException(
              "checkpoint tensor '"
                  + key
                  + "' is incompatible with group_size "
                  + group
                  + " and input size "
                  + input);
        }
        requireShape(value, out, Math.multiplyExact(input, bits) / 32);
        MLXArray scales = weights.get(stem + ".scales");
        if (scales == null) {
          throw new IllegalArgumentException("checkpoint missing tensor '" + stem + ".scales'");
        }
        requireShape(scales, out, input / group);
        MLXArray biases = weights.get(stem + ".biases");
        if (biases == null) {
          throw new IllegalArgumentException("checkpoint missing tensor '" + stem + ".biases'");
        }
        requireShape(biases, out, input / group);
      } else {
        if (weights.containsKey(stem + ".scales")) {
          throw new IllegalArgumentException(
              "checkpoint tensor '" + key + "' is floating but carries quantization sidecars");
        }
        requireShape(value, out, input);
      }
    }
  }

  /**
   * The output width of one affine-layout text stem (the 1-D width for a norm). An unrecognized
   * stem is a programming error, since the tensor plan validated the name set first.
   */
  private static int textOutputWidth(Qwen3VlDescriptor d, String stem) {
    DecoderConfig c = d.text().dimensions();
    if (stem.equals("model.embed_tokens") || stem.equals("lm_head")) {
      return c.vocabSize();
    }
    if (stem.endsWith(".q_proj")) {
      return Math.multiplyExact(c.numAttentionHeads(), d.text().headDim());
    }
    if (stem.endsWith(".k_proj") || stem.endsWith(".v_proj")) {
      return Math.multiplyExact(c.numKeyValueHeads(), d.text().headDim());
    }
    int hidden = c.hiddenSize();
    if (stem.endsWith(".o_proj") || stem.endsWith(".down_proj")) {
      return hidden;
    }
    if (stem.endsWith(".gate_proj") || stem.endsWith(".up_proj")) {
      return c.intermediateSize();
    }
    if (stem.endsWith(".q_norm") || stem.endsWith(".k_norm")) {
      return d.text().headDim();
    }
    if (stem.endsWith(".input_layernorm")
        || stem.endsWith(".post_attention_layernorm")
        || stem.equals("model.norm")) {
      return hidden;
    }
    throw new IllegalArgumentException("unexpected text checkpoint key '" + stem + ".weight'");
  }

  /** The input width of one affine-layout text stem (the rotation width for a norm). */
  private static int textInputWidth(Qwen3VlDescriptor d, String stem) {
    DecoderConfig c = d.text().dimensions();
    int hidden = c.hiddenSize();
    if (stem.equals("model.embed_tokens") || stem.equals("lm_head")) {
      return hidden;
    }
    if (stem.endsWith(".o_proj")) {
      return Math.multiplyExact(c.numAttentionHeads(), d.text().headDim());
    }
    if (stem.endsWith(".down_proj")) {
      return c.intermediateSize();
    }
    if (stem.endsWith(".q_norm") || stem.endsWith(".k_norm")) {
      return d.text().headDim();
    }
    if (stem.endsWith(".input_layernorm")
        || stem.endsWith(".post_attention_layernorm")
        || stem.equals("model.norm")) {
      return hidden;
    }
    if (stem.endsWith(".q_proj")
        || stem.endsWith(".k_proj")
        || stem.endsWith(".v_proj")
        || stem.endsWith(".gate_proj")
        || stem.endsWith(".up_proj")) {
      return hidden;
    }
    throw new IllegalArgumentException("unexpected text checkpoint key '" + stem + ".weight'");
  }

  /**
   * The tied-head alias rule, both layouts: when the config declares tied embeddings and the
   * checkpoint nevertheless stores an {@code lm_head} block, it must be bit-for-bit the token
   * embeddings themselves. Under the affine layout every stored sidecar ({@code .scales}, {@code
   * .biases}) must agree with the embedding's too; a sidecar present on only one side is an alias
   * mismatch, as is one that differs.
   */
  private static void checkTiedLmHeadAlias(
      Qwen3VlDescriptor d, Map<String, MLXArray> textWeights, boolean affine) {
    if (!d.text().head().tied() || !textWeights.containsKey("lm_head.weight")) {
      return;
    }
    for (String suffix :
        affine ? new String[] {"weight", "scales", "biases"} : new String[] {"weight"}) {
      MLXArray head = textWeights.get("lm_head." + suffix);
      MLXArray embedding = textWeights.get("model.embed_tokens." + suffix);
      if (head == null && embedding == null) {
        continue;
      }
      if (head == null || embedding == null || !SmolVlmModel.sameWeights(head, embedding)) {
        throw new IllegalArgumentException("tied lm_head alias differs from token embeddings");
      }
    }
  }

  private static void validateVisionShapes(Qwen3VlDescriptor d, Map<String, MLXArray> vision) {
    int hidden = d.visionHidden();
    requireShape(
        vision.get("patchEmbedding.weight"),
        hidden,
        d.temporalPatchSize(),
        d.patchSize(),
        d.patchSize(),
        d.inChannels());
    requireShape(vision.get("patchEmbedding.bias"), hidden);
    requireShape(vision.get("positionEmbedding.weight"), d.numPositionEmbeddings(), hidden);
    int intermediate = d.visionIntermediate();
    for (int i = 0; i < d.depth(); i++) {
      String layer = "layer" + i + ".";
      requireShape(
          vision.get(layer + "attention.qkv.weight"), Math.multiplyExact(3, hidden), hidden);
      requireShape(vision.get(layer + "attention.qkv.bias"), Math.multiplyExact(3, hidden));
      requireShape(vision.get(layer + "attention.proj.weight"), hidden, hidden);
      requireShape(vision.get(layer + "attention.proj.bias"), hidden);
      for (String norm : new String[] {"norm1", "norm2"}) {
        requireShape(vision.get(layer + norm + ".weight"), hidden);
        requireShape(vision.get(layer + norm + ".bias"), hidden);
      }
      requireShape(vision.get(layer + "mlp.0.weight"), intermediate, hidden);
      requireShape(vision.get(layer + "mlp.0.bias"), intermediate);
      requireShape(vision.get(layer + "mlp.2.weight"), hidden, intermediate);
      requireShape(vision.get(layer + "mlp.2.bias"), hidden);
    }
    requireShape(vision.get("merger.norm.weight"), hidden);
    requireShape(vision.get("merger.norm.bias"), hidden);
    int m = Math.multiplyExact(hidden, Math.multiplyExact(d.mergeSize(), d.mergeSize()));
    requireShape(vision.get("merger.linearFc1.weight"), m, m);
    requireShape(vision.get("merger.linearFc1.bias"), m);
    int out = d.outHiddenSize();
    requireShape(vision.get("merger.linearFc2.weight"), out, m);
    requireShape(vision.get("merger.linearFc2.bias"), out);
    for (int j = 0; j < d.deepstackIndexes().length; j++) {
      String prefix = "deepstackMerger" + j + ".";
      requireShape(vision.get(prefix + "norm.weight"), m);
      requireShape(vision.get(prefix + "norm.bias"), m);
      requireShape(vision.get(prefix + "linearFc1.weight"), m, m);
      requireShape(vision.get(prefix + "linearFc1.bias"), m);
      requireShape(vision.get(prefix + "linearFc2.weight"), out, m);
      requireShape(vision.get(prefix + "linearFc2.bias"), out);
    }
  }

  /**
   * The affine-layout vision split: every {@code vision_tower.*} key is mapped through the same
   * tower-key scheme as the HF layout (the two subtrees are identical, so the mapped names and the
   * shape validation are shared) but without the PyTorch OIHW-to-MLX conv transpose — the converted
   * artifact's conv weight already arrives channels-last, and transposing it would corrupt it.
   */
  private static Map<String, MLXArray> affineVisionWeights(
      Map<String, MLXArray> weights, Qwen3VlDescriptor d) {
    Map<String, MLXArray> vision = new LinkedHashMap<>();
    for (var entry : weights.entrySet()) {
      String key = entry.getKey();
      if (!key.startsWith(Qwen3VlDescriptor.VISION_TOWER_PREFIX)) {
        continue;
      }
      String towerKey =
          Qwen3VlVisionTower.visionKeyToTowerKey(
              Qwen3VlVisionTower.VISUAL_PREFIX
                  + key.substring(Qwen3VlDescriptor.VISION_TOWER_PREFIX.length()),
              d.depth(),
              d.deepstackIndexes());
      if (towerKey == null) {
        throw new IllegalArgumentException("unrecognized vision checkpoint key '" + key + "'");
      }
      if (vision.put(towerKey, entry.getValue()) != null) {
        throw new IllegalArgumentException("duplicate vision checkpoint key '" + key + "'");
      }
    }
    return vision;
  }

  @Override
  public ModelMetadata metadata() {
    return metadata;
  }

  @Override
  public GenerationResult generate(GenerationRequest request, Consumer<GenerationEvent> listener) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(listener, "listener");
    int[] original = request.promptTokenIds();
    DecoderModel.validateTokenIds(original, request.config(), metadata.vocabSize());
    if (request.tokenizer() != null) {
      if (request.tokenizer().vocabSize() > metadata.vocabSize()) {
        throw new IllegalArgumentException(
            "tokenizer vocabulary size "
                + request.tokenizer().vocabSize()
                + " exceeds checkpoint vocabulary size "
                + metadata.vocabSize());
      }
      Qwen3VlPromptTokens requestTokens;
      try {
        requestTokens = resolveTokens(request.tokenizer(), tokens);
      } catch (IllegalArgumentException failure) {
        throw new IllegalArgumentException(
            "request tokenizer does not match model prompt sentinels", failure);
      }
      if (!requestTokens.equals(tokens)) {
        throw new IllegalArgumentException(
            "request tokenizer does not match model prompt sentinels");
      }
    }
    // The declared video token is validated at load but carried as -1 (video generation is out of
    // scope), so the prompt scan that the expander would run from a declared carrier never fires;
    // reject it here, before the wrapper grammar, with the expander's own message.
    int video = descriptor.videoTokenId();
    if (video >= 0) {
      for (int id : original) {
        if (id == video) {
          throw new IllegalArgumentException(
              "prompt contains the video token (ID " + video + "); video is not supported");
        }
      }
    }
    Qwen3VlPromptPlan plan = plan(original, request.images());
    // The text decoder's family descriptor never declares a checkpoint sliding window, so the
    // generic resolution would fire first; reject both sliding request modes here with the
    // composite's own message.
    if (request.cachePolicy().mode() == GenerationCachePolicy.Mode.SLIDING_WINDOW
        || request.cachePolicy().mode() == GenerationCachePolicy.Mode.SLIDING_WINDOW_FROM_MODEL) {
      throw new IllegalArgumentException("sliding image-prefill retention is not yet supported");
    }
    KVCachePolicy cachePolicy = decoder.resolveCachePolicy(request.cachePolicy());
    plan.requireCacheCapacity(cachePolicy, request.config().maxNewTokens());
    if (plan.requiredCacheCapacity(request.config().maxNewTokens()) > descriptor.contextLength()) {
      throw new IllegalArgumentException(
          "generation exceeds context length: expanded prompt "
              + plan.expandedLength()
              + ", unexpanded "
              + original.length);
    }
    try {
      return decoder.runStepLoop(
          request,
          request.config(),
          original,
          plan.expandedIds(),
          plan.expandedLength(),
          cachePolicy,
          listener,
          (activation, caches) ->
              prefill(request.cancellationToken(), request.images(), plan, activation, caches),
          (ids, caches) -> {
            observer.accept("decode");
            int position = plan.ropePositionAt(caches.get(0).nextPosition());
            return decoder.decode(
                ids,
                caches,
                mrope.prepare(ids.scope(), 1, 1, new int[][] {{position}, {position}, {position}}),
                null);
          });
    } catch (PreparationCancelled cancelled) {
      GenerationResult result =
          new GenerationResult(
              Arrays.stream(original).boxed().toList(),
              List.of(),
              FinishReason.CANCELLED,
              List.of(),
              request.tokenizer() == null ? null : "",
              plan.expandedLength());
      try {
        listener.accept(
            request.tokenizer() == null
                ? GenerationEvent.finished(FinishReason.CANCELLED)
                : GenerationEvent.finished(FinishReason.CANCELLED, ""));
      } catch (RuntimeException failure) {
        System.getLogger(Qwen3VlModel.class.getName())
            .log(System.Logger.Level.WARNING, "generation terminal listener failed", failure);
      }
      return result;
    }
  }

  /**
   * The pure expansion of one unexpanded prompt against the request's images: per-image geometry
   * from the configured processor (no pixels), the wrapper-grammar expansion, and the expanded
   * length bound. Zero-token requests stop at this validation without pixel or native work.
   */
  Qwen3VlPromptPlan plan(int[] ids, List<RgbImage> images) {
    List<Qwen3VlGeometryPlan> geometries = new ArrayList<>(images.size());
    for (RgbImage image : images) {
      geometries.add(processor.geometry(image.width(), image.height()));
    }
    Qwen3VlPromptPlan plan = Qwen3VlPromptExpander.expand(ids, tokens, geometries);
    if (plan.expandedLength() > descriptor.contextLength()) {
      throw new IllegalArgumentException(
          "expanded prompt exceeds context length: expanded "
              + plan.expandedLength()
              + ", unexpanded "
              + ids.length);
    }
    return plan;
  }

  /**
   * Runs and evaluates the request's first step, releasing all visual work before returning.
   * Package access lets lifecycle tests exercise this production boundary with caller-owned KV.
   */
  DecoderModel.FirstStepResult prefill(
      CancellationToken cancellation,
      List<RgbImage> images,
      Qwen3VlPromptPlan plan,
      MLXScope activation,
      List<KVCache> caches) {
    checkCancellation(cancellation);
    int length = plan.expandedLength();
    int[][] before = decoder.preflightBatch(1, length, caches, new int[] {length});
    DecoderModel.FirstStepResult result;
    try (MLXScope prefill = activation.newChild()) {
      Qwen3VlVisionFeatures features = features(cancellation, images, prefill);
      observer.accept("before-prefill");
      checkCancellation(cancellation);
      MLXArray ids = MLX.array(prefill, plan.expandedIds(), new int[] {1, length});
      MultiAxisRotation rotation =
          mrope.prepare(
              prefill,
              1,
              length,
              new int[][] {
                plan.temporalPositions(), plan.heightPositions(), plan.widthPositions()
              });
      int[] destinations = plan.featureDestinations();
      int rows = destinations.length;
      int hidden = decoder.config().hiddenSize();
      DecoderModel.EmbeddingHook embeddingHook =
          features == null
              ? null
              : embedded ->
                  VisualFeatureInjection.replaceImageRows(
                      embedded,
                      MLXShape.reshape(features.merged(), new int[] {1, rows, hidden}),
                      destinations);
      DecoderModel.PostBlockHook postBlockHook =
          features == null
              ? null
              : (layerIndex, hiddenStates) -> {
                if (layerIndex >= features.deepstackLevels().size()) {
                  return hiddenStates;
                }
                return VisualFeatureInjection.addMaskedFeatures(
                    hiddenStates.scope(),
                    hiddenStates,
                    MLXShape.reshape(
                        features.deepstackLevels().get(layerIndex), new int[] {1, rows, hidden}),
                    destinations);
              };
      MLXArray logits =
          decoder.stepLogits(
              ids, caches, new int[] {length}, embeddingHook, rotation, postBlockHook);
      List<MLXArray> evaluated = new ArrayList<>();
      evaluated.add(logits);
      evaluated.addAll(Arrays.asList(DecoderModel.stepCacheArrays(caches)));
      MLX.eval(evaluated.toArray(MLXArray[]::new));
      observer.accept("prefill-evaluated");
      result =
          new DecoderModel.FirstStepResult(
              MLX.hoist(logits, activation),
              cs -> DecoderModel.poisonBatchIfMutated(cs, before, 1));
    } catch (RuntimeException | Error failure) {
      DecoderModel.poisonBatchIfMutated(caches, before, 1);
      throw failure;
    }
    try {
      observer.accept("prefill-closed");
    } catch (RuntimeException | Error failure) {
      DecoderModel.poisonBatchIfMutated(caches, before, 1);
      throw failure;
    }
    return result;
  }

  /**
   * Processes and evaluates one image at a time. Only the final main/DeepStack rows survive each
   * image child; processed Java patches and native activations are released before the next image.
   * All returned rows belong to the caller's prefill scope, never to the model or decode scope.
   */
  Qwen3VlVisionFeatures features(
      CancellationToken cancellation, List<RgbImage> images, MLXScope prefill) {
    if (images.isEmpty()) {
      return null;
    }
    List<MLXArray> merged = new ArrayList<>();
    List<List<MLXArray>> levels = new ArrayList<>();
    for (int i = 0; i < tower.deepstackIndexes().length; i++) {
      levels.add(new ArrayList<>());
    }
    for (RgbImage image : images) {
      checkCancellation(cancellation);
      observer.accept("preprocess");
      checkCancellation(cancellation);
      try (MLXScope vision = prefill.newChild()) {
        final var processed = processor.process(image);
        checkCancellation(cancellation);
        observer.accept("vision");
        checkCancellation(cancellation);
        var output = tower.forward(vision, List.of(processed));
        merged.add(MLX.hoist(output.merged(), prefill));
        for (int level = 0; level < levels.size(); level++) {
          levels.get(level).add(MLX.hoist(output.deepstackLevels().get(level), prefill));
        }
      }
      observer.accept("image-closed");
      checkCancellation(cancellation);
    }
    List<MLXArray> deepstack = new ArrayList<>();
    for (List<MLXArray> parts : levels) {
      deepstack.add(joinFeatures(parts));
    }
    return new Qwen3VlVisionFeatures(joinFeatures(merged), deepstack);
  }

  private static MLXArray joinFeatures(List<MLXArray> parts) {
    if (parts.size() == 1) {
      return parts.getFirst();
    }
    MLXArray joined = MLXShape.concatenate(parts.toArray(MLXArray[]::new), 0);
    MLX.eval(joined);
    parts.forEach(MLXArray::close);
    return joined;
  }

  /**
   * The package-test probe of the text-only prefill: the same first-step computation {@code
   * generate} runs (expansion, multi-axis rotation, cached prefill) with no images, on a fresh
   * generation scope the caller passes, returning the evaluated prefill logits. The caller must
   * release the scope.
   */
  float[] textOnlyPrefillLogitsForTest(MLXScope generation, int[] ids) {
    Objects.requireNonNull(generation, "generation");
    Objects.requireNonNull(ids, "ids");
    Qwen3VlPromptPlan plan = plan(ids, List.of());
    int layers = decoder.config().numHiddenLayers();
    List<KVCache> caches = new ArrayList<>(layers);
    for (int i = 0; i < layers; i++) {
      caches.add(new KVCache(generation));
    }
    DecoderModel.FirstStepResult result =
        prefill(CancellationToken.NONE, List.of(), plan, generation, caches);
    List<MLXArray> evaluated = new ArrayList<>();
    evaluated.add(result.logits());
    evaluated.addAll(Arrays.asList(DecoderModel.stepCacheArrays(caches)));
    MLX.eval(evaluated.toArray(MLXArray[]::new));
    return result.logits().toFloatArray();
  }

  /**
   * The package-test probe of one prefill with images, on a fresh generation scope the caller
   * passes and must release: the same plan, expansion, cached first step and evaluation {@code
   * generate} runs. With no images it is the plain text forward over the whole prompt, returning
   * every position's logits flat ({@code [T*V]}); with images it runs the vision tower and
   * DeepStack injection like a request and returns the last position's logits ({@code [V]}).
   *
   * <p>The text-only branch calls the plain decoder's {@code forward}, bypassing the composite's
   * multi-axis prefill. The text-only equivalence suite proves the two paths bit-identical. The
   * wide-attention golden's text decode case exercises the composite prefill through {@link
   * #decodeStepLogitsForTest} before checking the cached decode logits.
   */
  float[] prefillLogitsForTest(MLXScope generation, int[] ids, List<RgbImage> images) {
    Objects.requireNonNull(generation, "generation");
    Objects.requireNonNull(ids, "ids");
    Objects.requireNonNull(images, "images");
    Qwen3VlPromptPlan plan = plan(ids, images);
    List<KVCache> caches = freshCaches(generation);
    if (images.isEmpty()) {
      // The text-only multi-axis rotation is the family's base RoPE at the same sequential
      // positions, so the plain text forward is the prefill's own computation (proven
      // bit-for-bit by the text-only equivalence suite).
      return decoder
          .forward(
              MLX.array(generation, ids, new int[] {1, ids.length}), caches, new int[] {ids.length})
          .toFloatArray();
    }
    DecoderModel.FirstStepResult result =
        prefill(CancellationToken.NONE, images, plan, generation, caches);
    List<MLXArray> evaluated = new ArrayList<>();
    evaluated.add(result.logits());
    evaluated.addAll(Arrays.asList(DecoderModel.stepCacheArrays(caches)));
    MLX.eval(evaluated.toArray(MLXArray[]::new));
    return result.logits().toFloatArray();
  }

  /**
   * The package-test probe of one KV-cached decode step after one prefill, on a fresh generation
   * scope the caller passes and must release: the same prefill {@link
   * #prefillLogitsForTest(MLXScope, int[], List)} runs first (its prefill logits are evaluated),
   * then one decode step on {@code decodeTokenId} with the request's own multi-axis rotation at the
   * next expansion position, returning the step's last-position logits ({@code [V]}).
   */
  float[] decodeStepLogitsForTest(
      MLXScope generation, int[] ids, List<RgbImage> images, int decodeTokenId) {
    Objects.requireNonNull(generation, "generation");
    Objects.requireNonNull(ids, "ids");
    Objects.requireNonNull(images, "images");
    Qwen3VlPromptPlan plan = plan(ids, images);
    List<KVCache> caches = freshCaches(generation);
    DecoderModel.FirstStepResult result =
        prefill(CancellationToken.NONE, images, plan, generation, caches);
    List<MLXArray> evaluated = new ArrayList<>();
    evaluated.add(result.logits());
    evaluated.addAll(Arrays.asList(DecoderModel.stepCacheArrays(caches)));
    MLX.eval(evaluated.toArray(MLXArray[]::new));
    int position = plan.ropePositionAt(caches.get(0).nextPosition());
    return decoder
        .decode(
            MLX.array(generation, new int[] {decodeTokenId}, new int[] {1, 1}),
            caches,
            mrope.prepare(generation, 1, 1, new int[][] {{position}, {position}, {position}}),
            null)
        .toFloatArray();
  }

  private List<KVCache> freshCaches(MLXScope generation) {
    int layers = decoder.config().numHiddenLayers();
    List<KVCache> caches = new ArrayList<>(layers);
    for (int i = 0; i < layers; i++) {
      caches.add(new KVCache(generation));
    }
    return caches;
  }

  void setObserverForTest(Consumer<String> observer) {
    this.observer = Objects.requireNonNull(observer);
  }

  DecoderModel decoderForTest() {
    return decoder;
  }

  Qwen3VlVisionTower towerForTest() {
    return tower;
  }

  Qwen3VlDescriptor descriptorForTest() {
    return descriptor;
  }

  private static final class PreparationCancelled extends RuntimeException {}

  private static void checkCancellation(CancellationToken token) {
    if (token.isCancelled()) {
      throw new PreparationCancelled();
    }
  }

  private static void requireShape(MLXArray array, int... shape) {
    if (array == null) {
      throw new IllegalArgumentException("missing checkpoint weight");
    }
    if (!Arrays.equals(array.shape(), shape)) {
      throw new IllegalArgumentException(
          "checkpoint shape "
              + Arrays.toString(array.shape())
              + " differs from config "
              + Arrays.toString(shape));
    }
  }
}
