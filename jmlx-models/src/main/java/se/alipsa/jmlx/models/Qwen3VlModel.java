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
import se.alipsa.jmlx.vision.Qwen3VlImageProcessorResult;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;

/**
 * Local float-checkpoint Qwen3-VL image-and-text generation. Weights and computation use FLOAT32,
 * including up-casts from the FP16/BF16 storage the pinned artifacts use (a BF16 checkpoint
 * therefore occupies roughly twice its source weight memory after loading). Images are processed
 * once during prefill and run through the vision tower there; each DeepStack level is injected
 * after its configured text block in the same prefill, and the cached decode retains only the
 * self-attention KV plus the per-step multi-axis rotation derived from the request's expansion
 * plan.
 *
 * <p>Checkpoint provenance is selected from the shard headers' {@code __metadata__} blocks and the
 * config before any tensor is loaded (see {@link Qwen3VlProvenance}): this work unit loads the HF
 * float layout only, and an MLX affine-quantized checkpoint is recognized and rejected by name. The
 * load then parses the config strictly ({@link Qwen3VlDescriptor}), cross-checks the image
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
   * <p>Load order: provenance selection (headers plus the quantization declarations; MLX affine is
   * recognized and rejected as unsupported in this work unit), strict config parse, processor
   * cross-check, tokenizer sentinel resolution, combined tensor preflight and load, the FLOAT32
   * up-cast, the text/vision split with shape and alias validation, and finally the module
   * construction on the caller's scope.
   *
   * @throws IOException if a file cannot be read
   * @throws IllegalArgumentException when the provenance, config, processor config, tokenizer,
   *     tensors or shapes disagree
   */
  static Qwen3VlModel load(MLXScope scope, Path directory, JsonNode root) throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(root, "root");
    Qwen3VlProvenance.resolve(directory, root);
    Qwen3VlDescriptor d = Qwen3VlDescriptor.parse(root);
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
    Map<String, MLXArray> loaded = CheckpointLoader.load(scope, directory, d.tensorPlan());
    Map<String, MLXArray> weights = upcastAll(loaded);
    Map<String, MLXArray> textWeights = new LinkedHashMap<>();
    for (String key : weights.keySet()) {
      if (key.startsWith(LANGUAGE_PREFIX)) {
        textWeights.put("model." + key.substring(LANGUAGE_PREFIX.length()), weights.get(key));
      } else if (key.equals("lm_head.weight")) {
        textWeights.put(key, weights.get(key));
      }
    }
    ArchitectureMappings.tensorPlan(d.text()).validate(textWeights.keySet());
    validateTextShapes(d, textWeights);
    if (d.text().head().tied()
        && textWeights.containsKey("lm_head.weight")
        && !SmolVlmModel.sameWeights(
            textWeights.get("lm_head.weight"), textWeights.get("model.embed_tokens.weight"))) {
      throw new IllegalArgumentException("tied lm_head alias differs from token embeddings");
    }
    Map<String, MLXArray> vision =
        Qwen3VlVisionTower.visionWeights(weights, d.depth(), d.deepstackIndexes());
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
      } else if (key.endsWith("q_proj.weight") || key.endsWith("o_proj.weight")) {
        requireShape(value, qkv, hidden);
      } else if (key.endsWith("gate_proj.weight") || key.endsWith("up_proj.weight")) {
        requireShape(value, c.intermediateSize(), hidden);
      } else if (key.endsWith("down_proj.weight")) {
        requireShape(value, hidden, c.intermediateSize());
      } else {
        throw new IllegalArgumentException("unexpected text checkpoint key '" + key + "'");
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

  private DecoderModel.FirstStepResult prefill(
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
      List<Qwen3VlImageProcessorResult> processed = new ArrayList<>(images.size());
      for (RgbImage image : images) {
        checkCancellation(cancellation);
        observer.accept("preprocess");
        checkCancellation(cancellation);
        processed.add(processor.process(image));
        checkCancellation(cancellation);
      }
      Qwen3VlVisionFeatures features;
      if (processed.isEmpty()) {
        features = null;
      } else {
        observer.accept("vision");
        checkCancellation(cancellation);
        features = tower.forward(prefill, processed);
      }
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
