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
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.TokenizerException;
import se.alipsa.jmlx.vision.ImageTensor;
import se.alipsa.jmlx.vision.RgbImage;
import se.alipsa.jmlx.vision.SmolVlmImageProcessor;
import se.alipsa.jmlx.vision.SmolVlmImageProcessorResult;
import se.alipsa.jmlx.vision.SmolVlmProcessorConfig;
import tools.jackson.databind.JsonNode;

/**
 * Local float-checkpoint Idefics3/SmolVLM image-and-text generation. Weights and computation use
 * FLOAT32, including up-casts from FP16/BF16 storage (roughly twice the source weight memory).
 * Images are processed once during prefill; cached decode retains only self-attention KV.
 *
 * <p>The caller owns the loading scope. Requests are batch one and follow core's single platform
 * thread rule. Full and capacity-bounded full caches are supported; sliding retention is deferred
 * pending image-prefill equivalence evidence. Synthetic correctness does not imply Tier-B support.
 */
public final class SmolVlmModel implements TextGenerationModel {
  private static final int MAX_IMAGES = 16;
  private static final long MAX_PIXELS = 67_108_864L;
  private static final int MAX_TILES = 256;
  private final SmolVlmDescriptor descriptor;
  private final DecoderModel decoder;
  private final Idefics3VisionTower tower;
  private final SmolVlmImageProcessor processor;
  private final SmolVlmPromptTokens tokens;
  private final ModelMetadata metadata;
  private Consumer<String> observer = ignored -> {};

  private static final class TextDecoder extends DecoderModel {
    TextDecoder(MLXScope scope, ArchitectureDescriptor descriptor, Map<String, MLXArray> weights) {
      super(scope, descriptor, weights);
    }
  }

  private SmolVlmModel(
      SmolVlmDescriptor descriptor,
      DecoderModel decoder,
      Idefics3VisionTower tower,
      SmolVlmImageProcessor processor,
      SmolVlmPromptTokens tokens) {
    this.descriptor = descriptor;
    this.decoder = decoder;
    this.tower = tower;
    this.processor = processor;
    this.tokens = tokens;
    metadata =
        new VisionLanguageMetadata(
            "idefics3",
            decoder.config().vocabSize(),
            decoder.config().numHiddenLayers(),
            descriptor.layers());
  }

  /** Loads the checkpoint, tokenizer marker IDs and slow image processor from one directory. */
  public static SmolVlmModel load(MLXScope scope, Path directory) throws IOException {
    return load(scope, directory, JsonFiles.read(directory.resolve("config.json")));
  }

  static SmolVlmModel load(MLXScope scope, Path directory, JsonNode root) throws IOException {
    Objects.requireNonNull(scope, "scope");
    SmolVlmDescriptor d = SmolVlmDescriptor.parse(root);
    SmolVlmProcessorConfig config =
        SmolVlmProcessorConfig.fromJson(
            Files.readAllBytes(directory.resolve("preprocessor_config.json")));
    if (!config.doResize() || config.maxImageSizeLongestEdge() != d.imageSize()) {
      throw new IllegalArgumentException(
          "processor tile size must equal vision image_size with resize enabled");
    }
    JsonNode processorConfig = JsonFiles.read(directory.resolve("processor_config.json"));
    if (processorConfig.path("image_seq_len").asInt(-1) != d.tokensPerTile()) {
      throw new IllegalArgumentException(
          "processor image_seq_len disagrees with connector geometry");
    }
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    if (tokenizer.vocabSize() > d.text().dimensions().vocabSize()) {
      throw new IllegalArgumentException("tokenizer vocabulary exceeds checkpoint vocabulary");
    }
    SmolVlmPromptTokens tokens = resolveTokens(tokenizer);
    if (tokens.imageTokenId() != d.imageTokenId()) {
      throw new IllegalArgumentException("image_token_id disagrees with tokenizer <image>");
    }
    Map<String, MLXArray> loaded = CheckpointLoader.load(scope, directory, d.tensorPlan());
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    for (var entry : loaded.entrySet()) {
      MLXArray original = entry.getValue();
      if (!original.dtype().isInexact()) {
        throw new IllegalArgumentException(
            "vision-language weights must be floating: " + entry.getKey());
      }
      MLXArray value =
          original.dtype() == DType.FLOAT32 ? original : MLX.astype(original, DType.FLOAT32);
      if (value != original) {
        MLX.eval(value);
        original.close();
      }
      weights.put(entry.getKey(), value);
    }
    Map<String, MLXArray> textWeights = new LinkedHashMap<>();
    TensorPlan textPlan = ArchitectureMappings.tensorPlan(d.text());
    for (String key : weights.keySet()) {
      if (key.startsWith("model.text_model.") || key.equals("lm_head.weight")) {
        String local = key.startsWith("model.text_model.") ? "model." + key.substring(17) : key;
        textWeights.put(local, weights.get(key));
      }
    }
    textPlan.validate(textWeights.keySet());
    validateTextShapes(d, textWeights);
    if (d.text().head().tied()
        && textWeights.containsKey("lm_head.weight")
        && !sameWeights(
            textWeights.get("lm_head.weight"), textWeights.get("model.embed_tokens.weight"))) {
      throw new IllegalArgumentException("tied lm_head alias differs from token embeddings");
    }
    Map<String, MLXArray> vision = new LinkedHashMap<>();
    for (var entry : d.visionKeys().entrySet()) {
      MLXArray value = weights.get("model.vision_model." + entry.getKey());
      if (entry.getValue().equals("patchEmbedding.weight")) {
        requireShape(value, d.visionHidden(), 3, d.patchSize(), d.patchSize());
        // PyTorch OIHW -> MLX OHWI; provenance is the descriptor's explicit HF key mapping.
        MLXArray transposed = MLXShape.transpose(value, new int[] {0, 2, 3, 1});
        MLX.eval(transposed);
        value.close();
        value = transposed;
      }
      vision.put(entry.getValue(), value);
    }
    requireShape(
        vision.get("positionEmbedding.weight"),
        Math.multiplyExact(d.imageSize() / d.patchSize(), d.imageSize() / d.patchSize()),
        d.visionHidden());
    requireShape(vision.get("patchEmbedding.bias"), d.visionHidden());
    for (int i = 0; i < d.layers(); i++) {
      requireShape(
          vision.get("layer" + i + ".feedForward.0.weight"), d.intermediate(), d.visionHidden());
    }
    MLXArray projection = weights.get("model.connector.modality_projection.proj.weight");
    requireShape(
        projection,
        d.text().dimensions().hiddenSize(),
        Math.multiplyExact(d.visionHidden(), Math.multiplyExact(d.scale(), d.scale())));
    Idefics3Connector connector = new Idefics3Connector(scope, d.scale(), projection);
    Idefics3VisionTower tower =
        new Idefics3VisionTower(scope, d.layers(), d.heads(), d.normEps(), vision, connector);
    return new SmolVlmModel(
        d,
        new TextDecoder(scope, d.text(), textWeights),
        tower,
        new SmolVlmImageProcessor(config),
        tokens);
  }

  static SmolVlmPromptTokens resolveTokens(HfTokenizer tokenizer) {
    int base = special(tokenizer, "<row_1_col_1>") - 1;
    for (int r = 1; r <= 6; r++) {
      for (int c = 1; c <= 6; c++) {
        if (special(tokenizer, "<row_" + r + "_col_" + c + ">") != base + (r - 1) * 6 + c) {
          throw new IllegalArgumentException("row/column markers must follow row-major ID order");
        }
      }
    }
    int video = -1;
    // Encoding alone could map an absent marker to ordinary characters. Require a declared special.
    try {
      video = special(tokenizer, "<video>");
    } catch (se.alipsa.jmlx.tokenizer.TokenizerException absent) {
      // The pinned tokenizer has no video token; structured video content is rejected upstream.
    }
    return new SmolVlmPromptTokens(
        special(tokenizer, "<image>"),
        video,
        special(tokenizer, "<fake_token_around_image>"),
        special(tokenizer, "<global-img>"),
        base,
        ints(tokenizer.encode("\n", false)),
        ints(tokenizer.encode("\n\n", false)));
  }

  private static int special(HfTokenizer tokenizer, String name) {
    // The sentinels are ordinary vocabulary entries, not configured special tokens, so this is a
    // neutral name-to-ID lookup; the checks below independently require a declared special marker.
    int id = tokenizer.tokenId(name).orElseThrow();
    if (!tokenizer.decode(List.of(id), true).isEmpty()
        || !tokenizer.encode(name, false).equals(List.of(id))) {
      throw new IllegalArgumentException("image marker must be a declared special token: " + name);
    }
    return id;
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
      SmolVlmPromptTokens requestTokens;
      try {
        requestTokens = resolveTokens(request.tokenizer());
      } catch (TokenizerException | IllegalArgumentException failure) {
        throw new IllegalArgumentException(
            "request tokenizer does not match model image markers", failure);
      }
      if (!requestTokens.equals(tokens)) {
        throw new IllegalArgumentException("request tokenizer does not match model image markers");
      }
    }
    SmolVlmPromptPlan plan = plan(original, request.images());
    KVCachePolicy cachePolicy = decoder.resolveCachePolicy(request.cachePolicy());
    if (cachePolicy.mode() == KVCachePolicy.Mode.SLIDING_WINDOW) {
      throw new IllegalArgumentException("sliding image-prefill retention is not yet supported");
    }
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
          (activation, caches) -> prefill(request, plan, activation, caches));
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
        System.getLogger(SmolVlmModel.class.getName())
            .log(System.Logger.Level.WARNING, "generation terminal listener failed", failure);
      }
      return result;
    }
  }

  SmolVlmPromptPlan plan(int[] ids, List<RgbImage> images) {
    if (images.size() > MAX_IMAGES) {
      throw new IllegalArgumentException("at most " + MAX_IMAGES + " images per request");
    }
    long pixels = 0;
    int tiles = 0;
    List<SmolVlmImageGrid> grids = new ArrayList<>();
    for (RgbImage image : images) {
      pixels = Math.addExact(pixels, (long) image.width() * image.height());
      var geometry = processor.geometry(image.height(), image.width());
      tiles = Math.addExact(tiles, geometry.tileCount());
      if (pixels > MAX_PIXELS || tiles > MAX_TILES) {
        throw new IllegalArgumentException("request exceeds aggregate pixel/tile budget");
      }
      grids.add(new SmolVlmImageGrid(geometry.rows(), geometry.cols()));
    }
    SmolVlmPromptPlan plan =
        SmolVlmPromptExpander.expand(ids, tokens, grids, descriptor.tokensPerTile());
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
      GenerationRequest request,
      SmolVlmPromptPlan plan,
      MLXScope activation,
      List<KVCache> caches) {
    checkCancellation(request);
    int[][] before =
        decoder.preflightBatch(1, plan.expandedLength(), caches, new int[] {plan.expandedLength()});
    DecoderModel.FirstStepResult result;
    try (MLXScope prefill = activation.newChild()) {
      MLXArray features = features(request, prefill);
      checkCancellation(request);
      MLXArray ids = MLX.array(prefill, plan.expandedIds(), new int[] {1, plan.expandedLength()});
      MLXArray logits =
          decoder.stepLogits(
              ids,
              caches,
              new int[] {plan.expandedLength()},
              features == null ? null : embedded -> merge(embedded, features, plan));
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

  MLXArray features(GenerationRequest request, MLXScope prefill) {
    List<MLXArray> projected = new ArrayList<>();
    for (RgbImage image : request.images()) {
      checkCancellation(request);
      observer.accept("preprocess");
      checkCancellation(request);
      SmolVlmImageProcessorResult prepared = processor.process(image);
      checkCancellation(request);
      List<int[]> masks = prepared.pixelMasks();
      for (int i = 0; i < prepared.tiles().size(); i++) {
        checkCancellation(request);
        ImageTensor tile = prepared.tiles().get(i);
        try (MLXScope vision = prefill.newChild()) {
          MLXArray pixels = MLX.array(vision, tile.values(), tile.shape());
          int[] shape = tile.shape();
          int[] patchMask =
              Idefics3VisionTower.patchMaskFromPixelMask(
                  masks.get(i), shape[1], shape[2], descriptor.patchSize());
          observer.accept("vision");
          MLXArray hidden = tower.forward(pixels, patchMask);
          observer.accept("connector");
          MLXArray output = tower.connector().forward(hidden);
          MLX.eval(output);
          projected.add(MLX.hoist(output, prefill));
        }
      }
    }
    return projected.isEmpty() ? null : MLXShape.concatenate(projected.toArray(MLXArray[]::new), 1);
  }

  static MLXArray merge(MLXArray embedded, MLXArray features, SmolVlmPromptPlan plan) {
    if (features.ndim() != 3
        || features.shape()[0] != 1
        || features.shape()[1] != plan.totalFeatureRows()
        || features.shape()[2] != embedded.shape()[2]
        || features.dtype() != embedded.dtype()) {
      throw new IllegalArgumentException(
          "projected image features disagree with prompt destinations");
    }
    int[] gather =
        Idefics3VisionTower.mergeGatherIndices(
            plan.expandedLength(), Arrays.stream(plan.featureDestinations()).boxed().toList());
    MLXArray source = MLXShape.concatenate(new MLXArray[] {embedded, features}, 1);
    return MLXShape.takeAxis(
        source, MLX.array(embedded.scope(), gather, new int[] {gather.length}), 1);
  }

  void setObserverForTest(Consumer<String> observer) {
    this.observer = Objects.requireNonNull(observer);
  }

  DecoderModel decoderForTest() {
    return decoder;
  }

  Idefics3VisionTower towerForTest() {
    return tower;
  }

  private static final class PreparationCancelled extends RuntimeException {}

  private static void checkCancellation(GenerationRequest request) {
    if (request.cancellationToken().isCancelled()) {
      throw new PreparationCancelled();
    }
  }

  private static int[] ints(List<Integer> ids) {
    return ids.stream().mapToInt(Integer::intValue).toArray();
  }

  /** Numerical device equality: signed zeros agree; NaNs fail alias validation, even in pairs. */
  static boolean sameWeights(MLXArray head, MLXArray embedding) {
    try (MLXArray equal = MLXOps.equal(head, embedding);
        MLXArray all = MLXOps.all(equal);
        MLXArray scalar = MLX.astype(all, DType.INT32)) {
      return scalar.toIntArray()[0] != 0;
    }
  }

  private static void requireShape(MLXArray array, int... shape) {
    if (!Arrays.equals(array.shape(), shape)) {
      throw new IllegalArgumentException(
          "checkpoint shape "
              + Arrays.toString(array.shape())
              + " differs from config "
              + Arrays.toString(shape));
    }
  }

  private static void validateTextShapes(SmolVlmDescriptor d, Map<String, MLXArray> weights) {
    DecoderConfig c = d.text().dimensions();
    int hidden = c.hiddenSize();
    for (var entry : weights.entrySet()) {
      String key = entry.getKey();
      MLXArray value = entry.getValue();
      if (key.equals("lm_head.weight") || key.equals("model.embed_tokens.weight")) {
        requireShape(value, c.vocabSize(), hidden);
      } else if (key.endsWith("norm.weight")) {
        requireShape(value, hidden);
      } else {
        int rows =
            key.contains("k_proj") || key.contains("v_proj")
                ? c.numKeyValueHeads() * d.text().headDim()
                : hidden;
        int columns = hidden;
        if (key.contains("gate_proj") || key.contains("up_proj")) {
          rows = c.intermediateSize();
        } else if (key.contains("down_proj")) {
          columns = c.intermediateSize();
        }
        if (key.endsWith(".bias")) {
          requireShape(value, rows);
        } else {
          requireShape(value, rows, columns);
        }
      }
    }
  }
}
