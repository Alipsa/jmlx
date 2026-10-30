package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.nn.RopeSpec;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.Phi3VGeometryPlan;
import se.alipsa.jmlx.vision.Phi3VImageProcessor;
import se.alipsa.jmlx.vision.Phi3VImageProcessorResult;
import se.alipsa.jmlx.vision.Phi3VProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;

/**
 * Local float-checkpoint Phi-3.5-Vision LLaVA-style image-and-text generation. Weights and
 * computation use FLOAT32, including up-casts from FP16/BF16 storage (roughly twice the source
 * weight memory). The reference's lossy reference-tokenizer prompt expansion is reproduced from the
 * request's retained source text (findings §13): generation requires a tokenizer-backed request and
 * re-expands {@link GenerationRequest#promptText()} against the images, so the request's own token
 * IDs are used only for event accounting.
 *
 * <p>Images are processed once during prefill: per image, the CLIP tower runs over the real crop
 * slots in a private vision scope, the high-definition rows are hoisted to the prefill scope, and
 * the shared projector runs once over all images' concatenated rows. The projected rows replace the
 * sentinel rows' embeddings through the post-lookup hook; cached decode retains only self-attention
 * KV. With {@code su} rope scaling, the single decode step that first reaches {@code
 * original_max_position_embeddings + 1} total width resets the caches and re-prefills the whole
 * history under the long factor table (the reference's boundary recompute, pinned lines 1650-1655);
 * a prompt already at or beyond that width never triggers it.
 *
 * <p>The caller owns the loading scope. Requests are batch one and follow core's single platform
 * thread rule. Full and capacity-bounded full caches are supported; sliding retention is rejected
 * (the boundary re-prefill's retention equivalence is unproven). Synthetic correctness does not
 * imply Tier-B support.
 */
public final class Phi3VModel implements TextGenerationModel {
  private static final int MAX_IMAGES = 16;
  private static final int MAX_TOTAL_SLOTS = 80;
  private static final int CLIP_IMAGE = 336;
  private final Phi3VDescriptor descriptor;
  private final TextDecoder decoder;
  private final Phi3VVisionTower tower;
  private final Phi3VImageProcessor processor;
  private final HfTokenizer tokenizer;
  private final int clipLayers;
  private final int originalMaxPositions;
  private final int maxPositions;
  private final ModelMetadata metadata;
  private Consumer<String> observer = ignored -> {};

  private static final class TextDecoder extends DecoderModel {
    TextDecoder(MLXScope scope, ArchitectureDescriptor descriptor, Map<String, MLXArray> weights) {
      super(scope, descriptor, weights);
    }
  }

  private Phi3VModel(
      Phi3VDescriptor descriptor,
      TextDecoder decoder,
      Phi3VVisionTower tower,
      Phi3VImageProcessor processor,
      HfTokenizer tokenizer,
      int clipLayers) {
    this.descriptor = descriptor;
    this.decoder = decoder;
    this.tower = tower;
    this.processor = processor;
    this.tokenizer = tokenizer;
    this.clipLayers = clipLayers;
    RopeSpec.Su rope = (RopeSpec.Su) descriptor.text().rope();
    this.originalMaxPositions = rope.originalMaxPositions();
    this.maxPositions = rope.maxPositions();
    metadata =
        new VisionLanguageMetadata(
            "phi3_v", decoder.config().vocabSize(), decoder.config().numHiddenLayers(), clipLayers);
  }

  /** Loads the checkpoint, tokenizer and slow image processor from one directory. */
  public static Phi3VModel load(MLXScope scope, Path directory) throws IOException {
    return load(scope, directory, JsonFiles.read(directory.resolve("config.json")));
  }

  static Phi3VModel load(MLXScope scope, Path directory, JsonNode root) throws IOException {
    Objects.requireNonNull(scope, "scope");
    Phi3VDescriptor d = Phi3VDescriptor.parse(root);
    int clipLayers = discoverClipLayers(directory);
    Phi3VProcessorConfig preprocessor =
        Phi3VProcessorConfig.fromJson(
            Files.readAllBytes(directory.resolve("preprocessor_config.json")));
    if (preprocessor.numImgTokens() != null && preprocessor.numImgTokens() != d.numImgTokens()) {
      throw new IllegalArgumentException(
          "preprocessor num_img_tokens "
              + preprocessor.numImgTokens()
              + " disagrees with config img_processor num_img_tokens "
              + d.numImgTokens());
    }
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    if (tokenizer.vocabSize() > d.text().dimensions().vocabSize()) {
      throw new IllegalArgumentException("tokenizer vocabulary exceeds checkpoint vocabulary");
    }
    Map<String, MLXArray> loaded =
        CheckpointLoader.load(scope, directory, d.tensorPlan(clipLayers));
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
    for (String key : weights.keySet()) {
      if (!key.startsWith(Phi3VDescriptor.visionKey(""))) {
        textWeights.put(key, weights.get(key));
      }
    }
    ArchitectureMappings.tensorPlan(d.text()).validate(textWeights.keySet());
    String convKey =
        Phi3VDescriptor.visionKey("img_processor.vision_model.embeddings.patch_embedding.weight");
    MLXArray conv = weights.get(convKey);
    // PyTorch OIHW -> MLX OHWI; provenance is the descriptor's explicit HF key mapping.
    MLXArray convTransposed = MLXShape.transpose(conv, new int[] {0, 2, 3, 1});
    MLX.eval(convTransposed);
    conv.close();
    weights.put(convKey, convTransposed);
    Map<String, MLXArray> vision = new LinkedHashMap<>();
    Map<String, String> keyMap = d.visionKeys(clipLayers);
    Set<String> extracted = new HashSet<>();
    for (var entry : keyMap.entrySet()) {
      String full = Phi3VDescriptor.visionKey(entry.getKey());
      extracted.add(full);
      vision.put(entry.getValue(), weights.get(full));
    }
    // The plan-optional last layer and post_layernorm are loaded but never executed; release
    // their handles now (the reference loads and discards them as well).
    for (var entry : weights.entrySet()) {
      if (entry.getKey().startsWith(Phi3VDescriptor.visionKey(""))
          && !extracted.contains(entry.getKey())) {
        entry.getValue().close();
      }
    }
    int intermediate =
        weights.get(
                Phi3VDescriptor.visionKey(
                    "img_processor.vision_model.encoder.layers.0.mlp.fc1.weight"))
            .shape()[0];
    Phi3VVisionTower tower =
        new Phi3VVisionTower(
            scope, clipLayers, d.text().dimensions().hiddenSize(), intermediate, vision);
    return new Phi3VModel(
        d,
        new TextDecoder(scope, d.text(), textWeights),
        tower,
        new Phi3VImageProcessor(preprocessor),
        tokenizer,
        clipLayers);
  }

  /**
   * The checkpoint's CLIP encoder layer count, discovered from the shard tensor names without any
   * native work: the contiguous {@code encoder.layers.{i}.self_attn.q_proj.weight} run from layer
   * 0.
   *
   * @throws IllegalArgumentException when no CLIP encoder layer is present
   */
  private static int discoverClipLayers(Path directory) throws IOException {
    Set<String> names = new HashSet<>();
    for (Set<String> shardNames :
        SafetensorsHeaders.tensorNames(CheckpointLoader.shards(directory)).values()) {
      names.addAll(shardNames);
    }
    String prefix = Phi3VDescriptor.visionKey("img_processor.vision_model.encoder.layers.");
    int layers = 0;
    while (names.contains(prefix + layers + ".self_attn.q_proj.weight")) {
      layers++;
    }
    if (layers < 1) {
      throw new IllegalArgumentException("checkpoint has no CLIP encoder layers");
    }
    return layers;
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
    if (request.tokenizer() == null) {
      throw new IllegalArgumentException(
          "phi3_v generation requires a tokenizer-backed request: the retained prompt text is"
              + " needed to expand the image markers");
    }
    String promptText = request.promptText();
    if (promptText == null) {
      throw new IllegalArgumentException(
          "phi3_v generation requires a request with retained prompt text; pre-tokenized"
              + " requests carry none and are rejected");
    }
    if (request.tokenizer().vocabSize() > metadata.vocabSize()) {
      throw new IllegalArgumentException(
          "tokenizer vocabulary size "
              + request.tokenizer().vocabSize()
              + " exceeds checkpoint vocabulary size "
              + metadata.vocabSize());
    }
    Phi3VPromptPlan plan = plan(promptText, request.images());
    // Before the decoder's own resolution, which would reject sliding modes with its own
    // checkpoint-window error: the boundary re-prefill is not proven equivalent under sliding
    // retention for phi3_v.
    if (request.cachePolicy().mode() != GenerationCachePolicy.Mode.FULL) {
      throw new IllegalArgumentException(
          "sliding phi3_v retention is not yet supported (the boundary re-prefill equivalence"
              + " is unproven)");
    }
    KVCachePolicy cachePolicy = decoder.resolveCachePolicy(request.cachePolicy());
    plan.requireCacheCapacity(cachePolicy, request.config().maxNewTokens());
    // plan() bounds the expanded prompt alone; this additionally bounds the whole generation
    // (prompt plus the maximum new tokens), which is what the context window must hold.
    if (plan.requiredCacheCapacity(request.config().maxNewTokens()) > maxPositions) {
      throw new IllegalArgumentException(
          "generation exceeds context length: expanded prompt "
              + plan.expandedLength()
              + ", unexpanded "
              + original.length);
    }
    // The re-prefill history: the clamped expanded prompt (sentinel rows embed as row 0; their
    // rows are replaced by the hook anyway) plus every generated token, appended by the decode
    // step below.
    List<Integer> history = new ArrayList<>();
    for (int id : clampSentinels(plan.expandedIds())) {
      history.add(id);
    }
    final DecoderModel.EmbeddingHook[] hookRef = {null};
    try {
      return decoder.runStepLoop(
          request,
          request.config(),
          original,
          wrappedSentinels(plan.expandedIds()),
          plan.expandedLength(),
          cachePolicy,
          listener,
          (activation, caches) ->
              prefill(
                  request.cancellationToken(), request.images(), plan, activation, caches, hookRef),
          (ids, caches) -> decodeStep(ids, caches, history, hookRef));
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
        System.getLogger(Phi3VModel.class.getName())
            .log(System.Logger.Level.WARNING, "generation terminal listener failed", failure);
      }
      return result;
    }
  }

  /**
   * The pure expansion of one unexpanded prompt against the request's images: per-image geometry
   * and the aggregate slot budget from the configured processor (no pixels), the checkpoint
   * tokenizer's wrapper-grammar expansion, and the expanded-prompt context bound. A pre-cancelled
   * request is surfaced at the prefill's first boundary check instead, where the step loop converts
   * it to a cancelled result (the 7.3b precedent).
   *
   * <p>Package access lets the golden and lifecycle tests exercise this production boundary before
   * any native work.
   */
  Phi3VPromptPlan plan(String promptText, List<RgbImage> images) {
    Objects.requireNonNull(promptText, "promptText");
    Objects.requireNonNull(images, "images");
    List<Phi3VGeometryPlan> geometries = new ArrayList<>(images.size());
    long slots = 0;
    for (RgbImage image : images) {
      if (geometries.size() >= MAX_IMAGES) {
        throw new IllegalArgumentException("at most " + MAX_IMAGES + " images per request");
      }
      Phi3VGeometryPlan geometry = processor.geometry(image.width(), image.height());
      slots = Math.addExact(slots, geometry.cropBlocks() + 1L);
      if (slots > MAX_TOTAL_SLOTS) {
        throw new IllegalArgumentException(
            "request exceeds the aggregate image slot budget of " + MAX_TOTAL_SLOTS);
      }
      geometries.add(geometry);
    }
    Phi3VPromptPlan plan = Phi3VPromptExpander.expandRendered(tokenizer, promptText, geometries);
    if (plan.expandedLength() > maxPositions) {
      throw new IllegalArgumentException(
          "expanded prompt " + plan.expandedLength() + " exceeds context length " + maxPositions);
    }
    return plan;
  }

  /**
   * Runs and evaluates the request's first step, releasing all visual work before returning.
   * Package access lets the golden and lifecycle tests exercise this production boundary with
   * caller-owned KV; the hook reference carries the projected rows to the boundary re-prefill.
   *
   * <p>The caller must keep the activation scope's parent (the request's generation scope) open for
   * the whole request: when images are present, the stored hook retains the projected rows there,
   * since the boundary re-prefill re-injects them at a later decode step, after this step's
   * activation scope is closed.
   */
  DecoderModel.FirstStepResult prefill(
      CancellationToken cancellation,
      List<RgbImage> images,
      Phi3VPromptPlan plan,
      MLXScope activation,
      List<KVCache> caches,
      DecoderModel.EmbeddingHook[] hookRef) {
    checkCancellation(cancellation);
    Objects.requireNonNull(
        activation.parent(),
        "the prefill activation scope must be a child of the request's generation scope");
    int[][] before =
        decoder.preflightBatch(1, plan.expandedLength(), caches, new int[] {plan.expandedLength()});
    List<Phi3VGeometryPlan> geometries = new ArrayList<>(images.size());
    for (RgbImage image : images) {
      geometries.add(processor.geometry(image.width(), image.height()));
    }
    DecoderModel.FirstStepResult result;
    try (MLXScope prefill = activation.newChild()) {
      List<MLXArray> featureBlocks = new ArrayList<>();
      for (int i = 0; i < images.size(); i++) {
        checkCancellation(cancellation);
        observer.accept("preprocess");
        checkCancellation(cancellation);
        Phi3VImageProcessorResult prepared = processor.process(images.get(i));
        checkCancellation(cancellation);
        Phi3VGeometryPlan geometry = prepared.plan();
        Phi3VGeometryPlan planned = geometries.get(i);
        if (geometry.cropBlocks() != planned.cropBlocks()
            || geometry.gridH() != planned.gridH()
            || geometry.gridW() != planned.gridW()) {
          throw new IllegalArgumentException(
              "image " + i + " geometry changed between expansion and processing");
        }
        int realSlots = geometry.cropBlocks() + 1;
        try (MLXScope vision = prefill.newChild()) {
          observer.accept("vision");
          MLXArray pixels = pixels(vision, prepared, realSlots);
          MLXArray features = tower.forward(pixels);
          MLX.eval(features);
          featureBlocks.add(MLX.hoist(features, prefill));
        }
        observer.accept("image-closed");
      }
      checkCancellation(cancellation);
      DecoderModel.EmbeddingHook hook = null;
      if (!featureBlocks.isEmpty()) {
        observer.accept("project");
        // The per-image high-definition transform, then one projector call over all images'
        // concatenated pre-projector rows (the reference's single img_projection call).
        List<MLXArray> rows = new ArrayList<>();
        for (int i = 0; i < featureBlocks.size(); i++) {
          Phi3VGeometryPlan geometry = geometries.get(i);
          rows.add(tower.hdRows(featureBlocks.get(i), geometry.gridH(), geometry.gridW()));
        }
        MLXArray hd = MLXShape.concatenate(rows.toArray(MLXArray[]::new), 0);
        // The merge hook replaces rows in the 3D embedding, so the projector sees a [1, R, D]
        // view of the concatenated pre-projector rows.
        MLXArray projected =
            tower
                .projector()
                .forward(MLXShape.reshape(hd, new int[] {1, hd.shape()[0], hd.shape()[1]}));
        MLX.eval(projected);
        // Retained in the request's generation scope (the activation scope's parent), not this
        // step's activation scope: the boundary re-prefill re-injects the rows at a later decode
        // step, after the prefill step's activation scope has closed.
        MLXArray features = MLX.hoist(projected, activation.parent());
        hook = embedded -> merge(embedded, features, plan);
      }
      hookRef[0] = hook;
      observer.accept("before-prefill");
      checkCancellation(cancellation);
      MLXArray ids =
          MLX.array(
              prefill, clampSentinels(plan.expandedIds()), new int[] {1, plan.expandedLength()});
      MLXArray logits =
          decoder.stepLogits(ids, caches, new int[] {plan.expandedLength()}, hook, null, null);
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
   * One cached decode step: appends the emitted token to the re-prefill history and re-runs the
   * whole history as a fresh full prefill when that step first reaches {@code originalMaxPositions
   * + 1} total width (the reference's boundary recompute, which thereby selects the long factor
   * table), or otherwise decodes the single token. Package access lets the golden test drive the
   * loop step by step against the fresh-forward reference.
   */
  MLXArray decodeStep(
      MLXArray ids,
      List<KVCache> caches,
      List<Integer> history,
      DecoderModel.EmbeddingHook[] hookRef) {
    int token = ids.toIntArray()[0];
    history.add(token);
    observer.accept("decode");
    if (caches.get(0).nextPosition() == originalMaxPositions) {
      // The reference's boundary recompute: the step that first reaches original_max + 1
      // total width is re-run as a full prefill of the whole history, which selects the
      // long factor table (its width is strictly greater than original_max).
      observer.accept("re-prefill");
      for (KVCache cache : caches) {
        cache.reset();
      }
      int width = history.size();
      int[] full = new int[width];
      for (int i = 0; i < width; i++) {
        full[i] = history.get(i);
      }
      MLXArray fullIds = MLX.array(ids.scope(), full, new int[] {1, width});
      return decoder.stepLogits(fullIds, caches, new int[] {width}, hookRef[0], null, null);
    }
    return decoder.decode(ids, caches, null, null);
  }

  static MLXArray merge(MLXArray embedded, MLXArray features, Phi3VPromptPlan plan) {
    if (features.ndim() != 3
        || features.shape()[0] != 1
        || features.shape()[1] != plan.totalSentinelRows()
        || features.shape()[2] != embedded.shape()[2]
        || features.dtype() != embedded.dtype()) {
      throw new IllegalArgumentException(
          "projected image features disagree with prompt destinations");
    }
    // The embedded width, not the plan's: the boundary re-prefill re-embeds the whole history
    // (prompt plus generated tokens), and the gather must span it with the sentinel rows
    // replaced in place at their original positions.
    int width = embedded.shape()[1];
    int[] gather =
        Idefics3VisionTower.mergeGatherIndices(
            width, Arrays.stream(plan.sentinelDestinations()).boxed().toList());
    MLXArray source = MLXShape.concatenate(new MLXArray[] {embedded, features}, 1);
    return MLXShape.takeAxis(
        source, MLX.array(embedded.scope(), gather, new int[] {gather.length}), 1);
  }

  /**
   * The CLIP tower input for one image's real slots: the contiguous real-slot prefix of the
   * processor's slot-major CHW row, transposed to the tower's channels-last layout. Zero-padded
   * slots are never computed (the reference computes them and slices their rows away — a recorded
   * bit-identical deviation).
   */
  private static MLXArray pixels(
      MLXScope scope, Phi3VImageProcessorResult prepared, int realSlots) {
    int plane = CLIP_IMAGE * CLIP_IMAGE;
    float[] channelMajor =
        Arrays.copyOf(prepared.values(), Math.multiplyExact(realSlots, 3 * plane));
    MLXArray chw = MLX.array(scope, channelMajor, new int[] {realSlots, 3, CLIP_IMAGE, CLIP_IMAGE});
    return MLXShape.transpose(chw, new int[] {0, 2, 3, 1});
  }

  private static int[] clampSentinels(int[] ids) {
    int[] clamped = new int[ids.length];
    for (int i = 0; i < ids.length; i++) {
      clamped[i] = Math.max(ids[i], 0);
    }
    return clamped;
  }

  /**
   * The penalty frequency seed's sentinel rows: the reference keeps the {@code -i} sentinels in its
   * {@code input_ids} tensor, and torch's negative indexing wraps them to {@code vocab_size - i}
   * wherever the penalty gathers over that tensor.
   */
  private int[] wrappedSentinels(int[] ids) {
    int[] wrapped = new int[ids.length];
    for (int i = 0; i < ids.length; i++) {
      wrapped[i] = ids[i] < 0 ? metadata.vocabSize() + ids[i] : ids[i];
    }
    return wrapped;
  }

  /**
   * The lifecycle observer the generation tests use to observe (and fail at) each stage boundary.
   * Per request the events fire in order: per image {@code preprocess} (before any pixel or native
   * work), {@code vision} (inside the image's private child scope), {@code image-closed} (after it
   * closes); then, when any image was present, {@code project} (the shared high-definition
   * transform and projection); then {@code before-prefill} (before the decoder step), {@code
   * prefill-evaluated} (after the logits and every retained KV array are evaluated, before the
   * prefill scope closes) and {@code prefill-closed}; and per cached decode step {@code decode},
   * plus {@code re-prefill} at the single step that crosses the NTK-Su boundary.
   */
  void setObserverForTest(Consumer<String> observer) {
    this.observer = Objects.requireNonNull(observer);
  }

  DecoderModel decoderForTest() {
    return decoder;
  }

  Phi3VVisionTower towerForTest() {
    return tower;
  }

  Phi3VDescriptor descriptorForTest() {
    return descriptor;
  }

  private static final class PreparationCancelled extends RuntimeException {}

  private static void checkCancellation(CancellationToken cancellation) {
    if (cancellation.isCancelled()) {
      throw new PreparationCancelled();
    }
  }
}
