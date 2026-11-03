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
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.nn.MultiAxisRotation;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.Qwen3VlGeometryPlan;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessor;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/** Dense Qwen3.5 hybrid generation with request-owned recurrent/full state and optional vision. */
public final class Qwen35Model implements TextGenerationModel {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private final MLXScope scope;
  private final Qwen35Descriptor descriptor;
  private final Qwen35Decoder decoder;
  private final Qwen3VlVisionTower tower;
  private final Qwen3VlImageProcessor processor;
  private final ModelMetadata metadata;
  private StepBoundaryEvaluator evaluator = StepBoundaryEvaluator.NATIVE;
  private Consumer<String> observer = ignored -> {};

  private Qwen35Model(
      MLXScope scope, Qwen35Weights.Loaded loaded, Qwen3VlImageProcessor processor) {
    this.scope = scope;
    descriptor = loaded.checkpoint().descriptor();
    decoder = new Qwen35Decoder(scope, descriptor, loaded.tensors());
    this.processor = processor;
    if (loaded.checkpoint().imageCapable()) {
      Map<String, MLXArray> vision = new LinkedHashMap<>();
      loaded
          .tensors()
          .forEach(
              (key, value) -> {
                if (key.startsWith("vision.")) {
                  String mapped =
                      Qwen3VlVisionTower.visionKeyToTowerKey(
                          "model.visual." + key.substring("vision.".length()),
                          descriptor.vision().depth(),
                          new int[0]);
                  if (mapped == null || vision.putIfAbsent(mapped, value) != null) {
                    throw new IllegalArgumentException(
                        "Qwen3.5 invalid vision parameter remap: " + key);
                  }
                }
              });
      tower = new Qwen3VlVisionTower(scope, descriptor.vision().heads(), vision, new int[0]);
      metadata =
          new VisionLanguageMetadata(
              "qwen3_5",
              descriptor.text().vocabulary(),
              descriptor.text().layers(),
              descriptor.vision().depth());
    } else {
      tower = null;
      metadata =
          new DecoderMetadata(
              "qwen3_5", descriptor.text().vocabulary(), descriptor.text().layers());
    }
  }

  /** Loads strict HF or converted float/affine storage after complete native-free preflight. */
  public static Qwen35Model load(MLXScope scope, Path directory) throws IOException {
    return load(scope, directory, JsonFiles.read(directory.resolve("config.json")));
  }

  static Qwen35Model load(MLXScope scope, Path directory, JsonNode config) throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    Qwen35Checkpoint.Checked checked = Qwen35Checkpoint.inspect(directory, config);
    Qwen35Descriptor d = checked.descriptor();
    Qwen3VlImageProcessor processor =
        checked.imageCapable() ? processor(directory, d.vision(), checked.converted()) : null;
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
    requireTokenizer(tokenizer, d);
    return new Qwen35Model(scope, Qwen35Weights.load(scope, checked), processor);
  }

  private static Qwen3VlImageProcessor processor(
      Path directory, Qwen35Descriptor.Vision vision, boolean converted) throws IOException {
    Path file = directory.resolve("preprocessor_config.json");
    byte[] bytes;
    if (Files.isRegularFile(file)) {
      bytes = Files.readAllBytes(file);
    } else {
      // Transformers 5 saves this observed nested processor form; the image contract is identical.
      JsonNode root = JsonFiles.read(directory.resolve("processor_config.json"));
      JsonNode image = root.get("image_processor");
      if (image == null || !image.isObject()) {
        throw new IllegalArgumentException("Qwen3.5 processor_config.image_processor is required");
      }
      bytes = MAPPER.writeValueAsBytes(image);
    }
    JsonNode imageConfig = MAPPER.readTree(bytes);
    if (converted
        && imageConfig.path("image_processor_type").asString().equals("Qwen3VLImageProcessor")) {
      // mlx-vlm 0.6.8 serializes its numpy/PIL processor with explicit legacy pixel bounds.
      // Its image path always resizes with PIL bicubic; adapt only this observed serialization.
      ObjectNode normalized = ((ObjectNode) imageConfig).deepCopy();
      for (String bound : new String[] {"min_pixels", "max_pixels"}) {
        JsonNode value = normalized.get(bound);
        if (value == null
            || !value.isIntegralNumber()
            || !value.canConvertToInt()
            || value.asInt() <= 0) {
          throw new IllegalArgumentException(
              "Qwen3.5 converted processor requires positive " + bound);
        }
      }
      if ((normalized.has("do_resize")
              && (!normalized.get("do_resize").isBoolean()
                  || !normalized.get("do_resize").asBoolean()))
          || (normalized.has("resample")
              && (!normalized.get("resample").isIntegralNumber()
                  || normalized.get("resample").asInt() != 3))) {
        throw new IllegalArgumentException("Qwen3.5 converted processor requires bicubic resizing");
      }
      if (!normalized.has("size")) {
        normalized
            .putObject("size")
            .put("shortest_edge", normalized.get("min_pixels").asInt())
            .put("longest_edge", normalized.get("max_pixels").asInt());
      }
      normalized.put("image_processor_type", "Qwen2VLImageProcessor");
      normalized.put("do_resize", true);
      normalized.put("resample", 3);
      bytes = MAPPER.writeValueAsBytes(normalized);
    }
    Qwen3VlProcessorConfig config = Qwen3VlProcessorConfig.fromJson(bytes);
    if (config.patchSize() != vision.patch()
        || config.temporalPatchSize() != vision.temporalPatch()
        || config.mergeSize() != vision.merge()
        || !Arrays.equals(config.imageMean(), new float[] {.5f, .5f, .5f})
        || !Arrays.equals(config.imageStd(), new float[] {.5f, .5f, .5f})) {
      throw new IllegalArgumentException(
          "Qwen3.5 processor geometry or normalization disagrees with config");
    }
    return new Qwen3VlImageProcessor(config);
  }

  private static void requireTokenizer(HfTokenizer tokenizer, Qwen35Descriptor d) {
    if (tokenizer.vocabSize() > d.text().vocabulary()) {
      throw new IllegalArgumentException(
          "Qwen3.5 tokenizer vocabulary exceeds checkpoint vocabulary");
    }
    Qwen3VlPromptTokens declared = d.tokens();
    try {
      Qwen3VlPromptTokens resolved =
          Qwen3VlPromptTokens.resolve(
              tokenizer,
              declared.imageTokenId(),
              declared.videoTokenId(),
              declared.visionStartTokenId(),
              declared.visionEndTokenId());
      if (!resolved.equals(declared)) {
        throw new IllegalArgumentException("Qwen3.5 tokenizer sentinels disagree with config");
      }
    } catch (RuntimeException error) {
      throw new IllegalArgumentException(
          "Qwen3.5 tokenizer does not match checkpoint sentinels", error);
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
    if (request.cachePolicy().mode() != GenerationCachePolicy.Mode.FULL) {
      throw new IllegalArgumentException("Qwen3.5 sliding retention is not supported");
    }
    if (!request.images().isEmpty() && tower == null) {
      throw new IllegalArgumentException("Qwen3.5 text-only checkpoint does not accept images");
    }
    int[] original = request.promptTokenIds();
    DecoderModel.validateTokenIds(original, request.config(), descriptor.text().vocabulary());
    if (request.tokenizer() != null) {
      requireTokenizer(request.tokenizer(), descriptor);
    }
    Qwen3VlPromptPlan plan = plan(original, request.images());
    long required =
        Math.max(
            plan.expandedLength(), plan.requiredCacheCapacity(request.config().maxNewTokens()));
    KVCachePolicy policy = request.cachePolicy().resolve(null);
    policy.requireCapacity(required, "Qwen3.5 generation expanded prompt/history");
    if (required > descriptor.text().context()) {
      throw new IllegalArgumentException("Qwen3.5 generation exceeds max_position_embeddings");
    }
    return GenerationLoop.run(
        scope,
        descriptor.text().vocabulary(),
        evaluator,
        request,
        request.config(),
        original,
        plan.expandedIds(),
        plan.expandedLength(),
        listener,
        generation -> new RequestState(generation, policy, request, plan));
  }

  Qwen3VlPromptPlan plan(int[] ids, List<RgbImage> images) {
    if (!images.isEmpty() && processor == null) {
      throw new IllegalArgumentException("Qwen3.5 text-only checkpoint does not accept images");
    }
    List<Qwen3VlGeometryPlan> geometries = new ArrayList<>();
    for (RgbImage image : images) {
      geometries.add(processor.geometry(image.width(), image.height()));
    }
    Qwen3VlPromptPlan plan = Qwen3VlPromptExpander.expand(ids, descriptor.tokens(), geometries);
    if (plan.expandedLength() > descriptor.text().context()) {
      throw new IllegalArgumentException("Qwen3.5 expanded prompt exceeds max_position_embeddings");
    }
    return plan;
  }

  private final class RequestState implements GenerationLoop.State {
    private final Qwen35Decoder.Session session;
    private final GenerationRequest request;
    private final Qwen3VlPromptPlan plan;
    private MLXScope visualScope;

    RequestState(
        MLXScope generation,
        KVCachePolicy policy,
        GenerationRequest request,
        Qwen3VlPromptPlan plan) {
      session = decoder.newSession(generation, policy);
      session.checkpoint(() -> checkCancellation(request.cancellationToken()));
      this.request = request;
      this.plan = plan;
    }

    @Override
    public MLXArray firstStep(MLXScope activation) {
      checkCancellation(request.cancellationToken());
      MLXScope work = activation;
      MLXArray features = null;
      if (!request.images().isEmpty()) {
        visualScope = activation.newChild();
        work = visualScope;
        features = features(request.cancellationToken(), request.images(), work);
      }
      observer.accept("before-prefill");
      checkCancellation(request.cancellationToken());
      int length = plan.expandedLength();
      MLXArray ids = MLX.array(work, plan.expandedIds(), new int[] {1, length});
      MultiAxisRotation rotation =
          descriptor
              .text()
              .rope()
              .prepare(
                  work,
                  1,
                  length,
                  new int[][] {
                    plan.temporalPositions(), plan.heightPositions(), plan.widthPositions()
                  });
      final MLXArray rows = features;
      return session.build(
          ids,
          rotation,
          true,
          rows == null
              ? null
              : embedded ->
                  VisualFeatureInjection.replaceImageRows(
                      embedded,
                      MLXShape.reshape(
                          rows, new int[] {1, plan.totalFeatureRows(), descriptor.text().hidden()}),
                      plan.featureDestinations()));
    }

    @Override
    public MLXArray decodeStep(MLXScope activation, int token) {
      checkCancellation(request.cancellationToken());
      observer.accept("decode");
      int position = plan.ropePositionAt(session.position());
      MLXArray ids = MLX.array(activation, new int[] {token}, new int[] {1, 1});
      return session.build(
          ids,
          descriptor
              .text()
              .rope()
              .prepare(activation, 1, 1, new int[][] {{position}, {position}, {position}}),
          true);
    }

    @Override
    public MLXArray[] retainedArrays() {
      return session.retainedArrays();
    }

    @Override
    public void commitStep() {
      if (visualScope != null) {
        observer.accept("prefill-evaluated");
        closeVisual();
        observer.accept("prefill-closed");
      }
      session.commit();
    }

    @Override
    public void onStepFailure() {
      session.poison();
      closeVisual();
    }

    private void closeVisual() {
      if (visualScope != null) {
        visualScope.close();
        visualScope = null;
      }
    }

    @Override
    public void close() {
      try {
        closeVisual();
      } finally {
        session.close();
      }
    }
  }

  MLXArray features(CancellationToken cancellation, List<RgbImage> images, MLXScope target) {
    List<MLXArray> parts = new ArrayList<>();
    for (RgbImage image : images) {
      checkCancellation(cancellation);
      observer.accept("preprocess");
      checkCancellation(cancellation);
      try (MLXScope perImage = target.newChild()) {
        var processed = processor.process(image);
        observer.accept("vision");
        checkCancellation(cancellation);
        var output = tower.forward(perImage, List.of(processed));
        parts.add(MLX.hoist(output.merged(), target));
      }
      observer.accept("image-closed");
      checkCancellation(cancellation);
    }
    return Qwen3VlVisionTower.concatenate(parts);
  }

  private static void checkCancellation(CancellationToken token) {
    if (token.isCancelled()) {
      throw new GenerationLoop.Cancelled();
    }
  }

  void observerForTest(Consumer<String> observer) {
    this.observer = Objects.requireNonNull(observer);
  }

  void evaluatorForTest(StepBoundaryEvaluator evaluator) {
    this.evaluator = Objects.requireNonNull(evaluator);
  }

  Qwen35Decoder decoderForTest() {
    return decoder;
  }

  Qwen3VlVisionTower towerForTest() {
    return tower;
  }

  Qwen3VlImageProcessor processorForTest() {
    return processor;
  }

  Qwen35Descriptor descriptorForTest() {
    return descriptor;
  }
}
