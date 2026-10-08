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
import java.util.Set;
import java.util.function.Consumer;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Activation;
import se.alipsa.jmlx.nn.AttentionMask;
import se.alipsa.jmlx.nn.BidirectionalAttention;
import se.alipsa.jmlx.nn.CrossAttention;
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.EncoderBlock;
import se.alipsa.jmlx.nn.GatedMlp;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.RMSNorm;
import se.alipsa.jmlx.nn.Sequential;
import se.alipsa.jmlx.nn.StaticKVCache;
import se.alipsa.jmlx.nn.UnaryLayer;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.IncrementalTokenDecoder;
import se.alipsa.jmlx.tokenizer.TokenizerException;
import tools.jackson.databind.JsonNode;

/** Float32 T5/Flan-T5 encoder-decoder inference with request-owned static encoder projections. */
public final class T5Model extends Module implements TextGenerationModel {
  private static final System.Logger LOGGER = System.getLogger(T5Model.class.getName());
  private final Config config;
  private final T5LoadOptions options;
  private final int startToken;
  private final Embedding shared;
  private final Embedding encoderBias;
  private final Embedding decoderBias;
  private final List<EncoderBlock> encoder = new ArrayList<>();
  private final List<DecoderLayer> decoder = new ArrayList<>();
  private final RMSNorm encoderNorm;
  private final RMSNorm decoderNorm;
  private final Linear head;
  // Null unless a test installs one, so the per-step frequency copy only happens when observed.
  private Consumer<Map<Integer, Integer>> penaltyObserver;

  void penaltyObserver(Consumer<Map<Integer, Integer>> observer) {
    penaltyObserver = Objects.requireNonNull(observer);
  }

  private ProjectionObserver observer = (layer, cache) -> {};

  @FunctionalInterface
  interface ProjectionObserver {
    void initialized(int layer, StaticKVCache cache);
  }

  void projectionObserver(ProjectionObserver observer) {
    this.observer = Objects.requireNonNull(observer);
  }

  private record Config(
      int vocab,
      int model,
      int kv,
      int heads,
      int ff,
      int encoderLayers,
      int decoderLayers,
      int buckets,
      int distance,
      float epsilon,
      boolean gated,
      boolean tied) {}

  /** Loads with the default source resource limit. */
  public static T5Model load(MLXScope scope, Path directory) throws IOException {
    return load(scope, directory, T5LoadOptions.defaults());
  }

  /** Loads supported dense T5 weights into the caller-owned scope. */
  public static T5Model load(MLXScope scope, Path directory, T5LoadOptions options)
      throws IOException {
    // Deliberately no null checks here: config validation must fail before any native call, and
    // constructing a scope is itself native (see EncoderContractsTest's null-scope loads). The
    // dispatching TextGenerationModels entry points check their arguments.
    return load(scope, directory, options, JsonFiles.read(directory.resolve("config.json")));
  }

  /**
   * Loads supported dense T5 weights, reusing a config tree the caller has already read -- avoiding
   * a second parse of {@code config.json} when the dispatching loader (e.g. {@link
   * TextGenerationModels}) has already read it to select on {@code model_type} (PR #39 review,
   * finding 7c).
   */
  static T5Model load(MLXScope scope, Path directory, T5LoadOptions options, JsonNode root)
      throws IOException {
    Objects.requireNonNull(options);
    Objects.requireNonNull(root, "root");
    Config config = parse(root);
    Path generationFile = directory.resolve("generation_config.json");
    JsonNode generation = Files.exists(generationFile) ? JsonFiles.read(generationFile) : null;
    JsonNode start = generation == null ? null : generation.get("decoder_start_token_id");
    if (start == null || start.isNull()) {
      start = root.get("decoder_start_token_id");
    }
    if (start == null
        || !start.isIntegralNumber()
        || !start.canConvertToInt()
        || start.intValue() < 0
        || start.intValue() >= config.vocab()) {
      throw new IllegalArgumentException("t5 requires an in-vocabulary decoder_start_token_id");
    }
    Map<String, int[]> shapes = shapes(config);
    Set<String> aliases =
        config.tied()
            ? Set.of("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight")
            : Set.of("encoder.embed_tokens.weight", "decoder.embed_tokens.weight");
    try (MLXScope staging = scope.newChild()) {
      Map<String, MLXArray> weights =
          CheckpointLoader.load(
              staging, directory, new TensorPlan(shapes.keySet(), aliases, Set.of(), Set.of()));
      for (var entry : shapes.entrySet()) {
        if (!Arrays.equals(weights.get(entry.getKey()).shape(), entry.getValue())) {
          throw new IllegalArgumentException("t5 tensor shape: " + entry.getKey());
        }
      }
      for (String alias : aliases) {
        if (!weights.containsKey(alias)) {
          continue;
        }
        MLXArray aliasWeight = weights.get(alias);
        if (!Arrays.equals(aliasWeight.shape(), new int[] {config.vocab(), config.model()})) {
          throw new IllegalArgumentException("t5 tied alias differs from shared.weight: " + alias);
        }
        // Compared on device so a tied-embedding copy never pulls the whole matrices to the host.
        MLXArray equal =
            MLX.astype(
                MLXOps.all(MLXOps.equal(aliasWeight, weights.get("shared.weight"))), DType.INT32);
        MLX.eval(equal);
        if (equal.toIntArray()[0] != 1) {
          throw new IllegalArgumentException("t5 tied alias differs from shared.weight: " + alias);
        }
      }
      Map<String, MLXArray> promoted = new LinkedHashMap<>();
      for (var entry : weights.entrySet()) {
        DType dtype = entry.getValue().dtype();
        if (dtype != DType.FLOAT32 && dtype != DType.FLOAT16 && dtype != DType.BFLOAT16) {
          throw new IllegalArgumentException("t5 requires unquantized floating weights");
        }
        if (!aliases.contains(entry.getKey())) {
          promoted.put(
              entry.getKey(), MLX.hoist(MLX.astype(entry.getValue(), DType.FLOAT32), scope));
        }
      }
      return new T5Model(scope, config, options, start.intValue(), promoted);
    }
  }

  private T5Model(
      MLXScope scope,
      Config config,
      T5LoadOptions options,
      int startToken,
      Map<String, MLXArray> weights) {
    super(scope);
    this.config = config;
    this.options = options;
    this.startToken = startToken;
    shared = child("shared", new Embedding(scope, weights.get("shared.weight")));
    encoderBias =
        child(
            "encoderBias",
            new Embedding(
                scope,
                weights.get(
                    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")));
    decoderBias =
        child(
            "decoderBias",
            new Embedding(
                scope,
                weights.get(
                    "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")));
    for (int i = 0; i < config.encoderLayers(); i++) {
      String name = "encoder.block." + i + ".layer.";
      encoder.add(
          child(
              "encoder" + i,
              new EncoderBlock(
                  scope,
                  attention(scope, config, weights, name + "0.SelfAttention"),
                  norm(scope, config, weights, name + "0.layer_norm"),
                  ffn(scope, config, weights, name + "1.DenseReluDense"),
                  norm(scope, config, weights, name + "1.layer_norm"),
                  true)));
    }
    for (int i = 0; i < config.decoderLayers(); i++) {
      decoder.add(
          child(
              "decoder" + i,
              new DecoderLayer(scope, config, weights, "decoder.block." + i + ".layer.")));
    }
    encoderNorm = child("encoderNorm", norm(scope, config, weights, "encoder.final_layer_norm"));
    decoderNorm = child("decoderNorm", norm(scope, config, weights, "decoder.final_layer_norm"));
    head = config.tied() ? null : child("head", linear(scope, weights, "lm_head"));
    train(false);
  }

  @Override
  public ModelMetadata metadata() {
    return new Seq2SeqMetadata(
        "t5", config.vocab(), config.decoderLayers(), config.encoderLayers());
  }

  private static Config parse(JsonNode root) {
    if (!"t5".equals(root.path("model_type").asString())
        || !root.path("is_encoder_decoder").asBoolean(false)
        || !root.path("architectures").isArray()
        || root.path("architectures").size() != 1
        || !"T5ForConditionalGeneration".equals(root.path("architectures").get(0).asString())
        || root.has("quantization_config")
        || !root.path("pruned_heads").isEmpty()) {
      throw new IllegalArgumentException("unsupported t5 architecture or quantization config");
    }
    String projection = root.path("feed_forward_proj").asString("relu");
    if (!List.of("relu", "gated-gelu").contains(projection)) {
      throw new IllegalArgumentException("unsupported t5 feed_forward_proj: " + projection);
    }
    // HF derives dense_act_fn and is_gated_act from feed_forward_proj, then lets explicit
    // config values override them. This loader only implements the derived pair (gated-gelu is
    // tanh-GELU), so reject configs whose explicit keys disagree instead of silently applying a
    // different activation than HF would.
    String act = projection.equals("gated-gelu") ? "gelu_new" : "relu";
    if (root.has("dense_act_fn") && !act.equals(root.path("dense_act_fn").asString())
        || root.has("is_gated_act")
            && root.path("is_gated_act").asBoolean() != projection.startsWith("gated")) {
      throw new IllegalArgumentException(
          "t5 dense_act_fn/is_gated_act conflict with feed_forward_proj");
    }
    int layers = positive(root, "num_layers", -1);
    int buckets = positive(root, "relative_attention_num_buckets", 32);
    int distance = positive(root, "relative_attention_max_distance", 128);
    double epsilon = root.path("layer_norm_epsilon").asDouble(1e-6);
    if (buckets < 4
        || buckets % 4 != 0
        || distance <= buckets / 2
        || !Double.isFinite(epsilon)
        || epsilon <= 0) {
      throw new IllegalArgumentException("invalid t5 buckets, relative distance or norm epsilon");
    }
    return new Config(
        positive(root, "vocab_size", -1),
        positive(root, "d_model", -1),
        positive(root, "d_kv", 64),
        positive(root, "num_heads", -1),
        positive(root, "d_ff", -1),
        layers,
        positive(root, "num_decoder_layers", layers),
        buckets,
        distance,
        (float) epsilon,
        projection.equals("gated-gelu"),
        root.path("tie_word_embeddings").asBoolean(true));
  }

  private static int positive(JsonNode root, String key, int fallback) {
    JsonNode value = root.get(key);
    int result =
        value == null
            ? fallback
            : value.isIntegralNumber() && value.canConvertToInt() ? value.intValue() : -1;
    if (result <= 0) {
      throw new IllegalArgumentException("t5 " + key + " must be positive integer");
    }
    return result;
  }

  private static Map<String, int[]> shapes(Config c) {
    Map<String, int[]> result = new LinkedHashMap<>();
    result.put("shared.weight", new int[] {c.vocab(), c.model()});
    if (!c.tied()) {
      result.put("lm_head.weight", new int[] {c.vocab(), c.model()});
    }
    int projection = Math.multiplyExact(c.heads(), c.kv());
    for (String stack : List.of("encoder", "decoder")) {
      boolean decoder = stack.equals("decoder");
      int layers = decoder ? c.decoderLayers() : c.encoderLayers();
      result.put(stack + ".final_layer_norm.weight", new int[] {c.model()});
      result.put(
          stack + ".block.0.layer.0.SelfAttention.relative_attention_bias.weight",
          new int[] {c.buckets(), c.heads()});
      for (int i = 0; i < layers; i++) {
        String prefix = stack + ".block." + i + ".layer.";
        for (int layer = 0; layer < (decoder ? 2 : 1); layer++) {
          String attention =
              prefix + layer + (layer == 0 ? ".SelfAttention." : ".EncDecAttention.");
          for (String name : List.of("q", "k", "v")) {
            result.put(attention + name + ".weight", new int[] {projection, c.model()});
          }
          result.put(attention + "o.weight", new int[] {c.model(), projection});
          result.put(prefix + layer + ".layer_norm.weight", new int[] {c.model()});
        }
        String feed = prefix + (decoder ? 2 : 1);
        result.put(feed + ".layer_norm.weight", new int[] {c.model()});
        for (String name : c.gated() ? List.of("wi_0", "wi_1") : List.of("wi")) {
          result.put(feed + ".DenseReluDense." + name + ".weight", new int[] {c.ff(), c.model()});
        }
        result.put(feed + ".DenseReluDense.wo.weight", new int[] {c.model(), c.ff()});
      }
    }
    return result;
  }

  private static Linear linear(MLXScope scope, Map<String, MLXArray> weights, String name) {
    return new Linear(scope, weights.get(name + ".weight"), null);
  }

  private static RMSNorm norm(
      MLXScope scope, Config c, Map<String, MLXArray> weights, String name) {
    return new RMSNorm(scope, weights.get(name + ".weight"), c.epsilon());
  }

  private static BidirectionalAttention attention(
      MLXScope scope, Config c, Map<String, MLXArray> weights, String name) {
    return new BidirectionalAttention(
        scope,
        c.heads(),
        c.kv(),
        1f,
        linear(scope, weights, name + ".q"),
        linear(scope, weights, name + ".k"),
        linear(scope, weights, name + ".v"),
        linear(scope, weights, name + ".o"));
  }

  private static UnaryLayer ffn(
      MLXScope scope, Config c, Map<String, MLXArray> weights, String name) {
    return c.gated()
        ? new GatedMlp(
            scope,
            linear(scope, weights, name + ".wi_0"),
            linear(scope, weights, name + ".wi_1"),
            linear(scope, weights, name + ".wo"),
            Activation.GELU_TANH)
        : new Sequential(
            scope,
            linear(scope, weights, name + ".wi"),
            Activation.RELU.layer(scope),
            linear(scope, weights, name + ".wo"));
  }

  static int relativeBucket(int relative, boolean bidirectional, int buckets, int distance) {
    long position = relative;
    int bucket = 0;
    if (bidirectional) {
      buckets /= 2;
      if (position > 0) {
        bucket = buckets;
      }
      position = Math.abs(position);
    } else {
      position = Math.max(-position, 0);
    }
    int exact = buckets / 2;
    if (position < exact) {
      return bucket + (int) position;
    }
    int logarithmic =
        exact
            + (int)
                (Math.log((double) position / exact)
                    / Math.log((double) distance / exact)
                    * (buckets - exact));
    return bucket + Math.min(logarithmic, buckets - 1);
  }

  private MLXArray bias(MLXScope scope, int queries, int keys, int offset, boolean bidirectional) {
    int[] buckets = new int[Math.multiplyExact(queries, keys)];
    for (int q = 0; q < queries; q++) {
      for (int k = 0; k < keys; k++) {
        buckets[q * keys + k] =
            relativeBucket(k - (q + offset), bidirectional, config.buckets(), config.distance());
      }
    }
    MLXArray ids = MLX.array(scope, buckets, new int[] {queries, keys});
    MLXArray table = (bidirectional ? encoderBias : decoderBias).forward(ids);
    return MLXShape.reshape(
        MLXShape.transpose(table, new int[] {2, 0, 1}),
        new int[] {1, config.heads(), queries, keys});
  }

  MLXArray encode(MLXScope scope, int[] source, int[] padding) {
    MLXArray x = shared.forward(MLX.array(scope, source, new int[] {1, source.length}));
    MLXArray mask = AttentionMask.bidirectional(scope, new int[][] {padding}, source.length);
    MLXArray bias = bias(scope, source.length, source.length, 0, true);
    for (EncoderBlock block : encoder) {
      x = block.forward(x, mask, bias);
    }
    return encoderNorm.forward(x);
  }

  List<StaticKVCache> project(MLXScope request, MLXArray encoded) {
    List<StaticKVCache> result = new ArrayList<>();
    for (int i = 0; i < decoder.size(); i++) {
      StaticKVCache cache = new StaticKVCache(request);
      try (MLXScope stage = request.newChild()) {
        MLXArray source = MLXShape.broadcastTo(encoded, stage, encoded.shape());
        decoder.get(i).cross.initialize(source, cache);
        observer.initialized(i, cache);
      }
      cache.validate(1, config.heads(), encoded.shape()[1], config.kv());
      result.add(cache);
    }
    return result;
  }

  MLXArray forward(
      MLXScope scope,
      int[] history,
      int sourceLength,
      int[] sourceMask,
      List<StaticKVCache> cross,
      List<KVCache> caches) {
    int offset = caches == null ? 0 : caches.getFirst().nextPosition();
    int keys = Math.addExact(offset, history.length);
    int[] flat = new int[Math.multiplyExact(history.length, keys)];
    for (int q = 0; q < history.length; q++) {
      for (int k = 0; k <= q + offset; k++) {
        flat[q * keys + k] = 1;
      }
    }
    MLXArray mask =
        MLX.astype(MLX.array(scope, flat, new int[] {1, 1, history.length, keys}), DType.BOOL);
    MLXArray sourcePadding =
        AttentionMask.bidirectional(scope, new int[][] {sourceMask}, history.length);
    MLXArray bias = bias(scope, history.length, keys, offset, false);
    MLXArray x = shared.forward(MLX.array(scope, history, new int[] {1, history.length}));
    for (int i = 0; i < decoder.size(); i++) {
      cross.get(i).validate(1, config.heads(), sourceLength, config.kv());
      x =
          decoder
              .get(i)
              .forward(
                  x,
                  mask,
                  bias,
                  sourcePadding,
                  cross.get(i),
                  caches == null ? null : caches.get(i));
    }
    x = decoderNorm.forward(x);
    if (config.tied()) {
      x =
          MLXOps.multiply(
              x,
              MLX.full(
                  scope, new int[] {1}, (float) Math.pow(config.model(), -0.5), DType.FLOAT32));
      return shared.project(x);
    }
    return head.forward(x);
  }

  @Override
  public GenerationResult generate(GenerationRequest request, Consumer<GenerationEvent> listener) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(listener, "listener");
    if (!request.images().isEmpty()) {
      throw new IllegalArgumentException("model_type t5 does not accept images");
    }
    GenerationConfig policy = request.config();
    int[] prompt = request.promptTokenIds();
    scope().checkAccess();
    DecoderModel.validateTokenIds(prompt, policy, config.vocab());
    if (prompt.length == 0 || prompt.length > options.maxSourceTokens()) {
      throw new IllegalArgumentException("t5 source exceeds maxSourceTokens or is empty");
    }
    if (request.cachePolicy().mode() != GenerationCachePolicy.Mode.FULL) {
      throw new IllegalArgumentException("t5 does not support " + request.cachePolicy().mode());
    }
    KVCachePolicy cachePolicy = request.cachePolicy().resolve(null);
    // A bounded FULL capacity must cover the target budget before any source encoding.
    cachePolicy.requireCapacity(policy.maxNewTokens(), "generation");
    HfTokenizer tokenizer = request.tokenizer();
    if (tokenizer != null && tokenizer.vocabSize() > config.vocab()) {
      throw new IllegalArgumentException(
          "tokenizer vocabulary size "
              + tokenizer.vocabSize()
              + " exceeds checkpoint vocabulary size "
              + config.vocab());
    }
    List<Integer> generated = new ArrayList<>();
    List<Double> logProbabilities = new ArrayList<>();
    StringBuilder generatedText = tokenizer == null ? null : new StringBuilder();
    IncrementalTokenDecoder textDecoder = null;
    LinkedHashMap<Integer, Integer> frequencies = PenaltyInputs.frequencies(new int[] {startToken});
    FinishReason reason = FinishReason.MAX_TOKENS;
    if (!request.cancellationToken().isCancelled() && policy.maxNewTokens() > 0) {
      textDecoder = tokenizer == null ? null : tokenizer.newIncrementalDecoder(true);
      try (MLXScope generation = scope().newChild();
          SamplingPipeline sampler =
              new SamplingPipeline(
                  generation, policy, config.vocab(), StepBoundaryEvaluator.NATIVE)) {
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < this.decoder.size(); i++) {
          caches.add(new KVCache(generation, cachePolicy));
        }
        int[] sourceMask = new int[prompt.length];
        Arrays.fill(sourceMask, 1);
        MLXArray encoded;
        try (MLXScope stage = generation.newChild()) {
          encoded = MLX.hoist(encode(stage, prompt, sourceMask), generation);
          MLX.eval(encoded);
        }
        if (request.cancellationToken().isCancelled()) {
          reason = FinishReason.CANCELLED;
        }
        List<StaticKVCache> cross =
            request.cancellationToken().isCancelled() ? List.of() : project(generation, encoded);
        int[] input = new int[] {startToken};
        for (int step = 0; step < policy.maxNewTokens(); step++) {
          if (request.cancellationToken().isCancelled()) {
            reason = FinishReason.CANCELLED;
            break;
          }
          try (MLXScope activation = generation.newChild()) {
            MLXArray logits = forward(activation, input, prompt.length, sourceMask, cross, caches);
            if (penaltyObserver != null) {
              penaltyObserver.accept(Map.copyOf(frequencies));
            }
            SamplingPipeline.Selection selection =
                sampler.select(
                    logits,
                    PenaltyInputs.from(frequencies, config.vocab()),
                    step,
                    cacheArrays(caches));
            int next = selection.tokenId();
            boolean eos = policy.eosTokenIds().contains(next);
            if (!eos && policy.stopTokenIds().contains(next)) {
              reason = FinishReason.STOP_TOKEN;
              break;
            }
            generated.add(next);
            frequencies.merge(next, 1, Integer::sum);
            if (selection.logProbability() != null) {
              logProbabilities.add(selection.logProbability());
            }
            String textDelta = null;
            if (textDecoder != null) {
              try {
                textDelta = textDecoder.append(next);
                generatedText.append(textDelta);
              } catch (TokenizerException e) {
                throw new GenerationAbortedException(
                    toList(prompt), generated, "output decoder", next, e);
              }
            }
            try {
              listener.accept(DecoderModel.tokenEvent(next, textDelta, selection.logProbability()));
            } catch (RuntimeException e) {
              throw new GenerationAbortedException(toList(prompt), generated, e);
            }
            if (eos) {
              reason = FinishReason.EOS;
              break;
            }
            input = new int[] {next};
          }
        }
      }
    } else if (request.cancellationToken().isCancelled()) {
      reason = FinishReason.CANCELLED;
    }
    String terminalDelta = null;
    if (tokenizer != null) {
      terminalDelta = "";
      if (textDecoder != null) {
        try {
          terminalDelta = textDecoder.finish();
        } catch (TokenizerException e) {
          throw new GenerationAbortedException(
              toList(prompt), generated, "output decoder finish", null, e);
        }
      }
      generatedText.append(terminalDelta);
    }
    GenerationResult result =
        new GenerationResult(
            toList(prompt),
            generated,
            reason,
            logProbabilities,
            generatedText == null ? null : generatedText.toString());
    try {
      listener.accept(
          terminalDelta == null
              ? GenerationEvent.finished(reason)
              : GenerationEvent.finished(reason, terminalDelta));
    } catch (RuntimeException e) {
      LOGGER.log(System.Logger.Level.WARNING, "generation terminal listener failed", e);
    }
    return result;
  }

  private static List<Integer> toList(int[] ids) {
    return Arrays.stream(ids).boxed().toList();
  }

  private static MLXArray[] cacheArrays(List<KVCache> caches) {
    MLXArray[] arrays = new MLXArray[caches.size() * 2];
    for (int i = 0; i < caches.size(); i++) {
      arrays[2 * i] = caches.get(i).keys();
      arrays[2 * i + 1] = caches.get(i).values();
    }
    return arrays;
  }

  private static final class DecoderLayer extends Module {
    private final BidirectionalAttention self;
    private final CrossAttention cross;
    private final RMSNorm selfNorm;
    private final RMSNorm crossNorm;
    private final RMSNorm ffNorm;
    private final UnaryLayer ff;

    DecoderLayer(MLXScope scope, Config c, Map<String, MLXArray> weights, String name) {
      super(scope);
      self = child("self", attention(scope, c, weights, name + "0.SelfAttention"));
      String crossName = name + "1.EncDecAttention";
      cross =
          child(
              "cross",
              new CrossAttention(
                  scope,
                  c.heads(),
                  c.kv(),
                  1f,
                  linear(scope, weights, crossName + ".q"),
                  linear(scope, weights, crossName + ".k"),
                  linear(scope, weights, crossName + ".v"),
                  linear(scope, weights, crossName + ".o")));
      selfNorm = child("selfNorm", norm(scope, c, weights, name + "0.layer_norm"));
      crossNorm = child("crossNorm", norm(scope, c, weights, name + "1.layer_norm"));
      ffNorm = child("ffNorm", norm(scope, c, weights, name + "2.layer_norm"));
      ff = child("ff", ffn(scope, c, weights, name + "2.DenseReluDense"));
    }

    MLXArray forward(
        MLXArray input,
        MLXArray mask,
        MLXArray bias,
        MLXArray sourceMask,
        StaticKVCache crossCache,
        KVCache selfCache) {
      MLXArray x = MLXOps.add(input, self.forward(selfNorm.forward(input), mask, bias, selfCache));
      x = MLXOps.add(x, cross.forward(crossNorm.forward(x), crossCache, sourceMask));
      return MLXOps.add(x, ff.forward(ffNorm.forward(x)));
    }
  }
}
