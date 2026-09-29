package se.alipsa.jmlx.models;

import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Pattern;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Attention;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Embedding;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Head;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Mlp;
import se.alipsa.jmlx.models.ArchitectureDescriptor.MlpLayout;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Moe;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Norm;
import se.alipsa.jmlx.models.ArchitectureDescriptor.NormKind;
import se.alipsa.jmlx.nn.Activation;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.JsonNode;

/** Maps Hugging Face configuration fields to validated decoder capabilities. */
public final class ArchitectureMappings {
  private static final System.Logger LOGGER =
      System.getLogger(ArchitectureMappings.class.getName());
  private static final Set<String> CONSUMED =
      Set.of(
          "model_type",
          "vocab_size",
          "hidden_size",
          "intermediate_size",
          "num_hidden_layers",
          "num_attention_heads",
          "num_key_value_heads",
          "rms_norm_eps",
          "rope_theta",
          "rope_parameters",
          "rope_scaling",
          "tie_word_embeddings",
          "attention_bias",
          "mlp_bias",
          "hidden_act",
          "hidden_activation",
          "num_local_experts",
          "num_experts_per_tok",
          "head_dim",
          "layer_types",
          "use_sliding_window",
          "sliding_window",
          "max_window_layers",
          "max_position_embeddings",
          "partial_rotary_factor");
  private static final Map<String, String> IGNORED =
      Map.ofEntries(
          Map.entry("architectures", "metadata only"),
          Map.entry("torch_dtype", "storage dtype"),
          Map.entry("dtype", "storage dtype"),
          Map.entry("transformers_version", "export metadata"),
          Map.entry("bos_token_id", "tokenizer setting"),
          Map.entry("eos_token_id", "generation setting"),
          Map.entry("pad_token_id", "generation setting"),
          Map.entry("initializer_range", "training only"),
          Map.entry("attention_dropout", "training only"),
          Map.entry("use_cache", "inference always caches"),
          Map.entry("pretraining_tp", "training only"),
          Map.entry("router_aux_loss_coef", "training only"),
          Map.entry("router_jitter_noise", "training only"),
          Map.entry("output_router_logits", "false during inference"));
  private static final Map<String, Function<JsonNode, ArchitectureDescriptor>> MAPPINGS =
      Map.of(
          "llama", node -> dense(node, false),
          "qwen2", node -> dense(node, true),
          "mistral", node -> dense(node, false),
          "phi3", node -> dense(node, false),
          "gemma", node -> dense(node, false),
          "mixtral", node -> dense(node, false));

  private ArchitectureMappings() {}

  /** Model types whose numerical configuration this loader understands. */
  public static Set<String> supportedModelTypes() {
    return MAPPINGS.keySet();
  }

  /** Expands a descriptor into its checkpoint tensor names. */
  public static TensorPlan tensorPlan(ArchitectureDescriptor descriptor) {
    Set<String> required = new HashSet<>();
    Set<String> optional = new HashSet<>();
    final Set<String> forbidden = new HashSet<>();
    required.add("model.embed_tokens.weight");
    required.add("model.norm.weight");
    (descriptor.head().tied() ? optional : required).add("lm_head.weight");
    for (int i = 0; i < descriptor.dimensions().numHiddenLayers(); i++) {
      String layer = "model.layers." + i + ".";
      required.add(layer + "input_layernorm.weight");
      required.add(layer + "post_attention_layernorm.weight");
      if (descriptor.attention().fusedQkv()) {
        required.add(layer + "self_attn.qkv_proj.weight");
        forbidden.add(layer + "self_attn.qkv_proj.bias");
        forbidden.add(layer + "self_attn.q_proj.weight");
        forbidden.add(layer + "self_attn.k_proj.weight");
        forbidden.add(layer + "self_attn.v_proj.weight");
        forbidden.add(layer + "self_attn.q_proj.bias");
        forbidden.add(layer + "self_attn.k_proj.bias");
        forbidden.add(layer + "self_attn.v_proj.bias");
      } else {
        forbidden.add(layer + "self_attn.qkv_proj.weight");
        forbidden.add(layer + "self_attn.qkv_proj.bias");
      }
      for (String projection :
          descriptor.attention().fusedQkv()
              ? Set.of("o_proj")
              : Set.of("q_proj", "k_proj", "v_proj", "o_proj")) {
        String key = layer + "self_attn." + projection;
        required.add(key + ".weight");
        boolean bias =
            "o_proj".equals(projection)
                ? descriptor.attention().outBias()
                : descriptor.attention().qkvBias();
        (bias ? required : forbidden).add(key + ".bias");
      }
      if (descriptor.moe() != null) {
        required.add(layer + "block_sparse_moe.gate.weight");
        for (int expert = 0; expert < descriptor.moe().experts(); expert++) {
          String prefix = layer + "block_sparse_moe.experts." + expert + ".";
          required.add(prefix + "w1.weight");
          required.add(prefix + "w2.weight");
          required.add(prefix + "w3.weight");
        }
        for (String projection : Set.of("gate_proj", "up_proj", "down_proj")) {
          forbidden.add(layer + "mlp." + projection + ".weight");
        }
        continue;
      }
      if (descriptor.mlp().layout() == MlpLayout.FUSED_GATE_UP) {
        required.add(layer + "mlp.gate_up_proj.weight");
        forbidden.add(layer + "mlp.gate_up_proj.bias");
        forbidden.add(layer + "mlp.gate_proj.weight");
        forbidden.add(layer + "mlp.up_proj.weight");
        forbidden.add(layer + "mlp.gate_proj.bias");
        forbidden.add(layer + "mlp.up_proj.bias");
      } else {
        forbidden.add(layer + "mlp.gate_up_proj.weight");
        forbidden.add(layer + "mlp.gate_up_proj.bias");
      }
      for (String projection :
          descriptor.mlp().layout() == MlpLayout.FUSED_GATE_UP
              ? Set.of("down_proj")
              : Set.of("gate_proj", "up_proj", "down_proj")) {
        String key = layer + "mlp." + projection;
        required.add(key + ".weight");
        (descriptor.mlp().bias() ? required : forbidden).add(key + ".bias");
      }
    }
    return new TensorPlan(
        required, optional, forbidden, Set.of(Pattern.compile(".*\\.rotary_emb\\.inv_freq$")));
  }

  /** Parses a configuration and warns about unrecognized keys. */
  public static ArchitectureDescriptor parse(JsonNode config) {
    return parse(
        config, key -> LOGGER.log(System.Logger.Level.WARNING, "unknown config key: " + key));
  }

  /** Parses a configuration, sending each unrecognized top-level key to {@code unknownKeySink}. */
  public static ArchitectureDescriptor parse(JsonNode config, Consumer<String> unknownKeySink) {
    Objects.requireNonNull(config, "config");
    Objects.requireNonNull(unknownKeySink, "unknownKeySink");
    String type = requiredText(config, "model_type");
    Function<JsonNode, ArchitectureDescriptor> mapping = MAPPINGS.get(type);
    if (mapping == null) {
      if ("gemma2".equals(type) || "gemma3".equals(type)) {
        throw new IllegalArgumentException(
            "config.json model_type '" + type + "' is deferred to a later milestone");
      }
      throw new IllegalArgumentException("config.json model_type '" + type + "' is unsupported");
    }
    rejectUnsupportedNumerics(config, type);
    ArchitectureDescriptor descriptor = mapping.apply(config);
    for (Map.Entry<String, JsonNode> entry : config.properties()) {
      String key = entry.getKey();
      if (!CONSUMED.contains(key) && !IGNORED.containsKey(key) && !key.endsWith("_pdrop")) {
        unknownKeySink.accept(key);
      }
    }
    return descriptor;
  }

  private static ArchitectureDescriptor dense(JsonNode node, boolean qwen2) {
    int hidden = requiredInt(node, "hidden_size");
    int heads = requiredInt(node, "num_attention_heads");
    String modelType = requiredText(node, "model_type");
    boolean gemma = "gemma".equals(modelType);
    if (gemma && !node.hasNonNull("head_dim")) {
      throw new IllegalArgumentException("config.json gemma requires head_dim");
    }
    if (!gemma
        && node.hasNonNull("head_dim")
        && (!node.get("head_dim").canConvertToInt()
            || node.get("head_dim").intValue() * heads != hidden)) {
      throw new IllegalArgumentException(
          "config.json head_dim * num_attention_heads must equal hidden_size");
    }
    String hiddenAct = node.path("hidden_act").asString("silu");
    if (!gemma && !"silu".equals(hiddenAct) && !"swish".equals(hiddenAct)) {
      throw new IllegalArgumentException(
          "config.json declares hidden_act '"
              + hiddenAct
              + "', but this decoder only implements silu");
    }
    if (node.path("use_sliding_window").asBoolean(false)) {
      throw new IllegalArgumentException(
          "config.json enables use_sliding_window, which this decoder does not implement");
    }
    if (!("mistral".equals(modelType) || "phi3".equals(modelType) || "mixtral".equals(modelType))
        && node.hasNonNull("sliding_window")) {
      if (!qwen2) {
        throw new IllegalArgumentException(
            "config.json sliding_window is unsupported for " + modelType);
      }
    }
    if (!qwen2 && node.hasNonNull("max_window_layers")) {
      throw new IllegalArgumentException(
          "config.json max_window_layers is unsupported for " + modelType);
    }
    if (!gemma && node.hasNonNull("hidden_activation")) {
      throw new IllegalArgumentException(
          "config.json hidden_activation is unsupported for " + modelType);
    }
    int layers = requiredInt(node, "num_hidden_layers");
    JsonNode layerTypes = node.get("layer_types");
    if (layerTypes != null && !layerTypes.isNull()) {
      if ("mistral".equals(modelType) || "phi3".equals(modelType)) {
        throw new IllegalArgumentException(
            "config.json layer_types schedules are unsupported for " + modelType);
      }
      if (!layerTypes.isArray() || layerTypes.size() != layers) {
        throw new IllegalArgumentException(
            "config.json layer_types length must match num_hidden_layers");
      }
      for (JsonNode layerType : layerTypes) {
        if (!"full_attention".equals(layerType.asString())) {
          throw new IllegalArgumentException(
              "config.json layer_types contains unsupported attention");
        }
      }
    }
    JsonNode parameters = node.get("rope_parameters");
    JsonNode legacyScaling = node.get("rope_scaling");
    if (parameters != null && !parameters.isNull() && !parameters.isObject()) {
      throw new IllegalArgumentException("config.json rope_parameters must be an object");
    }
    double topTheta = node.path("rope_theta").asDouble(10_000);
    JsonNode thetaSource = parameters != null && !parameters.isNull() ? parameters : legacyScaling;
    double nestedTheta =
        thetaSource == null ? topTheta : thetaSource.path("rope_theta").asDouble(topTheta);
    if (node.hasNonNull("rope_theta")
        && thetaSource != null
        && thetaSource.hasNonNull("rope_theta")
        && Double.compare(topTheta, nestedTheta) != 0) {
      throw new IllegalArgumentException(
          "config.json rope_theta conflicts with "
              + (thetaSource == parameters ? "rope_parameters" : "rope_scaling")
              + ".rope_theta");
    }
    float theta = (float) nestedTheta;
    int headDim = gemma ? requiredInt(node, "head_dim") : hidden / heads;
    float partial =
        (float)
            node.path("partial_rotary_factor")
                .asDouble(
                    thetaSource == null
                        ? 1
                        : thetaSource.path("partial_rotary_factor").asDouble(1));
    final int rotaryDims = (int) (headDim * partial);
    final RopeSpec rope = parseRope(node, theta);
    final float eps = (float) node.path("rms_norm_eps").asDouble(1e-6);
    boolean qkvBias = qwen2 || node.path("attention_bias").asBoolean(false);
    boolean mlpBias = node.path("mlp_bias").asBoolean(false);
    if ("phi3".equals(modelType) && (qkvBias || mlpBias)) {
      throw new IllegalArgumentException(
          "config.json phi3 attention_bias or mlp_bias is unsupported for fused projections");
    }
    if (gemma && (qkvBias || mlpBias)) {
      throw new IllegalArgumentException(
          "config.json gemma attention_bias or mlp_bias is unsupported");
    }
    if ("mixtral".equals(modelType) && mlpBias) {
      throw new IllegalArgumentException("config.json mixtral mlp_bias=true is unsupported");
    }
    Activation activation = Activation.SILU;
    if (gemma) {
      String gemmaActivation =
          node.path("hidden_activation").isNull()
              ? "gelu_pytorch_tanh"
              : node.path("hidden_activation").asString("gelu_pytorch_tanh");
      activation =
          switch (gemmaActivation) {
            case "gelu_pytorch_tanh", "gelu_new" -> Activation.GELU_TANH;
            case "gelu" -> Activation.GELU;
            default ->
                throw new IllegalArgumentException(
                    "config.json hidden_activation '" + gemmaActivation + "' is unsupported");
          };
    }
    Integer slidingWindow = null;
    if (("mistral".equals(modelType) || "phi3".equals(modelType) || "mixtral".equals(modelType))
        && node.hasNonNull("sliding_window")) {
      if (!node.get("sliding_window").canConvertToInt()) {
        throw new IllegalArgumentException("config.json sliding_window must be an integer");
      }
      slidingWindow = node.get("sliding_window").intValue();
    }
    DecoderConfig dimensions =
        new DecoderConfig(
            requiredText(node, "model_type"),
            requiredInt(node, "vocab_size"),
            hidden,
            requiredInt(node, "intermediate_size"),
            layers,
            heads,
            node.path("num_key_value_heads").asInt(heads),
            eps,
            theta,
            node.path("tie_word_embeddings").asBoolean(gemma),
            node.path("attention_bias").asBoolean(false),
            mlpBias);
    Moe moe =
        "mixtral".equals(modelType)
            ? new Moe(
                requiredInt(node, "num_local_experts"), requiredInt(node, "num_experts_per_tok"))
            : null;
    return new ArchitectureDescriptor(
        dimensions,
        headDim,
        rotaryDims,
        rope,
        new Norm(NormKind.RMS, eps, gemma),
        new Mlp(
            "phi3".equals(modelType) ? MlpLayout.FUSED_GATE_UP : MlpLayout.SEPARATE_GATE_UP,
            activation,
            mlpBias),
        new Attention(qkvBias, !qwen2 && qkvBias, "phi3".equals(modelType), slidingWindow),
        new Head(dimensions.tieWordEmbeddings()),
        new Embedding(gemma),
        moe);
  }

  private static void rejectUnsupportedNumerics(JsonNode node, String type) {
    if (node.hasNonNull("quantization") || node.hasNonNull("quantization_config")) {
      String key = node.hasNonNull("quantization") ? "quantization" : "quantization_config";
      throw new IllegalArgumentException(
          "config.json "
              + key
              + " declares quantized weights, which this decoder does not yet implement");
    }
    for (String key :
        Set.of(
            "num_local_experts",
            "num_experts_per_tok",
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "query_pre_attn_scalar")) {
      if (node.hasNonNull(key)) {
        if ("mixtral".equals(type)
            && ("num_local_experts".equals(key) || "num_experts_per_tok".equals(key))) {
          continue;
        }
        throw new IllegalArgumentException("config.json " + key + " is unsupported for " + type);
      }
    }
    if (node.path("output_router_logits").asBoolean(false)) {
      throw new IllegalArgumentException("config.json output_router_logits=true is unsupported");
    }
  }

  private static RopeSpec parseRope(JsonNode config, float theta) {
    JsonNode scaling = config.get("rope_scaling");
    JsonNode parameters = config.get("rope_parameters");
    if (scaling != null && !scaling.isNull() && !scaling.isObject()) {
      throw new IllegalArgumentException("config.json rope_scaling must be an object");
    }
    if (parameters != null && !parameters.isNull() && !parameters.isObject()) {
      throw new IllegalArgumentException("config.json rope_parameters must be an object");
    }
    if (scaling != null && !scaling.isNull() && parameters != null && !parameters.isNull()) {
      throw new IllegalArgumentException(
          "config.json rope_scaling and rope_parameters both present");
    }
    JsonNode source = scaling != null && !scaling.isNull() ? scaling : parameters;
    if (source == null || source.isNull()) {
      return new RopeSpec.Base(theta);
    }
    String prefix = source == scaling ? "rope_scaling" : "rope_parameters";
    String type = source.path("rope_type").asString(source.path("type").asString("default"));
    Set<String> keys =
        switch (type) {
          case "default" -> Set.of("rope_type", "type", "rope_theta", "partial_rotary_factor");
          case "linear", "dynamic" ->
              Set.of(
                  "rope_type",
                  "type",
                  "rope_theta",
                  "factor",
                  "original_max_position_embeddings",
                  "partial_rotary_factor");
          case "llama3" ->
              Set.of(
                  "rope_type",
                  "type",
                  "rope_theta",
                  "factor",
                  "low_freq_factor",
                  "high_freq_factor",
                  "original_max_position_embeddings",
                  "partial_rotary_factor");
          case "yarn" ->
              Set.of(
                  "rope_type",
                  "type",
                  "rope_theta",
                  "factor",
                  "original_max_position_embeddings",
                  "beta_fast",
                  "beta_slow",
                  "attention_factor",
                  "truncate",
                  "mscale",
                  "mscale_all_dim",
                  "partial_rotary_factor");
          default ->
              throw new IllegalArgumentException(
                  "config.json " + prefix + ".rope_type '" + type + "' is not supported");
        };
    for (Map.Entry<String, JsonNode> entry : source.properties()) {
      if (!keys.contains(entry.getKey())) {
        throw new IllegalArgumentException(
            "config.json " + prefix + "." + entry.getKey() + " is unsupported");
      }
    }
    if ("default".equals(type)) {
      return new RopeSpec.Base(theta);
    }
    float factor = requiredFloat(source, "factor", prefix);
    return switch (type) {
      case "linear" -> new RopeSpec.Linear(theta, factor);
      case "dynamic" ->
          new RopeSpec.DynamicNtk(theta, factor, optionalContext(source, config, prefix));
      case "llama3" ->
          new RopeSpec.Llama3(
              theta,
              factor,
              requiredFloat(source, "low_freq_factor", prefix),
              requiredFloat(source, "high_freq_factor", prefix),
              requiredInt(source, "original_max_position_embeddings"));
      case "yarn" ->
          new RopeSpec.Yarn(
              theta,
              factor,
              optionalContext(source, config, prefix),
              (float) source.path("beta_fast").asDouble(32),
              (float) source.path("beta_slow").asDouble(1),
              (float) source.path("mscale").asDouble(0),
              (float) source.path("mscale_all_dim").asDouble(0),
              source.hasNonNull("attention_factor")
                  ? requiredFloat(source, "attention_factor", prefix)
                  : null,
              source.path("truncate").asBoolean(true));
      default -> throw new AssertionError(type);
    };
  }

  private static int optionalContext(JsonNode source, JsonNode config, String prefix) {
    JsonNode context =
        source.hasNonNull("original_max_position_embeddings")
            ? source.get("original_max_position_embeddings")
            : config.get("max_position_embeddings");
    if (context == null || !context.canConvertToInt() || context.intValue() <= 0) {
      throw new IllegalArgumentException(
          "config.json "
              + prefix
              + ".original_max_position_embeddings or max_position_embeddings is required");
    }
    return context.intValue();
  }

  private static float requiredFloat(JsonNode node, String key, String prefix) {
    JsonNode value = node.get(key);
    if (value == null || !value.isNumber() || !Float.isFinite(value.floatValue())) {
      throw new IllegalArgumentException(
          "config.json " + prefix + "." + key + " must be a finite number");
    }
    return value.floatValue();
  }

  private static int requiredInt(JsonNode node, String name) {
    if (!node.has(name) || !node.get(name).canConvertToInt()) {
      throw new IllegalArgumentException("config.json is missing integer field '" + name + "'");
    }
    return node.get(name).intValue();
  }

  private static String requiredText(JsonNode node, String name) {
    if (!node.hasNonNull(name) || node.get(name).asString().isBlank()) {
      throw new IllegalArgumentException("config.json is missing string field '" + name + "'");
    }
    return node.get(name).asString();
  }
}
