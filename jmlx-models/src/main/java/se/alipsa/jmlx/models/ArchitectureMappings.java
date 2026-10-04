package se.alipsa.jmlx.models;

import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Attention;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Embedding;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Head;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Mlp;
import se.alipsa.jmlx.models.ArchitectureDescriptor.MlpLayout;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Moe;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Norm;
import se.alipsa.jmlx.models.ArchitectureDescriptor.NormKind;
import se.alipsa.jmlx.models.ArchitectureDescriptor.Quantization;
import se.alipsa.jmlx.nn.Activation;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.JsonNode;

/** Maps Hugging Face configuration fields to validated decoder capabilities. */
public final class ArchitectureMappings {
  private static final System.Logger LOGGER =
      System.getLogger(ArchitectureMappings.class.getName());
  // Gemma v1 checkpoints ship hidden_act "gelu" next to hidden_activation "gelu_pytorch_tanh".
  // A non-null hidden_activation selects the activation (the variant Gemma was trained with); only
  // when it is missing or null does hidden_act decide. Hugging Face transformers 4.57 and 5.x
  // GemmaMLP read hidden_act alone, so a config carrying both keys with different values is the
  // one case where this loader deliberately differs from it.
  private static final Set<String> GEMMA_ACTIVATIONS =
      Set.of("gelu", "gelu_pytorch_tanh", "gelu_new");
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
          "partial_rotary_factor",
          "quantization",
          "quantization_config");
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
  private static final Map<String, Family> FAMILIES =
      Map.of(
          "llama",
          Family.builder(WindowPolicy.REJECT)
              .honorsAttentionBias()
              .honorsMlpBias()
              .acceptsLayerTypes()
              .build(),
          "qwen2",
          Family.builder(WindowPolicy.IGNORE)
              .qwen2Bias()
              .honorsAttentionBias()
              .acceptsMaxWindowLayers()
              .acceptsLayerTypes()
              .build(),
          "qwen3",
          Family.builder(WindowPolicy.IGNORE)
              .honorsAttentionBias()
              .honorsExplicitHeadDim()
              .acceptsMaxWindowLayers()
              .acceptsLayerTypes()
              .qkNorm()
              .build(),
          "mistral",
          Family.builder(WindowPolicy.USE).build(),
          "phi3",
          Family.builder(WindowPolicy.USE).fusedProjections().build(),
          "gemma",
          Family.builder(WindowPolicy.REJECT)
              .gemma()
              .honorsAttentionBias()
              .acceptsLayerTypes()
              .build(),
          "mixtral",
          Family.builder(WindowPolicy.USE).moe().acceptsLayerTypes().build());

  /** How a family treats a non-null {@code sliding_window} config field. */
  private enum WindowPolicy {
    REJECT,
    IGNORE,
    USE
  }

  /**
   * Config-level capabilities that differ between decoder families. Adding a family means adding
   * one entry to {@link #FAMILIES}; {@link #dense} never branches on a model type string.
   *
   * @param qwen2Bias qkv projections always carry a bias and the output projection never does
   * @param gemma explicit head_dim, offset RMSNorm, scaled embeddings, tied head, Gemma activations
   * @param fusedProjections fused qkv and gate/up projections
   * @param moe sparse mixture-of-experts MLP
   * @param honorsAttentionBias whether {@code attention_bias} turns on projection biases
   * @param honorsMlpBias whether {@code mlp_bias} turns on MLP projection biases
   * @param acceptsMaxWindowLayers whether {@code max_window_layers} is a recognised field
   * @param acceptsLayerTypes whether a {@code layer_types} schedule may be present
   * @param honorsExplicitHeadDim whether an explicit {@code head_dim} is honored when it
   *     differs from {@code hidden_size / num_attention_heads} (Qwen3; Gemma is stricter and
   *     requires the field outright)
   * @param qkNorm per-head QK normalization: {@code self_attn.q_norm.weight}/{@code
   *     k_norm.weight} over {@code head_dim} are required, after projection and before RoPE
   * @param window treatment of {@code sliding_window}
   */
  private record Family(
      boolean qwen2Bias,
      boolean gemma,
      boolean fusedProjections,
      boolean moe,
      boolean honorsAttentionBias,
      boolean honorsMlpBias,
      boolean acceptsMaxWindowLayers,
      boolean acceptsLayerTypes,
      boolean honorsExplicitHeadDim,
      boolean qkNorm,
      WindowPolicy window) {

    /** Rejects capability combinations the tensor plan and assembler cannot express. */
    Family {
      Objects.requireNonNull(window, "window");
      if (fusedProjections && (qwen2Bias || honorsAttentionBias || honorsMlpBias)) {
        throw new IllegalStateException("fused projections cannot carry biases");
      }
      if (moe && honorsMlpBias) {
        throw new IllegalStateException("mixture-of-experts experts cannot carry biases");
      }
    }

    static Builder builder(WindowPolicy window) {
      return new Builder(window);
    }

    /** Names each capability at the call site; every flag defaults to off. */
    private static final class Builder {
      private final WindowPolicy window;
      private boolean qwen2Bias;
      private boolean gemma;
      private boolean fusedProjections;
      private boolean moe;
      private boolean honorsAttentionBias;
      private boolean honorsMlpBias;
      private boolean acceptsMaxWindowLayers;
      private boolean acceptsLayerTypes;
      private boolean honorsExplicitHeadDim;
      private boolean qkNorm;

      private Builder(WindowPolicy window) {
        this.window = window;
      }

      Builder qwen2Bias() {
        qwen2Bias = true;
        return this;
      }

      Builder gemma() {
        gemma = true;
        return this;
      }

      Builder fusedProjections() {
        fusedProjections = true;
        return this;
      }

      Builder moe() {
        moe = true;
        return this;
      }

      Builder honorsAttentionBias() {
        honorsAttentionBias = true;
        return this;
      }

      Builder honorsMlpBias() {
        honorsMlpBias = true;
        return this;
      }

      Builder acceptsMaxWindowLayers() {
        acceptsMaxWindowLayers = true;
        return this;
      }

      Builder acceptsLayerTypes() {
        acceptsLayerTypes = true;
        return this;
      }

      Builder honorsExplicitHeadDim() {
        honorsExplicitHeadDim = true;
        return this;
      }

      Builder qkNorm() {
        qkNorm = true;
        return this;
      }

      Family build() {
        return new Family(
            qwen2Bias,
            gemma,
            fusedProjections,
            moe,
            honorsAttentionBias,
            honorsMlpBias,
            acceptsMaxWindowLayers,
            acceptsLayerTypes,
            honorsExplicitHeadDim,
            qkNorm,
            window);
      }
    }
  }

  private ArchitectureMappings() {}

  /** Model types whose numerical configuration this loader understands. */
  public static Set<String> supportedModelTypes() {
    return FAMILIES.keySet();
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
      // QK normalization (Qwen3) is float-only: the names do not match the affine allow-list, so
      // a quantized companion for them is rejected as an unexpected tensor.
      for (String norm : Set.of("q_norm", "k_norm")) {
        String key = layer + "self_attn." + norm;
        (descriptor.attention().qkNorm() ? required : forbidden).add(key + ".weight");
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
    if (descriptor.quantization() != null) {
      addQuantizedCompanions(required, optional);
      addQuantizedCompanions(optional, optional);
    }
    return new TensorPlan(
        required, optional, forbidden, Set.of(Pattern.compile(".*\\.rotary_emb\\.inv_freq$")));
  }

  /**
   * Every eligible weight ({@code *_proj.weight}, the embedding table, {@code lm_head.weight}) may
   * travel with {@code .scales} and {@code .biases}; norm weights stay float. The linear bias of a
   * projection is {@code .bias}, distinct from the quantization offsets {@code .biases}.
   */
  private static void addQuantizedCompanions(Set<String> keys, Set<String> optional) {
    for (String key : Set.copyOf(keys)) {
      boolean quantized =
          key.endsWith("_proj.weight")
              || key.equals("model.embed_tokens.weight")
              || key.equals("lm_head.weight");
      if (quantized) {
        String stem = key.substring(0, key.length() - ".weight".length());
        optional.add(stem + ".scales");
        optional.add(stem + ".biases");
      }
    }
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
    Family family = FAMILIES.get(type);
    if (family == null) {
      if ("gemma2".equals(type) || "gemma3".equals(type) || "qwen3_moe".equals(type)) {
        throw new IllegalArgumentException(
            "config.json model_type '" + type + "' is deferred to a later milestone");
      }
      throw new IllegalArgumentException("config.json model_type '" + type + "' is unsupported");
    }
    rejectUnsupportedNumerics(config, type, family.moe());
    ArchitectureDescriptor descriptor = dense(config, family);
    for (Map.Entry<String, JsonNode> entry : config.properties()) {
      String key = entry.getKey();
      if (!CONSUMED.contains(key) && !IGNORED.containsKey(key) && !key.endsWith("_pdrop")) {
        unknownKeySink.accept(key);
      }
    }
    return descriptor;
  }

  private static ArchitectureDescriptor dense(JsonNode node, Family family) {
    int hidden = requiredInt(node, "hidden_size");
    int heads = requiredInt(node, "num_attention_heads");
    final String modelType = requiredText(node, "model_type");
    final boolean qwen2 = family.qwen2Bias();
    boolean gemma = family.gemma();
    if (gemma && !node.hasNonNull("head_dim")) {
      throw new IllegalArgumentException("config.json gemma requires head_dim");
    }
    if (!gemma
        && !family.honorsExplicitHeadDim()
        && node.hasNonNull("head_dim")
        && (!node.get("head_dim").canConvertToInt()
            || node.get("head_dim").intValue() * heads != hidden)) {
      throw new IllegalArgumentException(
          "config.json head_dim * num_attention_heads must equal hidden_size");
    }
    String hiddenAct = node.path("hidden_act").asString(gemma ? "gelu_pytorch_tanh" : "silu");
    if (gemma
        ? !GEMMA_ACTIVATIONS.contains(hiddenAct)
        : !"silu".equals(hiddenAct) && !"swish".equals(hiddenAct)) {
      throw new IllegalArgumentException(
          "config.json declares hidden_act '"
              + hiddenAct
              + "', but this decoder only implements "
              + (gemma ? String.join(", ", new TreeSet<>(GEMMA_ACTIVATIONS)) : "silu"));
    }
    if (node.path("use_sliding_window").asBoolean(false)) {
      throw new IllegalArgumentException(
          "config.json enables use_sliding_window, which this decoder does not implement");
    }
    if (family.window() == WindowPolicy.REJECT && node.hasNonNull("sliding_window")) {
      throw new IllegalArgumentException(
          "config.json sliding_window is unsupported for " + modelType);
    }
    if (!family.acceptsMaxWindowLayers() && node.hasNonNull("max_window_layers")) {
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
      if (!family.acceptsLayerTypes()) {
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
    int headDim;
    if (node.hasNonNull("head_dim")) {
      // Gemma requires the field outright; honorsExplicitHeadDim families (Qwen3) accept a
      // value that differs from hidden_size / num_attention_heads; all other families must
      // satisfy the product check above.
      if (!node.get("head_dim").canConvertToInt()) {
        throw new IllegalArgumentException("config.json head_dim must be an integer");
      }
      headDim = node.get("head_dim").intValue();
    } else {
      headDim = hidden / heads;
    }
    double partial =
        node.path("partial_rotary_factor")
            .asDouble(
                thetaSource == null ? 1 : thetaSource.path("partial_rotary_factor").asDouble(1));
    final int rotaryDims = (int) (headDim * partial);
    final RopeSpec rope = parseRope(node, theta);
    final float eps = (float) node.path("rms_norm_eps").asDouble(1e-6);
    // Hugging Face hardcodes bias=False in several families' projections and ignores the flag
    // there.
    boolean attentionBias =
        family.honorsAttentionBias() && node.path("attention_bias").asBoolean(false);
    boolean qkvBias = qwen2 || attentionBias;
    final boolean mlpBias = family.honorsMlpBias() && node.path("mlp_bias").asBoolean(false);
    if (gemma && qkvBias) {
      throw new IllegalArgumentException(
          "config.json " + modelType + " attention_bias is unsupported");
    }
    Activation activation = Activation.SILU;
    if (gemma) {
      String gemmaActivation =
          node.hasNonNull("hidden_activation")
              ? node.get("hidden_activation").asString()
              : hiddenAct;
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
    if (family.window() == WindowPolicy.USE && node.hasNonNull("sliding_window")) {
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
            attentionBias,
            mlpBias);
    Moe moe =
        family.moe()
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
            family.fusedProjections() ? MlpLayout.FUSED_GATE_UP : MlpLayout.SEPARATE_GATE_UP,
            activation,
            mlpBias),
        new Attention(qkvBias, !qwen2 && qkvBias, family.fusedProjections(), slidingWindow,
            family.qkNorm()),
        new Head(dimensions.tieWordEmbeddings()),
        new Embedding(gemma),
        moe,
        parseQuantization(node, family.moe()));
  }

  private static void rejectUnsupportedNumerics(JsonNode node, String type, boolean moe) {
    for (String key :
        Set.of(
            "num_local_experts",
            "num_experts_per_tok",
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "query_pre_attn_scalar")) {
      if (node.hasNonNull(key)) {
        if (moe && ("num_local_experts".equals(key) || "num_experts_per_tok".equals(key))) {
          continue;
        }
        throw new IllegalArgumentException("config.json " + key + " is unsupported for " + type);
      }
    }
    if (node.path("output_router_logits").asBoolean(false)) {
      throw new IllegalArgumentException("config.json output_router_logits=true is unsupported");
    }
  }

  /**
   * Reads an MLX-style {@code quantization} block ({@code group_size}, {@code bits}). Only the
   * default affine mode is supported, and not for mixture-of-experts, whose stacked expert tensors
   * have no quantized path yet. A per-layer override in the block is rejected rather than ignored:
   * every layer would silently be read with the global setting. Both values must be present
   * integral JSON numbers: a fractional value such as {@code 64.5} is rejected rather than
   * truncated, and a missing one is rejected rather than defaulted, so a malformed config cannot
   * select different packing parameters than it declares.
   */
  private static Quantization parseQuantization(JsonNode node, boolean moe) {
    JsonNode block = node.hasNonNull("quantization") ? node.get("quantization") : null;
    JsonNode alias =
        node.hasNonNull("quantization_config") ? node.get("quantization_config") : null;
    if (block == null && alias == null) {
      return null;
    }
    String key = block != null ? "quantization" : "quantization_config";
    JsonNode q = block != null ? block : alias;
    if (!q.isObject()) {
      throw new IllegalArgumentException("config.json " + key + " must be an object");
    }
    if (block != null && alias != null && !block.equals(alias)) {
      throw new IllegalArgumentException(
          "config.json quantization and quantization_config disagree");
    }
    if (q.hasNonNull("quant_method")) {
      throw new IllegalArgumentException(
          "config.json "
              + key
              + ".quant_method '"
              + q.get("quant_method").asString()
              + "' is unsupported; only MLX affine quantization is implemented");
    }
    if (q.hasNonNull("mode") && !"affine".equals(q.get("mode").asString())) {
      throw new IllegalArgumentException(
          "config.json " + key + ".mode '" + q.get("mode").asString() + "' is unsupported");
    }
    for (var entry : q.properties()) {
      String name = entry.getKey();
      if (!name.equals("group_size") && !name.equals("bits") && !name.equals("mode")) {
        throw new IllegalArgumentException(
            "config.json "
                + key
                + "."
                + name
                + " is unsupported (per-layer quantization overrides are not implemented)");
      }
    }
    if (moe) {
      throw new IllegalArgumentException(
          "config.json " + key + " is unsupported for mixture-of-experts models");
    }
    JsonNode groupSize = q.path("group_size");
    JsonNode bits = q.path("bits");
    // isIntegralNumber covers "missing" (a MissingNode is not a number) as well as fractional,
    // so both failure shapes are described by the message below.
    if (!groupSize.isIntegralNumber() || !groupSize.canConvertToInt()) {
      throw new IllegalArgumentException(
          "config.json " + key + ".group_size is missing or non-integer");
    }
    if (!bits.isIntegralNumber() || !bits.canConvertToInt()) {
      throw new IllegalArgumentException("config.json " + key + ".bits is missing or non-integer");
    }
    return new Quantization(groupSize.intValue(), bits.intValue());
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
    float factor =
        "yarn".equals(type) && source.has("factor") && source.get("factor").isNull()
            ? yarnFactorFromContext(source, config, prefix)
            : requiredFloat(source, "factor", prefix);
    return switch (type) {
      case "linear" -> new RopeSpec.Linear(theta, factor);
      case "dynamic" ->
          new RopeSpec.DynamicNtk(theta, factor, requiredInt(config, "max_position_embeddings"));
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
              orDefault(source, "beta_fast", 32, prefix),
              orDefault(source, "beta_slow", 1, prefix),
              (float) source.path("mscale").asDouble(0),
              (float) source.path("mscale_all_dim").asDouble(0),
              source.hasNonNull("attention_factor")
                  ? requiredFloat(source, "attention_factor", prefix)
                  : null,
              source.path("truncate").asBoolean(true));
      default -> throw new AssertionError(type);
    };
  }

  // Hugging Face writes `get(key) or default`, so null and 0 both select the default.
  private static float orDefault(JsonNode node, String key, float fallback, String prefix) {
    JsonNode value = node.get(key);
    if (value == null || value.isNull() || (value.isNumber() && value.doubleValue() == 0)) {
      return fallback;
    }
    return requiredFloat(node, key, prefix);
  }

  // Hugging Face derives a null YaRN factor from max_position_embeddings and the original context.
  private static float yarnFactorFromContext(JsonNode source, JsonNode config, String prefix) {
    int original = optionalContext(source, config, prefix);
    JsonNode max = config.get("max_position_embeddings");
    if (max == null || !max.canConvertToInt() || max.intValue() <= 0) {
      throw new IllegalArgumentException(
          "config.json " + prefix + ".factor is null and max_position_embeddings is missing");
    }
    return (float) max.intValue() / original;
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
