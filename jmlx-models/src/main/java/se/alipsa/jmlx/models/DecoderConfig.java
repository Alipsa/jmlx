package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Architecture fields shared by Hugging Face Llama and Qwen2 decoder checkpoints. */
public record DecoderConfig(
    String modelType,
    int vocabSize,
    int hiddenSize,
    int intermediateSize,
    int numHiddenLayers,
    int numAttentionHeads,
    int numKeyValueHeads,
    float rmsNormEps,
    float ropeTheta,
    boolean tieWordEmbeddings,
    boolean attentionBias,
    boolean mlpBias) {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** Validates decoder dimensions and the attention-head configuration. */
  public DecoderConfig {
    modelType = ModelTypes.requireValid(modelType);
    if (vocabSize <= 0 || hiddenSize <= 0 || intermediateSize <= 0 || numHiddenLayers <= 0) {
      throw new IllegalArgumentException("decoder dimensions must be positive");
    }
    if (numAttentionHeads <= 0
        || numKeyValueHeads <= 0
        || hiddenSize % numAttentionHeads != 0
        || numAttentionHeads % numKeyValueHeads != 0) {
      throw new IllegalArgumentException("invalid attention-head configuration");
    }
  }

  /** Reads the relevant, stable architecture fields from a Hugging Face {@code config.json}. */
  public static DecoderConfig fromFile(Path file) throws IOException {
    JsonNode node;
    try {
      node = MAPPER.readTree(file.toFile());
    } catch (JacksonException e) {
      throw new IOException("failed to read " + file.toAbsolutePath().normalize(), e);
    }
    if (node.hasNonNull("rope_scaling")) {
      throw new IllegalArgumentException(
          "config.json declares rope_scaling, which this decoder does not yet implement");
    }
    if (node.hasNonNull("quantization") || node.hasNonNull("quantization_config")) {
      throw new IllegalArgumentException(
          "config.json declares quantized weights, which this decoder does not yet implement");
    }
    if (node.path("use_sliding_window").asBoolean(false)) {
      throw new IllegalArgumentException(
          "config.json enables sliding-window attention, which this decoder does not implement");
    }
    String hiddenAct = node.path("hidden_act").asString("silu");
    // HF's ACT2CLS maps both "silu" and "swish" to nn.SiLU -- the same function SwiGLU
    // implements, not two different activations.
    if (!"silu".equals(hiddenAct) && !"swish".equals(hiddenAct)) {
      throw new IllegalArgumentException(
          "config.json declares hidden_act '"
              + hiddenAct
              + "', but this decoder only implements"
              + " silu");
    }
    int hiddenSize = requiredInt(node, "hidden_size");
    int heads = requiredInt(node, "num_attention_heads");
    if (node.hasNonNull("head_dim")
        && (!node.get("head_dim").canConvertToInt()
            || node.get("head_dim").intValue() * heads != hiddenSize)) {
      throw new IllegalArgumentException(
          "config.json head_dim * num_attention_heads must equal hidden_size");
    }
    return new DecoderConfig(
        requiredText(node, "model_type"),
        requiredInt(node, "vocab_size"),
        hiddenSize,
        requiredInt(node, "intermediate_size"),
        requiredInt(node, "num_hidden_layers"),
        heads,
        node.path("num_key_value_heads").asInt(heads),
        (float) node.path("rms_norm_eps").asDouble(1e-6),
        (float) node.path("rope_theta").asDouble(10_000),
        node.path("tie_word_embeddings").asBoolean(false),
        node.path("attention_bias").asBoolean(false),
        node.path("mlp_bias").asBoolean(false));
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
