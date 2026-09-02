package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
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
    boolean tieWordEmbeddings) {

  public DecoderConfig {
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
    JsonNode node = new ObjectMapper().readTree(file.toFile());
    return new DecoderConfig(
        requiredText(node, "model_type"),
        requiredInt(node, "vocab_size"),
        requiredInt(node, "hidden_size"),
        requiredInt(node, "intermediate_size"),
        requiredInt(node, "num_hidden_layers"),
        requiredInt(node, "num_attention_heads"),
        node.path("num_key_value_heads").asInt(requiredInt(node, "num_attention_heads")),
        (float) node.path("rms_norm_eps").asDouble(1e-6),
        (float) node.path("rope_theta").asDouble(10_000),
        node.path("tie_word_embeddings").asBoolean(false));
  }

  private static int requiredInt(JsonNode node, String name) {
    if (!node.has(name) || !node.get(name).canConvertToInt()) {
      throw new IllegalArgumentException("config.json is missing integer field '" + name + "'");
    }
    return node.get(name).intValue();
  }

  private static String requiredText(JsonNode node, String name) {
    if (!node.hasNonNull(name) || node.get(name).asText().isBlank()) {
      throw new IllegalArgumentException("config.json is missing string field '" + name + "'");
    }
    return node.get(name).asText();
  }
}
