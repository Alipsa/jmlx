package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Common decoder dimensions; file loading requires a supported {@code model_type}. */
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
    return ArchitectureMappings.parse(node).dimensions();
  }
}
