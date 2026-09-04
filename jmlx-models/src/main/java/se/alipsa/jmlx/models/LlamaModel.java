package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code llama}. */
public final class LlamaModel extends DecoderModel {
  private LlamaModel(MLXScope scope, DecoderConfig config, Path directory) throws IOException {
    // Llama's q/k/v and o_proj bias are both config.json's explicit attention_bias flag, not
    // hardcoded like Qwen2's.
    super(
        scope,
        requireLlama(config),
        CheckpointLoader.load(scope, directory),
        config.attentionBias(),
        config.attentionBias());
  }

  /** Loads {@code directory}'s {@code config.json} and safetensors checkpoint shards. */
  public static LlamaModel load(MLXScope scope, Path directory) throws IOException {
    DecoderConfig config = DecoderConfig.fromFile(directory.resolve("config.json"));
    requireLlama(config);
    return create(scope, config, directory);
  }

  static LlamaModel create(MLXScope scope, DecoderConfig config, Path directory)
      throws IOException {
    return new LlamaModel(scope, config, directory);
  }

  private static DecoderConfig requireLlama(DecoderConfig config) {
    if (!"llama".equals(config.modelType())) {
      throw new IllegalArgumentException("expected model_type llama, got " + config.modelType());
    }
    return config;
  }
}
