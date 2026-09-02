package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code llama}. */
public final class LlamaModel extends DecoderModel {
  private LlamaModel(MLXScope scope, DecoderConfig config, Path directory) throws IOException {
    super(scope, requireLlama(config), CheckpointLoader.load(scope, directory));
  }

  public static LlamaModel load(MLXScope scope, Path directory) throws IOException {
    return new LlamaModel(
        scope, DecoderConfig.fromFile(directory.resolve("config.json")), directory);
  }

  private static DecoderConfig requireLlama(DecoderConfig config) {
    if (!"llama".equals(config.modelType()))
      throw new IllegalArgumentException("expected model_type llama, got " + config.modelType());
    return config;
  }
}
