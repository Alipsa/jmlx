package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code qwen2}. */
public final class QwenModel extends DecoderModel {
  private QwenModel(MLXScope scope, DecoderConfig config, Path directory) throws IOException {
    super(scope, requireQwen(config), CheckpointLoader.load(scope, directory));
  }

  public static QwenModel load(MLXScope scope, Path directory) throws IOException {
    return new QwenModel(
        scope, DecoderConfig.fromFile(directory.resolve("config.json")), directory);
  }

  private static DecoderConfig requireQwen(DecoderConfig config) {
    if (!"qwen2".equals(config.modelType()))
      throw new IllegalArgumentException("expected model_type qwen2, got " + config.modelType());
    return config;
  }
}
