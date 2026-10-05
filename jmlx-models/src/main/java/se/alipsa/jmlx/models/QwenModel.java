package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code qwen2} or {@code
 * qwen3}. Both families share the dense decoder stack; {@code qwen3} adds per-head QK normalization
 * and honors an explicit {@code head_dim}, expressed through the {@link ArchitectureDescriptor}
 * rather than this class.
 */
public final class QwenModel extends DecoderModel {
  private QwenModel(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    super(
        scope,
        descriptor,
        CheckpointLoader.load(scope, directory, ArchitectureMappings.tensorPlan(descriptor)));
  }

  /**
   * Loads a {@code qwen2} checkpoint from {@code directory}'s {@code config.json} and safetensors
   * shards. The {@code model_type} must be exactly {@code qwen2}; {@code qwen3} checkpoints load
   * through {@link TextGenerationModels#load(MLXScope, Path)} (or this class via a future typed
   * entry point), so a mistyped directory fails before any tensor is read.
   */
  public static QwenModel load(MLXScope scope, Path directory) throws IOException {
    return TextGenerationModels.loadDecoder(scope, directory, "qwen2", QwenModel.class);
  }

  static QwenModel create(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    return new QwenModel(scope, descriptor, directory);
  }
}
