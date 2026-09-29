package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code qwen2}. */
public final class QwenModel extends DecoderModel {
  private QwenModel(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    super(
        scope,
        descriptor,
        CheckpointLoader.load(scope, directory, ArchitectureMappings.tensorPlan(descriptor)));
  }

  /** Loads {@code directory}'s {@code config.json} and safetensors checkpoint shards. */
  public static QwenModel load(MLXScope scope, Path directory) throws IOException {
    return TextGenerationModels.loadDecoder(scope, directory, "qwen2", QwenModel.class);
  }

  static QwenModel create(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    return new QwenModel(scope, descriptor, directory);
  }
}
