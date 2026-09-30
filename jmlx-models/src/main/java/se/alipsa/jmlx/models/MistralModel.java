package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code mistral}. */
public final class MistralModel extends DecoderModel {
  private MistralModel(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    super(
        scope,
        descriptor,
        CheckpointLoader.load(scope, directory, ArchitectureMappings.tensorPlan(descriptor)));
  }

  /** Loads {@code directory}'s {@code config.json} and safetensors checkpoint shards. */
  public static MistralModel load(MLXScope scope, Path directory) throws IOException {
    return TextGenerationModels.loadDecoder(scope, directory, "mistral", MistralModel.class);
  }

  static MistralModel create(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    return new MistralModel(scope, descriptor, directory);
  }
}
