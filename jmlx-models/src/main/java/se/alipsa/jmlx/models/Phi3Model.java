package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code phi3}. */
public final class Phi3Model extends DecoderModel {
  private Phi3Model(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    super(
        scope,
        descriptor,
        CheckpointLoader.load(scope, directory, ArchitectureMappings.tensorPlan(descriptor)));
  }

  /** Loads {@code directory}'s {@code config.json} and safetensors checkpoint shards. */
  public static Phi3Model load(MLXScope scope, Path directory) throws IOException {
    return TextGenerationModels.loadDecoder(scope, directory, "phi3", Phi3Model.class);
  }

  static Phi3Model create(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    return new Phi3Model(scope, descriptor, directory);
  }
}
