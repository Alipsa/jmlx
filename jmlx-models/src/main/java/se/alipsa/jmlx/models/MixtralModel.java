package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads a Hugging Face Mixtral safetensors decoder checkpoint. */
public final class MixtralModel extends DecoderModel {
  private MixtralModel(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    super(
        scope,
        descriptor,
        CheckpointLoader.load(scope, directory, ArchitectureMappings.tensorPlan(descriptor)));
  }

  /** Loads a {@code mixtral} checkpoint from a local directory. */
  public static MixtralModel load(MLXScope scope, Path directory) throws IOException {
    return TextGenerationModels.loadDecoder(scope, directory, "mixtral", MixtralModel.class);
  }

  static MixtralModel create(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    return new MixtralModel(scope, descriptor, directory);
  }
}
