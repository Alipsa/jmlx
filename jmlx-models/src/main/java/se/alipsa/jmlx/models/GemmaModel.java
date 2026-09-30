package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads a Hugging Face Gemma v1 safetensors decoder checkpoint. */
public final class GemmaModel extends DecoderModel {
  private GemmaModel(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    super(
        scope,
        descriptor,
        CheckpointLoader.load(scope, directory, ArchitectureMappings.tensorPlan(descriptor)));
  }

  /** Loads a {@code gemma} checkpoint from a local directory. */
  public static GemmaModel load(MLXScope scope, Path directory) throws IOException {
    return TextGenerationModels.loadDecoder(scope, directory, "gemma", GemmaModel.class);
  }

  static GemmaModel create(MLXScope scope, ArchitectureDescriptor descriptor, Path directory)
      throws IOException {
    return new GemmaModel(scope, descriptor, directory);
  }
}
