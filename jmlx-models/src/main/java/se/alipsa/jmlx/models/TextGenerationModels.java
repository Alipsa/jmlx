package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;

/** Common loader for the currently supported decoder checkpoint architectures. */
public final class TextGenerationModels {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private TextGenerationModels() {}

  /** Loads a supported model using the {@code model_type} declared in {@code config.json}. */
  public static TextGenerationModel load(MLXScope scope, Path directory) throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    return loadDecoder(scope, directory, readConfig(directory));
  }

  static <T extends DecoderModel> T loadDecoder(
      MLXScope scope, Path directory, String expectedModelType, Class<T> expectedClass)
      throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(expectedModelType, "expectedModelType");
    Objects.requireNonNull(expectedClass, "expectedClass");
    ArchitectureDescriptor descriptor = readConfig(directory);
    if (!expectedModelType.equals(descriptor.modelType())) {
      throw new IllegalArgumentException(
          "expected model_type " + expectedModelType + ", got " + descriptor.modelType());
    }
    // Keeps legacy typed entry points safe if a future descriptor mapping changes its model class.
    return expectedClass.cast(loadDecoder(scope, directory, descriptor));
  }

  private static DecoderModel loadDecoder(
      MLXScope scope, Path directory, ArchitectureDescriptor descriptor) throws IOException {
    Objects.requireNonNull(descriptor, "descriptor");
    return switch (descriptor.modelType()) {
      case "llama" -> LlamaModel.create(scope, descriptor, directory);
      case "qwen2" -> QwenModel.create(scope, descriptor, directory);
      case "mistral" -> MistralModel.create(scope, descriptor, directory);
      case "phi3" -> Phi3Model.create(scope, descriptor, directory);
      case "gemma" -> GemmaModel.create(scope, descriptor, directory);
      case "mixtral" -> MixtralModel.create(scope, descriptor, directory);
      case String type ->
          throw new IllegalArgumentException("unsupported model_type '" + type + "'");
    };
  }

  private static ArchitectureDescriptor readConfig(Path directory) throws IOException {
    Path file = directory.resolve("config.json");
    try {
      return ArchitectureMappings.parse(MAPPER.readTree(file.toFile()));
    } catch (JacksonException e) {
      throw new IOException("failed to read " + file.toAbsolutePath().normalize(), e);
    }
  }
}
