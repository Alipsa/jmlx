package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Common loader for the currently supported decoder checkpoint architectures. */
public final class TextGenerationModels {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private TextGenerationModels() {}

  /** Loads a supported model using the {@code model_type} declared in {@code config.json}. */
  public static TextGenerationModel load(MLXScope scope, Path directory) throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    JsonNode root = readConfigTree(directory);
    if ("t5".equals(root.path("model_type").asString())) {
      return T5Model.load(scope, directory);
    }
    return loadDecoder(scope, directory, ArchitectureMappings.parse(root));
  }

  /** Loads T5 with an explicit source limit; other architectures reject these options. */
  public static TextGenerationModel load(MLXScope scope, Path directory, T5LoadOptions options)
      throws IOException {
    Objects.requireNonNull(options);
    if (!"t5".equals(readConfigTree(directory).path("model_type").asString())) {
      throw new IllegalArgumentException("T5LoadOptions applies only to model_type=t5");
    }
    return T5Model.load(scope, directory, options);
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
      case "qwen3" -> QwenModel.create(scope, descriptor, directory);
      case "mistral" -> MistralModel.create(scope, descriptor, directory);
      case "phi3" -> Phi3Model.create(scope, descriptor, directory);
      case "gemma" -> GemmaModel.create(scope, descriptor, directory);
      case "mixtral" -> MixtralModel.create(scope, descriptor, directory);
      case String type ->
          throw new IllegalArgumentException("unsupported model_type '" + type + "'");
    };
  }

  private static ArchitectureDescriptor readConfig(Path directory) throws IOException {
    return ArchitectureMappings.parse(readConfigTree(directory));
  }

  /**
   * Reads {@code config.json} exactly once, wrapping Jackson 3's unchecked parse and I/O failures
   * in the checked {@link IOException} this package's loaders contract on.
   */
  private static JsonNode readConfigTree(Path directory) throws IOException {
    Path file = directory.resolve("config.json");
    try {
      return MAPPER.readTree(file.toFile());
    } catch (JacksonException e) {
      throw new IOException("failed to read " + file.toAbsolutePath().normalize(), e);
    }
  }
}
