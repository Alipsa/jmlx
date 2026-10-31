package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.JsonNode;

/** Common loader for the currently supported decoder checkpoint architectures. */
public final class TextGenerationModels {

  private TextGenerationModels() {}

  /** Loads a supported model using the {@code model_type} declared in {@code config.json}. */
  public static TextGenerationModel load(MLXScope scope, Path directory) throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    JsonNode root = readConfigTree(directory);
    if ("t5".equals(root.path("model_type").asString())) {
      // Reuse the tree we just read to dispatch, so config.json is parsed once (PR #39 review,
      // finding 7c).
      return T5Model.load(scope, directory, T5LoadOptions.defaults(), root);
    }
    if ("idefics3".equals(root.path("model_type").asString())) {
      return SmolVlmModel.load(scope, directory, root);
    }
    String modelType = root.path("model_type").asString();
    if (modelType.startsWith("qwen3_vl")) {
      // The composite loader owns its own strict config parse, so dispatch on the raw model_type
      // before the generic decoder mapping: only the exact dense type loads, and a sibling type
      // (qwen3_vl_moe or a future variant) is rejected by name rather than misrouted.
      if (!"qwen3_vl".equals(modelType)) {
        throw new IllegalArgumentException(
            "model_type '"
                + modelType
                + "' is unsupported: only the dense qwen3_vl architecture loads");
      }
      return Qwen3VlModel.load(scope, directory, root);
    }
    if ("phi3_v".equals(root.path("model_type").asString())) {
      // The composite loader owns its own strict config parse (the phi3_v decoder fields plus the
      // embedded img_processor/embd_layer blocks), so dispatch on the raw model_type before the
      // generic decoder mapping, which cannot build the vision tower.
      return Phi3VModel.load(scope, directory, root);
    }
    return loadDecoder(scope, directory, ArchitectureMappings.parse(root));
  }

  /** Loads T5 with an explicit source limit; other architectures reject these options. */
  public static TextGenerationModel load(MLXScope scope, Path directory, T5LoadOptions options)
      throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(options);
    JsonNode root = readConfigTree(directory);
    if (!"t5".equals(root.path("model_type").asString())) {
      throw new IllegalArgumentException("T5LoadOptions applies only to model_type=t5");
    }
    return T5Model.load(scope, directory, options, root);
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
    return JsonFiles.read(directory.resolve("config.json"));
  }
}
