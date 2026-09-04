package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import se.alipsa.jmlx.memory.MLXScope;

/** Common loader for the currently supported decoder checkpoint architectures. */
public final class TextGenerationModels {
  private TextGenerationModels() {}

  /** Loads a supported model using the {@code model_type} declared in {@code config.json}. */
  public static TextGenerationModel load(MLXScope scope, Path directory) throws IOException {
    return loadDecoder(scope, directory, null, DecoderModel.class);
  }

  static <T extends DecoderModel> T loadDecoder(
      MLXScope scope, Path directory, String expectedModelType, Class<T> expectedClass)
      throws IOException {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(expectedClass, "expectedClass");
    DecoderConfig config = DecoderConfig.fromFile(directory.resolve("config.json"));
    if (expectedModelType != null && !expectedModelType.equals(config.modelType())) {
      throw new IllegalArgumentException(
          "expected model_type " + expectedModelType + ", got " + config.modelType());
    }
    DecoderModel model =
        switch (config.modelType()) {
          case "llama" -> LlamaModel.create(scope, config, directory);
          case "qwen2" -> QwenModel.create(scope, config, directory);
          case String type ->
              throw new IllegalArgumentException("unsupported model_type '" + type + "'");
        };
    return expectedClass.cast(model);
  }
}
