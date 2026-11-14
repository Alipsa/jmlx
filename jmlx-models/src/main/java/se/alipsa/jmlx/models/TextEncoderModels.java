package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Task-validated local safetensors loaders. */
public final class TextEncoderModels {
  private TextEncoderModels() {}

  static int inputLimit(Path directory) throws IOException {
    var config = JsonFiles.read(directory.resolve("config.json"));
    int positions = config.path("max_position_embeddings").asInt();
    int hidden = config.path("hidden_size").asInt();
    if (positions <= 0 || hidden <= 0) {
      throw new IllegalArgumentException("BERT requires positive position and hidden capacities");
    }
    return BertModels.pipeline(directory, positions, hidden).limit();
  }

  /** Loads a supported BERT artifact into a caller-owned scope. */
  public static TextEncoderModel load(MLXScope scope, Path directory) throws IOException {
    return new BertModels.Encoder(BertModels.load(scope, directory, BertModels.Task.ENCODER));
  }
}
