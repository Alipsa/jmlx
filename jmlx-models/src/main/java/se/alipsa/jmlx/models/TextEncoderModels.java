package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Task-validated local safetensors loaders. */
public final class TextEncoderModels {
  private TextEncoderModels() {}

  record InputSpec(int vocab, int types, int positions, int limit, Pooling pooling) {
    void validate(se.alipsa.jmlx.tokenizer.TokenizerEncoding input, Pooling override) {
      Pooling effective = override == null ? pooling : override;
      BertModels.validateInput(
          input, effective.mode() == Pooling.Mode.CLS, vocab, types, positions, limit);
    }
  }

  static InputSpec inputSpec(Path directory) throws IOException {
    var config = JsonFiles.read(directory.resolve("config.json"));
    int positions = BertModels.positive(config, "max_position_embeddings");
    int hidden = BertModels.positive(config, "hidden_size");
    BertModels.Pipeline pipeline = BertModels.pipeline(directory, positions, hidden);
    return new InputSpec(
        BertModels.positive(config, "vocab_size"),
        BertModels.positive(config, "type_vocab_size"),
        positions,
        pipeline.limit(),
        pipeline.pooling());
  }

  /** Loads a supported BERT artifact into a caller-owned scope. */
  public static TextEncoderModel load(MLXScope scope, Path directory) throws IOException {
    return new BertModels.Encoder(BertModels.load(scope, directory, BertModels.Task.ENCODER));
  }
}
