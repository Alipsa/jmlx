package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Loads Hugging Face safetensors checkpoints whose {@code model_type} is {@code qwen2}. */
public final class QwenModel extends DecoderModel {
  private QwenModel(MLXScope scope, DecoderConfig config, Path directory) throws IOException {
    // Qwen2 hardcodes q/k/v bias (true) and o_proj bias (false) in HF's modeling code; neither is
    // a config.json field, so config.attentionBias() must not drive either one here.
    super(scope, config, CheckpointLoader.load(scope, directory), true, false);
  }

  /** Loads {@code directory}'s {@code config.json} and safetensors checkpoint shards. */
  public static QwenModel load(MLXScope scope, Path directory) throws IOException {
    return TextGenerationModels.loadDecoder(scope, directory, "qwen2", QwenModel.class);
  }

  static QwenModel create(MLXScope scope, DecoderConfig config, Path directory) throws IOException {
    return new QwenModel(scope, config, directory);
  }
}
