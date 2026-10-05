package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import se.alipsa.jmlx.memory.MLXScope;

/** Task-validated local safetensors loaders. */
public final class TokenClassifiers {
  private TokenClassifiers() {}

  /** Loads a supported BERT artifact into a caller-owned scope. */
  public static TokenClassifier load(MLXScope scope, Path directory) throws IOException {
    return new BertModels.Token(BertModels.load(scope, directory, BertModels.Task.TOKEN));
  }
}
