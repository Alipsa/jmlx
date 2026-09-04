package se.alipsa.jmlx.models;

import java.util.Objects;

/** Architecture-neutral metadata exposed by a loaded text-generation model. */
public record ModelMetadata(String modelType, int vocabSize, int numHiddenLayers) {
  /** Validates the stable metadata shared by decoder architectures. */
  public ModelMetadata {
    Objects.requireNonNull(modelType, "modelType");
    if (modelType.isBlank()) {
      throw new IllegalArgumentException("modelType must not be blank");
    }
    if (vocabSize <= 0 || numHiddenLayers <= 0) {
      throw new IllegalArgumentException("model metadata dimensions must be positive");
    }
  }
}
