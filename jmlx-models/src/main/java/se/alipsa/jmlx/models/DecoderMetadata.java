package se.alipsa.jmlx.models;

/** Internal metadata implementation for the currently supported decoder architectures. */
record DecoderMetadata(String modelType, int vocabSize, int numHiddenLayers)
    implements ModelMetadata {
  DecoderMetadata {
    modelType = ModelTypes.requireValid(modelType);
    if (vocabSize <= 0 || numHiddenLayers <= 0) {
      throw new IllegalArgumentException("model metadata dimensions must be positive");
    }
  }
}
