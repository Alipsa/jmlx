package se.alipsa.jmlx.models;

record EncoderMetadata(String modelType, int vocabSize, int numHiddenLayers)
    implements ModelMetadata {
  @Override
  public int numEncoderLayers() {
    return numHiddenLayers;
  }
}
