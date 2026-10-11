package se.alipsa.jmlx.models;

import java.util.Set;

/** Metadata for an image-and-text conditional decoder. */
record VisionLanguageMetadata(
    String modelType, int vocabSize, int numHiddenLayers, int numEncoderLayers)
    implements ModelMetadata {
  @Override
  public Set<InputModality> inputModalities() {
    return Set.of(InputModality.TEXT, InputModality.IMAGE);
  }
}
