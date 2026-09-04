package se.alipsa.jmlx.models;

/**
 * Read-only, architecture-neutral metadata exposed by a loaded text-generation model.
 * Implementations are internal and may gain additional metadata as architectural support expands.
 */
public sealed interface ModelMetadata permits DecoderMetadata {
  /** The Hugging Face {@code model_type}. */
  String modelType();

  /** The vocabulary size. */
  int vocabSize();

  /** The number of decoder layers. */
  int numHiddenLayers();
}
