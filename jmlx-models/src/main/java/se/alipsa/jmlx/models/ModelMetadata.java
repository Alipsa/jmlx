package se.alipsa.jmlx.models;

/**
 * Read-only, architecture-neutral metadata exposed by loaded encoder, classification and generation
 * models. Implementations are internal and may gain additional metadata as architectural support
 * expands.
 */
public sealed interface ModelMetadata permits DecoderMetadata, EncoderMetadata, Seq2SeqMetadata {
  /** The Hugging Face {@code model_type}. */
  String modelType();

  /** The vocabulary size. */
  int vocabSize();

  /** Decoder depth for generation models, encoder depth for encoder-only models. */
  int numHiddenLayers();

  /** Encoder depth; zero for decoder-only models. */
  default int numEncoderLayers() {
    return 0;
  }
}
