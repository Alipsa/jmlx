package se.alipsa.jmlx.models;

import se.alipsa.jmlx.tokenizer.TokenizerEncoding;

/** Single-encoding inference using the caller-owned model scope. */
public interface TextEncoderModel extends AutoCloseable {
  /** Architecture-neutral metadata. */
  ModelMetadata metadata();

  /** Runs inference with artifact defaults. */
  EncoderResult encode(TokenizerEncoding input);

  /** Encodes with an explicit pooling policy. */
  EncoderResult encode(TokenizerEncoding input, Pooling pooling);

  /** Model lifetime is controlled by its caller-owned scope. */
  @Override
  default void close() {}
}
