package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;

/** Ordered sentence embeddings and the number of encoded input tokens, including special tokens. */
public record EmbeddingBatch(List<float[]> vectors, int inputTokens) {
  /** Takes owned copies of all vectors. */
  public EmbeddingBatch {
    Objects.requireNonNull(vectors, "vectors");
    vectors = vectors.stream().map(float[]::clone).toList();
    if (inputTokens < 0) {
      throw new IllegalArgumentException("inputTokens must not be negative");
    }
  }

  @Override
  public List<float[]> vectors() {
    return vectors.stream().map(float[]::clone).toList();
  }
}
