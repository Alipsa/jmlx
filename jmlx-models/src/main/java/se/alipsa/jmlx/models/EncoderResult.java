package se.alipsa.jmlx.models;

import java.util.Objects;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;

/** Hidden states retain every input row, including padded queries. */
public record EncoderResult(TokenizerEncoding encoding, float[][] hiddenStates, float[] embedding) {
  /** Copies every array and requires an encoding. */
  public EncoderResult {
    Objects.requireNonNull(encoding);
    hiddenStates = ResultArrays.copy(hiddenStates);
    embedding = embedding.clone();
  }

  @Override
  public float[][] hiddenStates() {
    return ResultArrays.copy(hiddenStates);
  }

  @Override
  public float[] embedding() {
    return embedding.clone();
  }
}
