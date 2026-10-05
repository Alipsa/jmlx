package se.alipsa.jmlx.models;

import java.util.List;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;

/** Ordered classifier logits and scores with defensive array copies. */
public record TokenClassificationResult(
    TokenizerEncoding encoding, float[][] logits, float[][] scores, List<String> labels) {
  /** Copies arrays and labels. */
  public TokenClassificationResult {
    logits = ResultArrays.copy(logits);
    scores = ResultArrays.copy(scores);
    labels = List.copyOf(labels);
  }

  @Override
  public float[][] logits() {
    return ResultArrays.copy(logits);
  }

  @Override
  public float[][] scores() {
    return ResultArrays.copy(scores);
  }
}
