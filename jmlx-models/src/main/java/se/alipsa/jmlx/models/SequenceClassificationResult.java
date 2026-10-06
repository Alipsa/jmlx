package se.alipsa.jmlx.models;

import java.util.List;

/** Ordered classifier logits and scores with defensive array copies. */
public record SequenceClassificationResult(float[] logits, float[] scores, List<String> labels) {
  /** Copies arrays and labels. */
  public SequenceClassificationResult {
    logits = logits.clone();
    scores = scores.clone();
    labels = List.copyOf(labels);
  }

  @Override
  public float[] logits() {
    return logits.clone();
  }

  @Override
  public float[] scores() {
    return scores.clone();
  }
}
