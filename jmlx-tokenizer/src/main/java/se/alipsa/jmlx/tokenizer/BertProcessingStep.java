package se.alipsa.jmlx.tokenizer;

import java.util.Objects;

/** BERT single-sequence post-processing tokens. */
record BertProcessingStep(ResolvedToken separator, ResolvedToken classification)
    implements PostProcessorStep {
  /** Validates both configured tokens. */
  BertProcessingStep {
    separator = Objects.requireNonNull(separator, "separator");
    classification = Objects.requireNonNull(classification, "classification");
  }
}
