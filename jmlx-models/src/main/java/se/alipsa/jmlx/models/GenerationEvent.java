package se.alipsa.jmlx.models;

/** One generated token, or the terminal event when {@link #finishReason()} is non-null. */
public record GenerationEvent(
    Integer tokenId, String textDelta, Double logProbability, FinishReason finishReason) {
  /** Creates a token event. Text decoding is added by tokenizer-aware adapters. */
  public static GenerationEvent token(int tokenId) {
    return new GenerationEvent(tokenId, "", null, null);
  }

  /** Creates the single terminal event emitted by a completed generation. */
  public static GenerationEvent finished(FinishReason reason) {
    return new GenerationEvent(null, "", null, reason);
  }
}
