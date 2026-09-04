package se.alipsa.jmlx.models;

/**
 * One generated token, or the terminal event when {@link #finishReason()} is non-null. In this
 * release {@link #textDelta()} and {@link #logProbability()} are always {@code null}: a
 * tokenizer-aware streaming adapter and log probabilities arrive in later Phase 6 milestones.
 */
public record GenerationEvent(
    Integer tokenId, String textDelta, Double logProbability, FinishReason finishReason) {
  /** Validates that this is either a token event or a terminal event, never both. */
  public GenerationEvent {
    if ((tokenId == null) == (finishReason == null)) {
      throw new IllegalArgumentException(
          "an event must contain exactly one of tokenId or finishReason");
    }
  }

  /**
   * Creates a token event. Null text and log-probability fields are not implemented in this
   * release.
   */
  public static GenerationEvent token(int tokenId) {
    return new GenerationEvent(tokenId, null, null, null);
  }

  /** Creates the single terminal event emitted by a completed generation. */
  public static GenerationEvent finished(FinishReason reason) {
    return new GenerationEvent(null, null, null, reason);
  }
}
