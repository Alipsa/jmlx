package se.alipsa.jmlx.models;

/**
 * One generated token, or the terminal event when {@link #finishReason()} is non-null. In this
 * release {@link #textDelta()} remains {@code null}; token events carry a log probability when the
 * request asks for one.
 */
public record GenerationEvent(
    Integer tokenId, String textDelta, Double logProbability, FinishReason finishReason) {
  /** Validates that this is either a token event or a terminal event, never both. */
  public GenerationEvent {
    if ((tokenId == null) == (finishReason == null)) {
      throw new IllegalArgumentException(
          "an event must contain exactly one of tokenId or finishReason");
    }
    if (finishReason != null && (textDelta != null || logProbability != null)) {
      throw new IllegalArgumentException("a terminal event cannot contain token data");
    }
  }

  /**
   * Creates a token event. Null text and log-probability fields are not implemented in this
   * release.
   */
  public static GenerationEvent token(int tokenId) {
    return new GenerationEvent(tokenId, null, null, null);
  }

  /** Creates a token event with its selected-token post-filter natural-log probability. */
  public static GenerationEvent token(int tokenId, double logProbability) {
    return new GenerationEvent(tokenId, null, logProbability, null);
  }

  /** Creates the single terminal event emitted by a completed generation. */
  public static GenerationEvent finished(FinishReason reason) {
    return new GenerationEvent(null, null, null, reason);
  }
}
