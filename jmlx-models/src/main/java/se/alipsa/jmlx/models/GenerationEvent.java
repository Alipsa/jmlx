package se.alipsa.jmlx.models;

import java.util.Objects;

/**
 * One generated token, or the terminal event when {@link #finishReason()} is non-null. A
 * tokenizer-backed request carries a non-null (possibly empty) decoded delta; a pretokenized
 * request retains null text. Token events carry a log probability when the request asks for one.
 */
public record GenerationEvent(
    Integer tokenId, String textDelta, Double logProbability, FinishReason finishReason) {
  /** Validates that this is either a token event or a terminal event, never both. */
  public GenerationEvent {
    if ((tokenId == null) == (finishReason == null)) {
      throw new IllegalArgumentException(
          "an event must contain exactly one of tokenId or finishReason");
    }
    if (finishReason != null && logProbability != null) {
      throw new IllegalArgumentException("a terminal event cannot contain a log probability");
    }
  }

  /** Creates a pretokenized token event without text or log probability. */
  public static GenerationEvent token(int tokenId) {
    return new GenerationEvent(tokenId, null, null, null);
  }

  /** Creates a token event with its selected-token post-filter natural-log probability. */
  public static GenerationEvent token(int tokenId, double logProbability) {
    return new GenerationEvent(tokenId, null, logProbability, null);
  }

  /** Creates a decoded token event. */
  public static GenerationEvent token(int tokenId, String textDelta) {
    return new GenerationEvent(tokenId, Objects.requireNonNull(textDelta), null, null);
  }

  /** Creates a decoded token event with a selected-token log probability. */
  public static GenerationEvent token(int tokenId, String textDelta, double logProbability) {
    return new GenerationEvent(tokenId, Objects.requireNonNull(textDelta), logProbability, null);
  }

  /** Creates the single terminal event emitted by a completed generation. */
  public static GenerationEvent finished(FinishReason reason) {
    return new GenerationEvent(null, null, null, reason);
  }

  /** Creates a terminal event carrying a tokenizer flush delta. */
  public static GenerationEvent finished(FinishReason reason, String textDelta) {
    return new GenerationEvent(null, Objects.requireNonNull(textDelta), null, reason);
  }
}
