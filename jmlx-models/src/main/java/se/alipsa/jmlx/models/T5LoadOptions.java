package se.alipsa.jmlx.models;

/** Explicit source resource limit; T5 has no absolute position-embedding capacity. */
public record T5LoadOptions(int maxSourceTokens) {
  /** Requires a positive limit. */
  public T5LoadOptions {
    if (maxSourceTokens <= 0) {
      throw new IllegalArgumentException("maxSourceTokens must be positive");
    }
  }

  /** Default source resource limit. */
  public static T5LoadOptions defaults() {
    return new T5LoadOptions(512);
  }
}
