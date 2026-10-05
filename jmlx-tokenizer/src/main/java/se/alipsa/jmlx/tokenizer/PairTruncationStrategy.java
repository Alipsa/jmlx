package se.alipsa.jmlx.tokenizer;

/** Selects which input loses tokens when encoding a pair with a maximum length. */
public enum PairTruncationStrategy {
  /** Balance lengths, preserving the shorter sequence until both have equal length. */
  LONGEST_FIRST,
  /** Remove tokens only from the first input. */
  ONLY_FIRST,
  /** Remove tokens only from the second input. */
  ONLY_SECOND
}
