package se.alipsa.jmlx.tokenizer;

/**
 * A half-open UTF-8 byte range in the original input string.
 *
 * @param startByte inclusive byte boundary
 * @param endByte exclusive byte boundary
 */
public record TokenOffset(int startByte, int endByte) {
  /** Offset used by special and padding tokens, which do not cover input text. */
  public static final TokenOffset NONE = new TokenOffset(0, 0);

  /** Validates an ordered, non-negative byte range. */
  public TokenOffset {
    if (startByte < 0 || endByte < startByte) {
      throw new IllegalArgumentException("TokenOffset must be an ordered, non-negative range");
    }
  }
}
