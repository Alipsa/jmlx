package se.alipsa.jmlx.tokenizer;

/**
 * One entry from {@code tokenizer.json}'s {@code added_tokens} array.
 *
 * @param id token id
 * @param content literal token text
 * @param singleWord whether matches must occupy a complete word
 * @param leftStrip whether adjacent whitespace on the left is consumed
 * @param rightStrip whether adjacent whitespace on the right is consumed
 * @param normalized whether matching happens against normalized text
 * @param special whether the token is special
 */
public record AddedToken(
    int id,
    String content,
    boolean singleWord,
    boolean leftStrip,
    boolean rightStrip,
    boolean normalized,
    boolean special) {

  /**
   * Compatibility constructor for an unstripped, non-normalized token.
   *
   * @param id token ID
   * @param content literal token text
   * @param special whether the token is special
   */
  public AddedToken(int id, String content, boolean special) {
    this(id, content, false, false, false, false, special);
  }
}
