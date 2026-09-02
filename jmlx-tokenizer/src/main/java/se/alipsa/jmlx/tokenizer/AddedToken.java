package se.alipsa.jmlx.tokenizer;

/**
 * One entry from {@code tokenizer.json}'s {@code added_tokens} array.
 *
 * @param id token id
 * @param content literal token text
 * @param special whether the token is special
 */
public record AddedToken(int id, String content, boolean special) {}
