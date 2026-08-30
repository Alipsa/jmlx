package se.alipsa.jmlx.tokenizer;

/** One entry from {@code tokenizer.json}'s {@code added_tokens} array. */
public record AddedToken(int id, String content, boolean special) {}
