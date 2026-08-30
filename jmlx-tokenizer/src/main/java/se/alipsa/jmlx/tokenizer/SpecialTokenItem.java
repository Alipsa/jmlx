package se.alipsa.jmlx.tokenizer;

/**
 * A {@code {"SpecialToken": {"id": "..."}}} template item, referencing a key into {@code
 * specialTokens}.
 */
public record SpecialTokenItem(String id) implements TemplateItem {}
