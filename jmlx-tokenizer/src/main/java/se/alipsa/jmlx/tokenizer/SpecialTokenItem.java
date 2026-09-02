package se.alipsa.jmlx.tokenizer;

/**
 * A {@code {"SpecialToken": {"id": "..."}}} template item, referencing a key into {@code
 * specialTokens}.
 *
 * @param id key in the enclosing step's special-token map
 */
public record SpecialTokenItem(String id) implements TemplateItem {}
