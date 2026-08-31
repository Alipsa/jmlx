package se.alipsa.jmlx.tokenizer;

/**
 * One {@link PostProcessorApplier#apply} output entry. {@code id}, when non-null, is an
 * already-known vocabulary id for {@code text} -- taken directly from a {@code TemplateProcessing}
 * special token's own {@code special_tokens.*.ids} entry -- and must be used as-is instead of
 * re-resolving {@code text} through {@link Vocabulary#idOf}: a template can reference a special
 * token whose id is recorded there while its literal string is absent from {@code model.vocab} +
 * {@code added_tokens}, in which case a fresh string lookup would throw even though the correct id
 * was available all along.
 */
public record ResolvedToken(String text, Integer id) {}
