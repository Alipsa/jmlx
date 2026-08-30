package se.alipsa.jmlx.tokenizer;

/**
 * One item of a {@code TemplateProcessing} template: either a literal special token or the real
 * sequence.
 */
public sealed interface TemplateItem permits SpecialTokenItem, SequenceItem {}
