package se.alipsa.jmlx.tokenizer;

/** One step of a (possibly {@code Sequence}-wrapped) {@code post_processor}. */
public sealed interface PostProcessorStep permits ByteLevelStep, TemplateProcessingStep {}
