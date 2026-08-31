package se.alipsa.jmlx.tokenizer;

/**
 * The {@code ByteLevel} post-processor step: a no-op on the token list (this port does not track
 * offsets).
 */
public record ByteLevelStep() implements PostProcessorStep {}
