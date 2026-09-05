package se.alipsa.jmlx.tokenizer;

/**
 * The {@code ByteLevel} post-processor and its offset-trimming choice.
 *
 * @param trimOffsets whether synthetic prefix spaces are removed from offsets
 */
public record ByteLevelStep(boolean trimOffsets) implements PostProcessorStep {
  /** Compatibility constructor using the Hugging Face default. */
  public ByteLevelStep() {
    this(true);
  }
}
