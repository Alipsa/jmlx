package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;

/**
 * A token listener aborted generation; the generated token IDs delivered before the failure remain
 * available.
 */
public final class GenerationAbortedException extends IllegalStateException {
  private final List<Integer> promptTokenIds;
  private final List<Integer> generatedTokenIds;

  GenerationAbortedException(
      List<Integer> promptTokenIds, List<Integer> generatedTokenIds, RuntimeException cause) {
    super(
        "generation listener aborted after " + generatedTokenIds.size() + " generated token(s)",
        cause);
    this.promptTokenIds = List.copyOf(Objects.requireNonNull(promptTokenIds, "promptTokenIds"));
    this.generatedTokenIds =
        List.copyOf(Objects.requireNonNull(generatedTokenIds, "generatedTokenIds"));
  }

  /** Prompt token IDs for the aborted generation. */
  public List<Integer> promptTokenIds() {
    return promptTokenIds;
  }

  /** Generated token IDs delivered before the listener failed. */
  public List<Integer> generatedTokenIds() {
    return generatedTokenIds;
  }
}
