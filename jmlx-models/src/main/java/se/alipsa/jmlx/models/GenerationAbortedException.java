package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;

/**
 * A token listener aborted generation; generated token IDs remain available. The sequence includes
 * the token whose listener callback threw, if one had already been selected.
 */
public final class GenerationAbortedException extends IllegalStateException {
  private final List<Integer> promptTokenIds;
  private final List<Integer> generatedTokenIds;

  GenerationAbortedException(
      List<Integer> promptTokenIds, List<Integer> generatedTokenIds, RuntimeException cause) {
    super(
        "generation listener aborted after generating " + generatedTokenIds.size() + " token(s)",
        cause);
    this.promptTokenIds = List.copyOf(Objects.requireNonNull(promptTokenIds, "promptTokenIds"));
    this.generatedTokenIds =
        List.copyOf(Objects.requireNonNull(generatedTokenIds, "generatedTokenIds"));
  }

  /** Prompt token IDs for the aborted generation. */
  public List<Integer> promptTokenIds() {
    return promptTokenIds;
  }

  /** Generated token IDs, including the selected token whose listener callback may have failed. */
  public List<Integer> generatedTokenIds() {
    return generatedTokenIds;
  }
}
