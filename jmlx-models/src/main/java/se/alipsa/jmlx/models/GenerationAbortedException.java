package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;

/** Generation aborted after it started because output decoding or a token listener failed. */
public final class GenerationAbortedException extends IllegalStateException {
  private final List<Integer> promptTokenIds;
  private final List<Integer> generatedTokenIds;
  private final String stage;
  private final Integer failingTokenId;

  GenerationAbortedException(
      List<Integer> promptTokenIds, List<Integer> generatedTokenIds, RuntimeException cause) {
    this(promptTokenIds, generatedTokenIds, "listener", null, cause);
  }

  GenerationAbortedException(
      List<Integer> promptTokenIds,
      List<Integer> generatedTokenIds,
      String stage,
      Integer failingTokenId,
      RuntimeException cause) {
    super(
        "generation "
            + stage
            + " aborted after prompt length "
            + promptTokenIds.size()
            + " and "
            + generatedTokenIds.size()
            + " generated token(s)"
            + (failingTokenId == null ? "" : "; failing token " + failingTokenId),
        cause);
    this.promptTokenIds = List.copyOf(Objects.requireNonNull(promptTokenIds, "promptTokenIds"));
    this.generatedTokenIds =
        List.copyOf(Objects.requireNonNull(generatedTokenIds, "generatedTokenIds"));
    this.stage = Objects.requireNonNull(stage, "stage");
    this.failingTokenId = failingTokenId;
  }

  /** Prompt token IDs for the aborted generation. */
  public List<Integer> promptTokenIds() {
    return promptTokenIds;
  }

  /** Generated token IDs, including the selected token whose listener callback may have failed. */
  public List<Integer> generatedTokenIds() {
    return generatedTokenIds;
  }

  /** Stage that failed after generation started. */
  public String stage() {
    return stage;
  }

  /** Token whose decoding failed, or null for listener failures. */
  public Integer failingTokenId() {
    return failingTokenId;
  }
}
