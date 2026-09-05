package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Immutable result of a completed generation, retaining prompt and generated token IDs separately.
 * For legacy compatibility, EOS is retained in generated IDs; an explicit stop token is excluded.
 * {@link #logProbabilities()} is empty when not requested, otherwise it aligns one-to-one with the
 * generated IDs.
 */
public record GenerationResult(
    List<Integer> promptTokenIds,
    List<Integer> generatedTokenIds,
    FinishReason finishReason,
    List<Double> logProbabilities) {
  /** Creates an immutable completed-generation result. */
  public GenerationResult {
    promptTokenIds = List.copyOf(Objects.requireNonNull(promptTokenIds, "promptTokenIds"));
    generatedTokenIds = List.copyOf(Objects.requireNonNull(generatedTokenIds, "generatedTokenIds"));
    finishReason = Objects.requireNonNull(finishReason, "finishReason");
    logProbabilities = List.copyOf(Objects.requireNonNull(logProbabilities, "logProbabilities"));
    if (!logProbabilities.isEmpty() && logProbabilities.size() != generatedTokenIds.size()) {
      throw new IllegalArgumentException(
          "logProbabilities must be empty or match generatedTokenIds cardinality");
    }
  }

  /** Returns prompt followed by generated IDs. */
  public List<Integer> tokenIds() {
    return Stream.concat(promptTokenIds.stream(), generatedTokenIds.stream()).toList();
  }
}
