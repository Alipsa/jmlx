package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;

/**
 * Immutable result of a completed generation, retaining prompt and generated token IDs separately.
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
  }

  /** Returns prompt followed by generated IDs. */
  public List<Integer> tokenIds() {
    java.util.ArrayList<Integer> all = new java.util.ArrayList<>(promptTokenIds);
    all.addAll(generatedTokenIds);
    return List.copyOf(all);
  }
}
