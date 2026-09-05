package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Immutable result of a completed generation, retaining prompt and generated token IDs separately.
 * For legacy compatibility, EOS is retained in generated IDs; an explicit stop token is excluded.
 * {@link #logProbabilities()} is empty when not requested, otherwise it aligns one-to-one with the
 * generated IDs. {@link #generatedText()} is non-null for tokenizer-backed requests and null for
 * pretokenized requests. Adding that record component changes record equality, hashing, and string
 * representation even though the four-argument compatibility constructor remains available.
 */
public record GenerationResult(
    List<Integer> promptTokenIds,
    List<Integer> generatedTokenIds,
    FinishReason finishReason,
    List<Double> logProbabilities,
    String generatedText) {
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

  /** Compatibility constructor for a pretokenized result with no decoded text. */
  public GenerationResult(
      List<Integer> promptTokenIds,
      List<Integer> generatedTokenIds,
      FinishReason finishReason,
      List<Double> logProbabilities) {
    this(promptTokenIds, generatedTokenIds, finishReason, logProbabilities, null);
  }

  /** Returns prompt followed by generated IDs. */
  public List<Integer> tokenIds() {
    return Stream.concat(promptTokenIds.stream(), generatedTokenIds.stream()).toList();
  }
}
