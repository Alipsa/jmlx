package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Immutable result of a completed generation, retaining prompt and generated token IDs separately.
 * For legacy compatibility, EOS is retained in generated IDs; an explicit stop token is excluded.
 * {@link #logProbabilities()} is empty when not requested, otherwise it aligns one-to-one with the
 * generated IDs. {@link #generatedText()} is non-null for tokenizer-backed requests and null for
 * pretokenized requests.
 *
 * <p>{@link #promptPositions()} is the effective input prompt length: the number of positions the
 * model actually consumed as prompt. For the decoder models, the batch scheduler, and T5 (whose
 * encoder consumes the source once) it equals {@code promptTokenIds().size()}. For vision models it
 * is the expanded prompt length, because each {@code <image>} placeholder stands for a
 * marker-delimited block of image tokens — while {@code promptTokenIds()} always retains the
 * original, unexpanded prompt IDs, so {@code promptPositions() >= promptTokenIds().size()}.
 *
 * <p>Adding a record component changes record equality, hashing, and string representation even
 * though the four- and five-argument compatibility constructors remain available.
 *
 * @param promptTokenIds the original, unexpanded prompt IDs, owned copy
 * @param generatedTokenIds the generated IDs, owned copy
 * @param finishReason why generation stopped
 * @param logProbabilities per generated-ID log probabilities, empty when not requested
 * @param generatedText the decoded generated text, or null for pretokenized requests
 * @param promptPositions the effective input prompt length, see the class documentation
 */
public record GenerationResult(
    List<Integer> promptTokenIds,
    List<Integer> generatedTokenIds,
    FinishReason finishReason,
    List<Double> logProbabilities,
    String generatedText,
    int promptPositions) {
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
    if (promptPositions < promptTokenIds.size()) {
      throw new IllegalArgumentException(
          "promptPositions "
              + promptPositions
              + " must be at least the prompt token count "
              + promptTokenIds.size());
    }
  }

  /**
   * Compatibility constructor for results whose effective prompt length is the prompt ID count (the
   * decoder models, the batch scheduler, T5); vision models use the six-argument form with their
   * expanded length.
   */
  public GenerationResult(
      List<Integer> promptTokenIds,
      List<Integer> generatedTokenIds,
      FinishReason finishReason,
      List<Double> logProbabilities,
      String generatedText) {
    // -1 keeps a null promptTokenIds failing with the same "promptTokenIds" NPE as the canonical
    // constructor; the sentinel is never otherwise observable.
    this(
        promptTokenIds,
        generatedTokenIds,
        finishReason,
        logProbabilities,
        generatedText,
        promptTokenIds == null ? -1 : promptTokenIds.size());
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
