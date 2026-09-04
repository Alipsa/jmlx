package se.alipsa.jmlx.models;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;

/** Immutable generation policy. Sampling fields are validated here and implemented in Phase 6.1. */
public record GenerationConfig(
    int maxNewTokens,
    OptionalLong seed,
    float temperature,
    int topK,
    float topP,
    float minP,
    float repetitionPenalty,
    float frequencyPenalty,
    float presencePenalty,
    Set<Integer> eosTokenIds,
    Set<Integer> stopTokenIds,
    boolean logProbabilities) {

  /** Validates a complete, immutable generation policy. */
  public GenerationConfig {
    if (maxNewTokens < 0) {
      throw new IllegalArgumentException("maxNewTokens must be non-negative");
    }
    Objects.requireNonNull(seed, "seed");
    if (!Float.isFinite(temperature) || temperature < 0) {
      throw new IllegalArgumentException("temperature must be non-negative");
    }
    if (topK < 0) {
      throw new IllegalArgumentException("topK must be non-negative");
    }
    probability("topP", topP);
    probability("minP", minP);
    positive("repetitionPenalty", repetitionPenalty);
    finite("frequencyPenalty", frequencyPenalty);
    finite("presencePenalty", presencePenalty);
    eosTokenIds = immutableIds("eosTokenIds", eosTokenIds);
    stopTokenIds = immutableIds("stopTokenIds", stopTokenIds);
  }

  /** The legacy decoder's deterministic greedy policy. */
  public static GenerationConfig greedyDefaults(int maxNewTokens, Set<Integer> eosTokenIds) {
    return new GenerationConfig(
        maxNewTokens, OptionalLong.empty(), 0, 0, 1, 0, 1, 0, 0, eosTokenIds, Set.of(), false);
  }

  private static Set<Integer> immutableIds(String name, Set<Integer> ids) {
    Objects.requireNonNull(ids, name);
    if (ids.stream().anyMatch(Objects::isNull)) {
      throw new IllegalArgumentException(name + " must not contain null");
    }
    return Set.copyOf(new LinkedHashSet<>(ids));
  }

  private static void probability(String name, float value) {
    if (!Float.isFinite(value) || value < 0 || value > 1) {
      throw new IllegalArgumentException(name + " must be between zero and one");
    }
  }

  private static void positive(String name, float value) {
    if (!Float.isFinite(value) || value <= 0) {
      throw new IllegalArgumentException(name + " must be positive");
    }
  }

  private static void finite(String name, float value) {
    if (!Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite");
    }
  }
}
