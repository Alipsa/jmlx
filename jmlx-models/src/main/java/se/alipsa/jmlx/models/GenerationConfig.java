package se.alipsa.jmlx.models;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;

/** Immutable greedy or explicitly seeded sampling policy. */
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
      throw new IllegalArgumentException("temperature must be finite and non-negative");
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
    if (temperature == 0) {
      if (seed.isPresent() || topK != 0 || topP != 1 || minP != 0) {
        throw new IllegalArgumentException(
            "greedy mode requires no seed, topK=0, topP=1, and minP=0");
      }
    } else if (seed.isEmpty()) {
      throw new IllegalArgumentException("sampled mode requires an explicit seed");
    }
  }

  /** The legacy decoder's deterministic greedy policy. */
  public static GenerationConfig greedyDefaults(int maxNewTokens, Set<Integer> eosTokenIds) {
    return greedyDefaults(maxNewTokens, eosTokenIds, Set.of());
  }

  /** Deterministic greedy policy with explicit non-EOS stop token IDs. */
  public static GenerationConfig greedyDefaults(
      int maxNewTokens, Set<Integer> eosTokenIds, Set<Integer> stopTokenIds) {
    return new GenerationConfig(
        maxNewTokens, OptionalLong.empty(), 0, 0, 1, 0, 1, 0, 0, eosTokenIds, stopTokenIds, false);
  }

  /** Seeded sampling with filtering disabled and penalties at their neutral values. */
  public static GenerationConfig samplingDefaults(
      int maxNewTokens, long seed, float temperature, Set<Integer> eosTokenIds) {
    return samplingDefaults(maxNewTokens, seed, temperature, eosTokenIds, Set.of());
  }

  /** Seeded sampling defaults with explicit non-EOS stop token IDs. */
  public static GenerationConfig samplingDefaults(
      int maxNewTokens,
      long seed,
      float temperature,
      Set<Integer> eosTokenIds,
      Set<Integer> stopTokenIds) {
    if (!(temperature > 0)) {
      throw new IllegalArgumentException("samplingDefaults temperature must be positive");
    }
    return new GenerationConfig(
        maxNewTokens,
        OptionalLong.of(seed),
        temperature,
        0,
        1,
        0,
        1,
        0,
        0,
        eosTokenIds,
        stopTokenIds,
        false);
  }

  private static Set<Integer> immutableIds(String name, Set<Integer> ids) {
    Objects.requireNonNull(ids, name);
    if (ids.stream().anyMatch(Objects::isNull)) {
      throw new IllegalArgumentException(name + " must not contain null");
    }
    return Set.copyOf(ids);
  }

  private static void probability(String name, float value) {
    if (!Float.isFinite(value) || value < 0 || value > 1) {
      throw new IllegalArgumentException(name + " must be finite and between zero and one");
    }
  }

  private static void positive(String name, float value) {
    if (!Float.isFinite(value) || value <= 0) {
      throw new IllegalArgumentException(name + " must be finite and positive");
    }
  }

  private static void finite(String name, float value) {
    if (!Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite");
    }
  }
}
