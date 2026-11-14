package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/** Shared mapping of framework generation overrides to the native request policy. */
public final class InferenceOptions {
  private InferenceOptions() {}

  /** Rejects model switching and stop strings and applies the supported nullable overrides. */
  public static GenerationConfig apply(
      InferenceClient client,
      String model,
      Integer maxTokens,
      Double temperature,
      Integer topK,
      Double topP,
      Double frequencyPenalty,
      Double presencePenalty,
      List<String> stops) {
    Objects.requireNonNull(client, "client");
    if (model != null && !model.equals(client.modelName())) {
      throw new IllegalArgumentException("this client is loaded with model " + client.modelName());
    }
    if (stops != null && !stops.isEmpty()) {
      throw new IllegalArgumentException(
          "stop strings are not supported; configure stop token IDs");
    }
    GenerationConfig base = client.defaults();
    float temp = temperature == null ? base.temperature() : temperature.floatValue();
    if (temperature != null && temperature != 0 && temp == 0) {
      throw new IllegalArgumentException(
          "temperature is too small for float32; use explicit zero for greedy");
    }
    return new GenerationConfig(
        maxTokens == null ? base.maxNewTokens() : maxTokens,
        temp == 0 ? OptionalLong.empty() : base.seed(),
        temp,
        topK == null ? (temp == 0 ? 0 : base.topK()) : topK,
        topP == null ? (temp == 0 ? 1 : base.topP()) : topP.floatValue(),
        temp == 0 ? 0 : base.minP(),
        base.repetitionPenalty(),
        frequencyPenalty == null ? base.frequencyPenalty() : frequencyPenalty.floatValue(),
        presencePenalty == null ? base.presencePenalty() : presencePenalty.floatValue(),
        base.eosTokenIds(),
        base.stopTokenIds(),
        base.logProbabilities());
  }
}
