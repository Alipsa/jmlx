package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Compares Java selection with the committed pinned-Python sampling fixture. */
@EnabledIfNativeAvailable
class SamplingOracleTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void matchesCommittedPythonSelectionsAndLogProbabilities() throws Exception {
    Path fixtures =
        Path.of(System.getProperty("jmlx.repository.root"), "tools/mlx-oracle/fixtures");
    JsonNode input = MAPPER.readTree(fixtures.resolve("phase6-1-sampling.input.json").toFile());
    JsonNode expected =
        MAPPER.readTree(fixtures.resolve("phase6-1-sampling.expected.json").toFile());

    for (int caseIndex = 0; caseIndex < input.path("cases").size(); caseIndex++) {
      JsonNode inputCase = input.path("cases").get(caseIndex);
      JsonNode expectedCase = expected.path("cases").get(caseIndex);
      float[] logits = floats(inputCase.path("logits"));
      GenerationConfig policy = policy(inputCase);
      LinkedHashMap<Integer, Integer> history = history(inputCase.path("history"));

      try (MLXScope generation = new MLXScope();
          SamplingPipeline pipeline = new SamplingPipeline(generation, policy, logits.length);
          MLXScope activation = generation.newChild()) {
        MLXArray array = MLX.array(activation, logits, new int[] {1, 1, logits.length});
        SamplingPipeline.Selection actual =
            pipeline.select(array, PenaltyInputs.from(history, logits.length), 0);

        assertEquals(
            expectedCase.path("selectedToken").intValue(),
            actual.tokenId(),
            inputCase.path("name").textValue());
        assertEquals(
            expectedCase.path("selectedLogProbability").doubleValue(),
            actual.logProbability(),
            1e-6,
            inputCase.path("name").textValue());
      }
    }
  }

  private static GenerationConfig policy(JsonNode inputCase) {
    JsonNode policy = inputCase.path("policy");
    float temperature = (float) policy.path("temperature").doubleValue();
    OptionalLong seed =
        temperature == 0
            ? OptionalLong.empty()
            : OptionalLong.of(inputCase.path("seed").longValue());
    return new GenerationConfig(
        1,
        seed,
        temperature,
        policy.path("topK").intValue(),
        (float) policy.path("topP").doubleValue(),
        (float) policy.path("minP").doubleValue(),
        (float) policy.path("repetitionPenalty").doubleValue(),
        (float) policy.path("frequencyPenalty").doubleValue(),
        (float) policy.path("presencePenalty").doubleValue(),
        Set.of(),
        Set.of(),
        true);
  }

  private static LinkedHashMap<Integer, Integer> history(JsonNode node) {
    LinkedHashMap<Integer, Integer> result = new LinkedHashMap<>();
    Iterator<Map.Entry<String, JsonNode>> fields = node.properties().iterator();
    while (fields.hasNext()) {
      Map.Entry<String, JsonNode> field = fields.next();
      result.put(Integer.parseInt(field.getKey()), field.getValue().intValue());
    }
    return result;
  }

  private static float[] floats(JsonNode node) {
    float[] result = new float[node.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = (float) node.get(i).doubleValue();
    }
    return result;
  }
}
