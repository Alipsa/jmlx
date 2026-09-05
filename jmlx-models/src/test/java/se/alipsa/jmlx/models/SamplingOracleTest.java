package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
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
    String repositoryRoot =
        Objects.requireNonNull(
            System.getProperty("jmlx.repository.root"),
            "jmlx.repository.root must be set by jmlx-models/build.gradle");
    Path fixtures = Path.of(repositoryRoot, "tools/mlx-oracle/fixtures");
    JsonNode input = MAPPER.readTree(fixtures.resolve("phase6-1-sampling.input.json").toFile());
    JsonNode expected =
        MAPPER.readTree(fixtures.resolve("phase6-1-sampling.expected.json").toFile());
    JsonNode inputCases = required(input, "cases");
    JsonNode expectedCases = required(expected, "cases");
    assertTrue(inputCases.isArray() && !inputCases.isEmpty(), "oracle cases must be non-empty");
    assertTrue(expectedCases.isArray(), "expected oracle cases must be an array");
    assertEquals(inputCases.size(), expectedCases.size(), "oracle case counts");

    for (int caseIndex = 0; caseIndex < inputCases.size(); caseIndex++) {
      JsonNode inputCase = inputCases.get(caseIndex);
      JsonNode expectedCase = expectedCases.get(caseIndex);
      String name = required(inputCase, "name").textValue();
      assertEquals(name, required(expectedCase, "name").textValue(), "oracle case order");
      float[] logits = floats(required(inputCase, "logits"));
      GenerationConfig policy = policy(inputCase);
      LinkedHashMap<Integer, Integer> history = history(required(inputCase, "history"));

      try (MLXScope generation = new MLXScope();
          SamplingPipeline pipeline = new SamplingPipeline(generation, policy, logits.length);
          MLXScope activation = generation.newChild()) {
        MLXArray array = MLX.array(activation, logits, new int[] {1, 1, logits.length});
        SamplingPipeline.Selection actual =
            pipeline.select(array, PenaltyInputs.from(history, logits.length), 0);

        assertEquals(required(expectedCase, "selectedToken").intValue(), actual.tokenId(), name);
        assertEquals(
            required(expectedCase, "selectedLogProbability").doubleValue(),
            actual.logProbability(),
            1e-6,
            name);
        if (policy.temperature() != 0) {
          JsonNode expectedVocabularyLogits = required(expectedCase, "vocabularyLogits");
          assertArrayEquals(
              floatsWithInfinity(expectedVocabularyLogits),
              actual.vocabularyLogits().toFloatArray(),
              1e-6f,
              name);
        }
      }
    }
  }

  private static GenerationConfig policy(JsonNode inputCase) {
    JsonNode policy = required(inputCase, "policy");
    float temperature = (float) required(policy, "temperature").doubleValue();
    OptionalLong seed =
        temperature == 0
            ? OptionalLong.empty()
            : OptionalLong.of(required(inputCase, "seed").longValue());
    return new GenerationConfig(
        1,
        seed,
        temperature,
        required(policy, "topK").intValue(),
        (float) required(policy, "topP").doubleValue(),
        (float) required(policy, "minP").doubleValue(),
        (float) required(policy, "repetitionPenalty").doubleValue(),
        (float) required(policy, "frequencyPenalty").doubleValue(),
        (float) required(policy, "presencePenalty").doubleValue(),
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

  private static float[] floatsWithInfinity(JsonNode node) {
    float[] result = new float[node.size()];
    for (int i = 0; i < result.length; i++) {
      JsonNode value = node.get(i);
      if (!value.isTextual()) {
        result[i] = (float) value.doubleValue();
      } else if ("Infinity".equals(value.textValue())) {
        result[i] = Float.POSITIVE_INFINITY;
      } else if ("-Infinity".equals(value.textValue())) {
        result[i] = Float.NEGATIVE_INFINITY;
      } else {
        throw new IllegalArgumentException("unexpected oracle float value: " + value.textValue());
      }
    }
    return result;
  }

  private static JsonNode required(JsonNode parent, String field) {
    return Objects.requireNonNull(parent.get(field), "missing oracle field: " + field);
  }
}
