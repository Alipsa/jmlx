package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.OptionalLong;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Native tests for the architecture-independent ordered sampling graph. */
@EnabledIfNativeAvailable
class SamplingPipelineTest {

  @Test
  void greedyAppliesPenaltiesAndReportsTheDefinedLogProbability() {
    GenerationConfig policy =
        new GenerationConfig(
            1, OptionalLong.empty(), 0, 0, 1, 0, 2, 1, 0.5f, Set.of(), Set.of(), true);
    LinkedHashMap<Integer, Integer> history = new LinkedHashMap<>();
    history.put(0, 2);
    try (MLXScope generation = new MLXScope();
        SamplingPipeline pipeline = new SamplingPipeline(generation, policy, 3);
        MLXScope activation = generation.newChild()) {
      MLXArray logits = MLX.array(activation, new float[] {6, 2, 1}, new int[] {1, 1, 3});

      SamplingPipeline.Selection selected =
          pipeline.select(logits, PenaltyInputs.from(history, 3), 0);

      // Token 0 becomes 6 / 2 - 2 * 1 - .5 = .5, so token 1 wins.
      assertEquals(1, selected.tokenId());
      assertEquals(0.0, selected.logProbability());
    }
  }

  @Test
  void sampledSelectionAndLogProbabilityRepeatForAnExplicitSeed() {
    GenerationConfig policy = sampledPolicy(true);
    SamplingPipeline.Selection first = select(policy);
    SamplingPipeline.Selection second = select(policy);

    assertEquals(first.tokenId(), second.tokenId());
    assertEquals(first.logProbability(), second.logProbability());
    assertTrue(first.tokenId() >= 0 && first.tokenId() < 4);
    assertTrue(Double.isFinite(first.logProbability()));
    assertTrue(first.logProbability() <= 0);
  }

  @Test
  void rejectsNonFiniteLogitsAsPolicyError() {
    GenerationConfig policy = sampledPolicy(false);
    try (MLXScope generation = new MLXScope();
        SamplingPipeline pipeline = new SamplingPipeline(generation, policy, 4);
        MLXScope activation = generation.newChild()) {
      MLXArray logits =
          MLX.array(activation, new float[] {1, Float.NaN, 2, 0}, new int[] {1, 1, 4});

      RuntimeException failure =
          assertThrows(
              RuntimeException.class,
              () -> pipeline.select(logits, new PenaltyInputs(new int[0], new float[0]), 7));
      IllegalStateException policyFailure = assertInstanceOf(IllegalStateException.class, failure);

      assertTrue(
          policyFailure
              .getMessage()
              .contains(
                  "sampling stage finite-logit validation failed at decode step 7: logits must be"
                      + " finite"));
    }
  }

  @Test
  void attributesTemperatureOverflowToScaling() {
    GenerationConfig policy =
        new GenerationConfig(
            1, OptionalLong.of(42), Float.MIN_VALUE, 0, 1, 0, 1, 0, 0, Set.of(), Set.of(), false);
    try (MLXScope generation = new MLXScope();
        SamplingPipeline pipeline = new SamplingPipeline(generation, policy, 2);
        MLXScope activation = generation.newChild()) {
      MLXArray logits = MLX.array(activation, new float[] {1, 0}, new int[] {1, 1, 2});

      IllegalStateException failure =
          assertThrows(
              IllegalStateException.class,
              () -> pipeline.select(logits, new PenaltyInputs(new int[0], new float[0]), 3));

      assertEquals(
          "sampling stage temperature scaling failed at decode step 3: tempered logits must be"
              + " finite",
          failure.getMessage());
    }
  }

  private static SamplingPipeline.Selection select(GenerationConfig policy) {
    try (MLXScope generation = new MLXScope();
        SamplingPipeline pipeline = new SamplingPipeline(generation, policy, 4);
        MLXScope activation = generation.newChild()) {
      MLXArray logits = MLX.array(activation, new float[] {1, 3, 2, 0}, new int[] {1, 1, 4});
      return pipeline.select(logits, new PenaltyInputs(new int[0], new float[0]), 0);
    }
  }

  private static GenerationConfig sampledPolicy(boolean logProbabilities) {
    return new GenerationConfig(
        1, OptionalLong.of(42), 1, 3, 0.9f, 0.05f, 1, 0, 0, Set.of(), Set.of(), logProbabilities);
  }
}
