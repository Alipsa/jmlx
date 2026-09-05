package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.OptionalLong;
import java.util.Set;
import org.junit.jupiter.api.Test;

class GenerationContractTest {
  @Test
  void requestOwnsPromptAndConfigOwnsTokenSets() {
    Set<Integer> eos = new java.util.LinkedHashSet<>(List.of(1, 2));
    GenerationConfig config = GenerationConfig.greedyDefaults(3, eos);
    int[] prompt = {4, 5};
    final GenerationRequest request = new GenerationRequest(prompt, config, CancellationToken.NONE);

    eos.clear();
    prompt[0] = 99;

    assertEquals(Set.of(1, 2), config.eosTokenIds());
    assertArrayEquals(new int[] {4, 5}, request.promptTokenIds());
  }

  @Test
  void rejectsInvalidPolicyBeforeNativeWork() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GenerationConfig(
                1, OptionalLong.empty(), -1, 0, 1, 0, 1, 0, 0, Set.of(), Set.of(), false));
    final IllegalArgumentException temperature =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new GenerationConfig(
                    1,
                    OptionalLong.empty(),
                    Float.POSITIVE_INFINITY,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    Set.of(),
                    Set.of(),
                    false));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GenerationConfig(
                1, OptionalLong.empty(), 1, 0, 1, 0, 1, 0, 0, Set.of(), Set.of(), false));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GenerationConfig(
                1, OptionalLong.of(1), 0, 0, 1, 0, 1, 0, 0, Set.of(), Set.of(), false));
    assertThrows(
        IllegalArgumentException.class, () -> GenerationConfig.samplingDefaults(1, 1, 0, Set.of()));
    assertTrue(temperature.getMessage().contains("finite"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GenerationRequest(
                new int[0], GenerationConfig.greedyDefaults(0, Set.of()), CancellationToken.NONE));
  }

  @Test
  void resultKeepsPromptAndGeneratedTokensSeparate() {
    GenerationResult result =
        new GenerationResult(List.of(1, 2), List.of(3), FinishReason.MAX_TOKENS, List.of());

    assertEquals(List.of(1, 2), result.promptTokenIds());
    assertEquals(List.of(3), result.generatedTokenIds());
    assertEquals(List.of(1, 2, 3), result.tokenIds());
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GenerationResult(
                List.of(1), List.of(2, 3), FinishReason.MAX_TOKENS, List.of(-0.5)));
  }

  @Test
  void greedyDefaultsAcceptsExplicitStopTokensAndEventsHaveOneKind() {
    GenerationConfig policy = GenerationConfig.greedyDefaults(2, Set.of(1), Set.of(2));

    assertEquals(Set.of(1), policy.eosTokenIds());
    assertEquals(Set.of(2), policy.stopTokenIds());
    assertThrows(IllegalArgumentException.class, () -> new GenerationEvent(null, null, null, null));
    assertThrows(
        IllegalArgumentException.class, () -> new GenerationEvent(1, null, null, FinishReason.EOS));
    assertThrows(
        IllegalArgumentException.class,
        () -> new GenerationEvent(null, null, -0.5, FinishReason.EOS));
  }

  @Test
  void penaltyInputsScaleWithDistinctHistoryRatherThanVocabulary() {
    java.util.LinkedHashMap<Integer, Integer> frequencies =
        PenaltyInputs.frequencies(new int[] {7, 7, 42, 7});
    PenaltyInputs inputs = PenaltyInputs.from(frequencies, 151_936);

    assertArrayEquals(new int[] {7, 42}, inputs.tokenIds());
    assertArrayEquals(new float[] {3, 1}, inputs.counts());
  }
}
