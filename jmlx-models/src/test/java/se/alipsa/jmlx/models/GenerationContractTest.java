package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
  }
}
