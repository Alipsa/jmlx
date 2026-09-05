package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

class Phase62GenerationContractTest {

  @Test
  void textFactoryPinsSpecialTokenPolicyAndPromptIds() {
    HfTokenizer tokenizer = qwenTokenizer();
    GenerationConfig config = GenerationConfig.greedyDefaults(2, Set.of());

    GenerationRequest request =
        GenerationRequest.text(
            tokenizer, "hello", PromptSpecialTokens.OMIT, config, CancellationToken.NONE);

    assertEquals(PromptSpecialTokens.OMIT, request.promptSpecialTokens());
    assertArrayEquals(
        tokenizer.encode("hello", false).stream().mapToInt(Integer::intValue).toArray(),
        request.promptTokenIds());
    assertThrows(
        IllegalArgumentException.class,
        () ->
            GenerationRequest.text(
                tokenizer,
                "hello",
                PromptSpecialTokens.PRETOKENIZED,
                config,
                CancellationToken.NONE));
  }

  @Test
  void pretokenizedCompatibilityKeepsNullText() {
    GenerationRequest request =
        new GenerationRequest(
            new int[] {1}, GenerationConfig.greedyDefaults(1, Set.of()), CancellationToken.NONE);
    GenerationResult result =
        new GenerationResult(List.of(1), List.of(2), FinishReason.MAX_TOKENS, List.of());

    assertEquals(PromptSpecialTokens.PRETOKENIZED, request.promptSpecialTokens());
    assertNull(result.generatedText());
    assertNull(GenerationEvent.finished(FinishReason.CANCELLED).textDelta());
    assertEquals("", GenerationEvent.finished(FinishReason.CANCELLED, "").textDelta());
  }

  private static HfTokenizer qwenTokenizer() {
    String root =
        java.util.Objects.requireNonNull(
            System.getProperty("jmlx.repository.root"),
            "jmlx.repository.root must be set by build.gradle");
    return HfTokenizer.fromFile(
        Path.of(
            root,
            "jmlx-tokenizer",
            "src",
            "test",
            "resources",
            "se",
            "alipsa",
            "jmlx",
            "tokenizer",
            "qwen-style.tokenizer.json"));
  }
}
