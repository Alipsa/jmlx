package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class PostProcessorApplierTest {

  @Test
  void byteLevelStepAloneIsANoOp() {
    List<String> tokens = List.of("low", "Ġthe");
    assertEquals(tokens, PostProcessorApplier.apply(List.of(new ByteLevelStep()), tokens, true));
  }

  @Test
  void templateProcessingPrependsBosTokenWhenAddSpecialTokensIsTrue() {
    var template =
        new TemplateProcessingStep(
            List.of(new SpecialTokenItem("<|begin_of_text|>"), new SequenceItem()),
            Map.of(
                "<|begin_of_text|>",
                new SpecialTokenInfo(
                    "<|begin_of_text|>", List.of(128000), List.of("<|begin_of_text|>"))));
    List<String> tokens = List.of("low", "Ġthe");
    assertEquals(
        List.of("<|begin_of_text|>", "low", "Ġthe"),
        PostProcessorApplier.apply(List.of(new ByteLevelStep(), template), tokens, true));
  }

  @Test
  void templateProcessingOmitsBosTokenWhenAddSpecialTokensIsFalse() {
    var template =
        new TemplateProcessingStep(
            List.of(new SpecialTokenItem("<|begin_of_text|>"), new SequenceItem()),
            Map.of(
                "<|begin_of_text|>",
                new SpecialTokenInfo(
                    "<|begin_of_text|>", List.of(128000), List.of("<|begin_of_text|>"))));
    List<String> tokens = List.of("low", "Ġthe");
    assertEquals(tokens, PostProcessorApplier.apply(List.of(template), tokens, false));
  }
}
