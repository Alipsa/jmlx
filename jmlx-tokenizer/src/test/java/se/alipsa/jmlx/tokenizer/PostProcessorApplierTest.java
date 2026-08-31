package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class PostProcessorApplierTest {

  @Test
  void byteLevelStepAloneIsANoOp() {
    List<String> tokens = List.of("low", "Ġthe");
    assertEquals(
        List.of(new ResolvedToken("low", null), new ResolvedToken("Ġthe", null)),
        PostProcessorApplier.apply(List.of(new ByteLevelStep()), tokens, true));
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
        List.of(
            new ResolvedToken("<|begin_of_text|>", 128000),
            new ResolvedToken("low", null),
            new ResolvedToken("Ġthe", null)),
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
    assertEquals(
        List.of(new ResolvedToken("low", null), new ResolvedToken("Ġthe", null)),
        PostProcessorApplier.apply(List.of(template), tokens, false));
  }

  @Test
  void mutatingTheSingleListAfterConstructionDoesNotAffectTheStoredTemplate() {
    // TemplateProcessingStep's compact constructor defensively copies single/specialTokens (PR
    // #14 review round 8, finding 2); without it, TokenizerJson's own List.copyOf(postProcessor)
    // is shallow (it freezes the outer list, not the steps inside it), so a caller mutating a
    // step's own mutable single list after construction still corrupted apply()'s output for
    // every later encode() call -- verified directly: clearing this list after construction used
    // to make apply() return an empty sequence instead of the real one.
    List<TemplateItem> mutableSingle =
        new ArrayList<>(List.of(new SpecialTokenItem("<|begin_of_text|>"), new SequenceItem()));
    var template =
        new TemplateProcessingStep(
            mutableSingle,
            Map.of(
                "<|begin_of_text|>",
                new SpecialTokenInfo(
                    "<|begin_of_text|>", List.of(128000), List.of("<|begin_of_text|>"))));
    mutableSingle.clear();
    List<String> tokens = List.of("low");
    assertEquals(
        List.of(new ResolvedToken("<|begin_of_text|>", 128000), new ResolvedToken("low", null)),
        PostProcessorApplier.apply(List.of(template), tokens, true));
  }

  @Test
  void mutatingTheSpecialTokensMapAfterConstructionDoesNotAffectTheStoredTemplate() {
    // The mirror of the test above, for specialTokens instead of single: removing an entry after
    // construction used to turn a valid template into apply()'s "references unknown special
    // token" throw (PR #14 review round 8, finding 2).
    Map<String, SpecialTokenInfo> mutableSpecialTokens =
        new HashMap<>(
            Map.of(
                "<|begin_of_text|>",
                new SpecialTokenInfo(
                    "<|begin_of_text|>", List.of(128000), List.of("<|begin_of_text|>"))));
    var template =
        new TemplateProcessingStep(
            List.of(new SpecialTokenItem("<|begin_of_text|>"), new SequenceItem()),
            mutableSpecialTokens);
    mutableSpecialTokens.remove("<|begin_of_text|>");
    List<String> tokens = List.of("low");
    assertEquals(
        List.of(new ResolvedToken("<|begin_of_text|>", 128000), new ResolvedToken("low", null)),
        PostProcessorApplier.apply(List.of(template), tokens, true));
  }

  @Test
  void templateProcessingUsesTheSpecialTokenInfoIdDirectlyWithoutAVocabularyLookup() {
    // The template's special token is present in special_tokens.ids but deliberately NOT in the
    // model.vocab/added_tokens a Vocabulary would be built from -- PostProcessorApplier must not
    // require a fresh string lookup to resolve it (see PR #14 review, finding 6).
    var template =
        new TemplateProcessingStep(
            List.of(new SpecialTokenItem("<|reserved|>"), new SequenceItem()),
            Map.of(
                "<|reserved|>",
                new SpecialTokenInfo("<|reserved|>", List.of(999), List.of("<|reserved|>"))));
    List<String> tokens = List.of("low");
    assertEquals(
        List.of(new ResolvedToken("<|reserved|>", 999), new ResolvedToken("low", null)),
        PostProcessorApplier.apply(List.of(template), tokens, true));
  }
}
