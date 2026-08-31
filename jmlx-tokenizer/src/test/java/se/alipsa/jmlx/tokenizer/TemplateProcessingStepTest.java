package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Unit coverage for {@link TemplateProcessingStep}'s own compact-constructor invariants -- the same
 * validate-in-the-type-itself precedent {@link SpecialTokenInfo} established in round 3 (PR #14
 * review round 9).
 */
class TemplateProcessingStepTest {

  @Test
  void constructorThrowsWhenSingleIsEmpty() {
    // An empty single can never emit the real encoded sequence under any addSpecialTokens
    // setting -- PostProcessorApplier#applyTemplate rebuilds its whole output purely from
    // single's items (PR #14 review round 9, finding 2).
    assertThrows(TokenizerException.class, () -> new TemplateProcessingStep(List.of(), Map.of()));
  }

  @Test
  void constructorThrowsWhenSingleHasOnlySpecialTokenItemsAndNoSequenceItem() {
    // Non-empty, but still can never emit the real sequence, for the same reason as the empty
    // case above.
    assertThrows(
        TokenizerException.class,
        () ->
            new TemplateProcessingStep(
                List.of(new SpecialTokenItem("<s>")),
                Map.of("<s>", new SpecialTokenInfo("<s>", List.of(1), List.of("<s>")))));
  }

  @Test
  void doesNotThrowWhenSingleHasASequenceItemAlongsideSpecialTokenItems() {
    assertDoesNotThrow(
        () ->
            new TemplateProcessingStep(
                List.of(new SpecialTokenItem("<s>"), new SequenceItem()),
                Map.of("<s>", new SpecialTokenInfo("<s>", List.of(1), List.of("<s>")))));
  }

  @Test
  void constructorPreservesSpecialTokensInsertionOrder() {
    // Map.copyOf (this record's original defensive-copy choice, round 8 finding 2) neither
    // preserves insertion order nor picks a stable order of its own -- it randomizes its
    // iteration order per JVM run, verified directly across repeated runs.
    // Collections.unmodifiableMap(new LinkedHashMap<>(...)) keeps both the defensive copy and the
    // loader's own tokenizer.json declaration order (PR #14 review round 9, finding 5).
    Map<String, SpecialTokenInfo> ordered = new LinkedHashMap<>();
    ordered.put("<a>", new SpecialTokenInfo("<a>", List.of(1), List.of("<a>")));
    ordered.put("<b>", new SpecialTokenInfo("<b>", List.of(2), List.of("<b>")));
    ordered.put("<c>", new SpecialTokenInfo("<c>", List.of(3), List.of("<c>")));
    var template = new TemplateProcessingStep(List.of(new SequenceItem()), ordered);
    assertEquals(List.of("<a>", "<b>", "<c>"), List.copyOf(template.specialTokens().keySet()));
  }
}
