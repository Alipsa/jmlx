package se.alipsa.jmlx.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class TemplateResourceLimitTest {
  @Test
  void sourceLengthHasAnExactBoundary() {
    var sourceOptions = TemplateOptions.builder().maxSourceLength(5).build();
    assertEquals("abcde", Template.parse("abcde", sourceOptions).render(Map.of()));
    assertResourceLimit(
        () -> Template.parse("abcde\n", sourceOptions),
        "Source length 6 exceeds the configured limit of 5",
        Optional.empty());
  }

  @Test
  void tokenCountHasAnExactBoundary() {
    // {{ a }} is exactly three tokens: OpenExpression, Identifier, CloseExpression.
    var tokenOptions = TemplateOptions.builder().maxTokenCount(3).build();
    assertEquals("", Template.parse("{{ a }}", tokenOptions).render(Map.of()));
    assertResourceLimit(
        () -> Template.parse("{{ a }}", TemplateOptions.builder().maxTokenCount(2).build()),
        "Token count exceeds the configured limit of 2",
        location(5, 1, 6));
  }

  @Test
  void astDepthHasAnExactBoundary() {
    // The two parenthesized productions in {{ ((1)) }} consume the parser's two nested levels.
    assertEquals(
        "1",
        Template.parse("{{ ((1)) }}", TemplateOptions.builder().maxAstDepth(2).build())
            .render(Map.of()));
    assertResourceLimit(
        () -> Template.parse("{{ ((1)) }}", TemplateOptions.builder().maxAstDepth(1).build()),
        "AST depth exceeds the configured limit of 1",
        location(5, 1, 6));
  }

  @Test
  void stepsHaveAnExactBoundaryAndExpressionLocation() {
    var template = Template.parse("{{ 1 }}{{ 2 }}");
    assertEquals("12", template.render(Map.of(), RenderOptions.builder().maxSteps(2).build()));
    assertResourceLimit(
        () -> template.render(Map.of(), RenderOptions.builder().maxSteps(1).build()),
        "Maximum render steps exceeded",
        location(10, 1, 11));
  }

  @Test
  void forMaterializationHasExactSimpleAndNestedBoundaries() {
    var simple = Template.parse("{% for x in [1,2] %}x{% endfor %}");
    assertEquals(
        "xx", simple.render(Map.of(), RenderOptions.builder().maxLoopIterations(2).build()));
    assertResourceLimit(
        () -> simple.render(Map.of(), RenderOptions.builder().maxLoopIterations(1).build()),
        "Maximum loop iterations exceeded",
        location(0, 1, 1));

    var nested =
        Template.parse("{% for x in [1,2] %}{% for y in [1,2,3] %}a{% endfor %}{% endfor %}");
    assertEquals(
        "aaaaaa", nested.render(Map.of(), RenderOptions.builder().maxLoopIterations(8).build()));
    assertResourceLimit(
        () -> nested.render(Map.of(), RenderOptions.builder().maxLoopIterations(7).build()),
        "Maximum loop iterations exceeded",
        location(20, 1, 21));
  }

  @Test
  void rangeMaterializationHasItsOwnExactBoundary() {
    var template = Template.parse("{{ range(0, 3) }}");
    assertEquals(
        "[0, 1, 2]",
        template.render(Map.of(), RenderOptions.builder().maxLoopIterations(3).build()));
    assertResourceLimit(
        () -> template.render(Map.of(), RenderOptions.builder().maxLoopIterations(2).build()),
        "Maximum loop iterations exceeded",
        location(3, 1, 4));
  }

  @Test
  void sequentialMacrosExitTheirDepthAndCallBlocksHaveAnExactBoundary() {
    var repeated = Template.parse("{% macro f() %}x{% endmacro %}{{ f() }}{{ f() }}{{ f() }}");
    assertEquals(
        "xxx", repeated.render(Map.of(), RenderOptions.builder().maxMacroDepth(1).build()));

    // wrap() and its caller() invocation are simultaneously active, so this requires depth two.
    var callBlock =
        Template.parse(
            "{% macro wrap() %}<{{ caller() }}>{% endmacro %}{% call wrap() %}x{% endcall %}");
    assertEquals(
        "<x>", callBlock.render(Map.of(), RenderOptions.builder().maxMacroDepth(2).build()));
    assertResourceLimit(
        () -> callBlock.render(Map.of(), RenderOptions.builder().maxMacroDepth(1).build()),
        "Maximum macro call depth exceeded",
        location(22, 1, 23));
  }

  @Test
  void outputLimitsChargeEachCumulativeRenderPathAndPreserveAppendableOnFailure() {
    assertEquals(
        "a",
        Template.parse("{{ 'a' }}")
            .render(Map.of(), RenderOptions.builder().maxOutputLength(1).build()));
    var simple = Template.parse("{{ 'a' }}{{ 'b' }}");
    assertEquals("ab", simple.render(Map.of(), RenderOptions.builder().maxOutputLength(2).build()));
    var oneCharacter = RenderOptions.builder().maxOutputLength(1).build();
    assertResourceLimit(
        () -> simple.render(Map.of(), oneCharacter),
        "Maximum output length exceeded",
        location(12, 1, 13));
    var output = new StringBuilder("unchanged");
    assertResourceLimit(
        () -> simple.render(Map.of(), output, oneCharacter),
        "Maximum output length exceeded",
        location(12, 1, 13));
    assertEquals("unchanged", output.toString());

    assertCumulativeOutputBoundary(
        "{% set x %}abcdef{% endset %}{{ x }}", 12, "abcdef", location(32, 1, 33));
    assertCumulativeOutputBoundary(
        "{% filter upper %}abcdef{% endfilter %}", 12, "ABCDEF", location(0, 1, 1));
    assertCumulativeOutputBoundary(
        "{% macro wrap() %}<{{ caller() }}>{% endmacro %}{% call wrap() %}abcdef{% endcall %}",
        22, "<abcdef>", location(48, 1, 49));
  }

  @Test
  void tojsonOutputGuardsHaveExactBoundaries() {
    // An empty object at depth zero renders as exactly "{\n}"; this is the renderedLength boundary.
    var prettyJson = Template.parse("{{ {} | tojson(indent=2) }}");
    assertEquals(
        "{\n}", prettyJson.render(Map.of(), RenderOptions.builder().maxOutputLength(3).build()));
    assertResourceLimit(
        () -> prettyJson.render(Map.of(), RenderOptions.builder().maxOutputLength(2).build()),
        "Maximum render output length exceeded",
        location(3, 1, 4));

    // An empty object's rendered length is 3 regardless of indent, so only jsonIndent prevents
    // this from allocating the indent before its output is charged.
    var wideIndent = Template.parse("{{ {} | tojson(indent=5) }}");
    assertEquals(
        "{\n}", wideIndent.render(Map.of(), RenderOptions.builder().maxOutputLength(5).build()));
    assertResourceLimit(
        () -> wideIndent.render(Map.of(), RenderOptions.builder().maxOutputLength(4).build()),
        "Maximum render output length exceeded",
        location(3, 1, 4));

    // A non-empty container lets the renderedLength guard reject its large indent first.
    assertResourceLimit(
        () ->
            Template.parse("{{ [1,2] | tojson(indent=5000000) }}")
                .render(Map.of(), RenderOptions.builder().maxOutputLength(10).build()),
        "Maximum render output length exceeded",
        location(3, 1, 4));
  }

  private static void assertCumulativeOutputBoundary(
      String source,
      int cumulativeLength,
      String expectedOutput,
      Optional<SourceLocation> failureLocation) {
    var template = Template.parse(source);
    assertEquals(
        expectedOutput,
        template.render(
            Map.of(), RenderOptions.builder().maxOutputLength(cumulativeLength).build()));
    assertResourceLimit(
        () ->
            template.render(
                Map.of(), RenderOptions.builder().maxOutputLength(cumulativeLength - 1).build()),
        "Maximum output length exceeded",
        failureLocation);
  }

  private static void assertResourceLimit(
      Executable executable, String message, Optional<SourceLocation> expectedLocation) {
    var error = assertThrows(TemplateRenderException.class, executable);
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
    assertEquals(message, error.getMessage());
    assertEquals(expectedLocation, error.location());
  }

  private static Optional<SourceLocation> location(int offset, int line, int column) {
    return Optional.of(new SourceLocation(offset, line, column));
  }
}
