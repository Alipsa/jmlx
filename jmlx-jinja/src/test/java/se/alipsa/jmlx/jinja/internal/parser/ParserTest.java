package se.alipsa.jmlx.jinja.internal.parser;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.TemplateOptions;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.TemplateSyntaxException;
import se.alipsa.jmlx.jinja.internal.ast.Expression;
import se.alipsa.jmlx.jinja.internal.ast.Statement;
import se.alipsa.jmlx.jinja.internal.lexer.Lexer;

class ParserTest {
  private static final TemplateOptions RAW =
      TemplateOptions.builder().trimBlocks(false).lstripBlocks(false).build();

  @Test
  void parsesTextAndComment() {
    var program = parse("a{# note #}b");
    assertEquals(3, program.body().size());
    assertEquals(" note ", ((Statement.Comment) program.body().get(1)).value());
  }

  @Test
  void reportsLocatedUnexpectedEnd() {
    var error = assertThrows(TemplateSyntaxException.class, () -> parse("{{ variable }}{{"));
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals(14, error.location().orElseThrow().offset());
  }

  @Test
  void preservesDescriptiveEndOfInputDiagnostics() {
    assertEquals(
        "Unexpected end of template",
        assertThrows(TemplateSyntaxException.class, () -> parse("{{")).getMessage());
    assertEquals(
        "Unknown statement, got end of template",
        assertThrows(TemplateSyntaxException.class, () -> parse("{%")).getMessage());
    assertEquals(
        "Parser Error: Expected closing expression token. End of template !== CloseExpression.",
        assertThrows(TemplateSyntaxException.class, () -> parse("{{''")).getMessage());
  }

  @Test
  void concatenatesStringsAndPreservesDuplicateObjectKeys() {
    assertEquals("abc", ((Expression.StringLiteral) expression("{{ 'a' 'b' 'c' }}")).value());
    assertEquals(2, ((Expression.ObjectLiteral) expression("{{ {'a':1,'a':2} }}")).value().size());
  }

  @Test
  void parsesPrecedenceAndStatements() {
    var outer = (Expression.BinaryExpression) expression("{{ 'a' in 'apple' == 'b' in 'banana' }}");
    assertEquals("in", outer.operator().value());
    assertEquals("in", outer.operator().value());
    assertEquals("==", ((Expression.BinaryExpression) outer.left()).operator().value());
    assertInstanceOf(
        Expression.MemberExpression.class,
        ((Expression.UnaryExpression) expression("{{ not test.x }}")).argument());
    assertInstanceOf(
        Statement.If.class, parse("{% if a %}yes{% else %}no{% endif %}").body().get(0));
    assertInstanceOf(
        Statement.For.class, parse("{% for x in xs %}{{ x }}{% endfor %}").body().get(0));
  }

  @Test
  void deeplyNestedInputIsResourceLimited() {
    var source = "{{ " + "(".repeat(5_000) + "1" + ")".repeat(5_000) + " }}";
    var error = assertThrows(TemplateRenderException.class, () -> parse(source));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
  }

  @Test
  void deeplyNestedElifChainsAreResourceLimited() {
    var source = new StringBuilder("{% if a %}x");
    for (var index = 0; index < 5_000; index++) {
      source.append("{% elif b").append(index).append(" %}y");
    }
    source.append("{% endif %}");
    var error = assertThrows(TemplateRenderException.class, () -> parse(source.toString()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
  }

  @Test
  void portsUpstreamTrailingCommaAndFilterQuirks() {
    assertThrows(TemplateSyntaxException.class, () -> expression("{{ (1,) }}"));
    assertInstanceOf(
        Statement.FilterStatement.class, parse("{% filter 42 %}x{% endfilter %}").body().get(0));
  }

  @Test
  void reportsUpstreamLoopVariableNodeTypes() {
    var integer = assertThrows(TemplateSyntaxException.class, () -> parse("{% for 1 in values %}"));
    assertEquals(
        "Expected identifier/tuple for the loop variable, got IntegerLiteral instead",
        integer.getMessage());
    var object = assertThrows(TemplateSyntaxException.class, () -> parse("{% for {} in values %}"));
    assertEquals(
        "Expected identifier/tuple for the loop variable, got ObjectLiteral instead",
        object.getMessage());
  }

  @Test
  void expressionRecordNamesUsedInDiagnosticsMatchVendoredAstDiscriminators() throws Exception {
    var astSource = Files.readString(Path.of("upstream/vendor/src/ast.ts"));
    var upstreamTypes = new HashSet<String>();
    var matcher = Pattern.compile("type\\s*=\\s*\\\"([^\\\"]+)\\\"").matcher(astSource);
    while (matcher.find()) {
      upstreamTypes.add(matcher.group(1));
    }
    for (var expressionType : Expression.class.getDeclaredClasses()) {
      if (Expression.class.isAssignableFrom(expressionType)) {
        assertTrue(
            upstreamTypes.contains(expressionType.getSimpleName()),
            () ->
                expressionType.getSimpleName()
                    + " is absent from the vendored upstream AST discriminators");
      }
    }
  }

  @Test
  void oversizedIntegersRemainIntegerLiterals() {
    assertInstanceOf(Expression.IntegerLiteral.class, expression("{{ 123456789012345678901234 }}"));
  }

  private static Statement.Program parse(String source) {
    return Parser.parse(Lexer.tokenize(source, RAW), RAW);
  }

  private static Expression expression(String source) {
    return (Expression) parse(source).body().get(0);
  }
}
