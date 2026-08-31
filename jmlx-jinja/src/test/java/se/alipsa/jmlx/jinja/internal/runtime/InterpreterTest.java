package se.alipsa.jmlx.jinja.internal.runtime;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.RenderOptions;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.Template;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.internal.JsFormat;
import se.alipsa.jmlx.jinja.internal.ast.Expression;

class InterpreterTest {
  @Test
  void treatsNaNAsFalsy() {
    assertFalse(
        Interpreter.truthy(new se.alipsa.jmlx.jinja.internal.Value.IntegerValue(Double.NaN)));
    assertFalse(Interpreter.truthy(new se.alipsa.jmlx.jinja.internal.Value.FloatValue(Double.NaN)));
    assertTrue(Interpreter.truthy(new se.alipsa.jmlx.jinja.internal.Value.IntegerValue(1d)));
    assertTrue(Interpreter.truthy(new se.alipsa.jmlx.jinja.internal.Value.FloatValue(1d)));
  }

  @Test
  void evaluatesTernaryExpressionsWithPinnedRuntimeTruthiness() {
    assertEquals(
        "yes|no|no|no|no|no|no|no|no|no|yes|yes|yes|yes|yes|yes|yes|yes",
        Template.parse(
                "{{ 'yes' if true else 'no' }}|{{ 'yes' if false else 'no' }}|"
                    + "{{ 'yes' if missing else 'no' }}|{{ 'yes' if none else 'no' }}|"
                    + "{{ 'yes' if '' else 'no' }}|{{ 'yes' if 0 else 'no' }}|"
                    + "{{ 'yes' if 0.0 else 'no' }}|{{ 'yes' if [] else 'no' }}|"
                    + "{{ 'yes' if {} else 'no' }}|{{ 'yes' if (0.0 / 0.0) else 'no' }}|"
                    + "{{ 'yes' if 'x' else 'no' }}|{{ 'yes' if 1 else 'no' }}|"
                    + "{{ 'yes' if 1.0 else 'no' }}|{{ 'yes' if [1] else 'no' }}|"
                    + "{{ 'yes' if {'x': 1} else 'no' }}|{{ 'yes' if (1, 2) else 'no' }}|"
                    + "{{ 'yes' if namespace(a=1) else 'no' }}|{{ 'yes' if range else 'no' }}")
            .render(Map.of()));
    assertEquals(
        "b", Template.parse("{{ 'a' if false else ('b' if true else 'c') }}").render(Map.of()));
    assertEquals(
        "safe",
        Template.parse("{{ raise_exception('boom') if false else 'safe' }}").render(Map.of()));
    var selectedBranch =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{{ raise_exception('boom') if true else 'safe' }}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.EXPLICIT_RAISE, selectedBranch.category());
    assertEquals(new SourceLocation(3, 1, 4), selectedBranch.location().orElseThrow());
  }

  @Test
  void evaluatesSequenceSlicesWithPinnedRuntimeSemantics() {
    assertEquals(
        "[1, 3]|[1, 2, 3]|[]|[]|[]||[2, 1]|😀B|olleh",
        Template.parse(
                "{{ [0,1,2,3,4][1:4:2] }}|{{ [0,1,2,3,4][-4:-1] }}|"
                    + "{{ [0,1,2][3000000000:] }}|{{ [0,1][::0] }}|{{ [][:] }}|"
                    + "{{ ''[:] }}|{{ (1,2)[::-1] }}|{{ 'A😀BC'[1:3] }}|"
                    + "{{ 'hello'[::-1] }}")
            .render(Map.of()));
    assertEquals(
        "[0, 1, 2]|[2]|[]",
        Template.parse("{{ [0,1,2][-3000000000:] }}|{{ [0,1,2][(1 + 1):] }}|{{ [][:] }}")
            .render(Map.of()));
    assertEquals("false", Template.parse("{{ (1,2)[:] is string }}").render(Map.of()));

    var receiver =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ {'a': 1}[:] }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, receiver.category());
    assertEquals(new SourceLocation(3, 1, 4), receiver.location().orElseThrow());
    assertEquals("Slice object must be an array or string", receiver.getMessage());
    var receiverBeforeBounds =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ {'a': 1}[missing + 1:] }}").render(Map.of()));
    assertEquals("Slice object must be an array or string", receiverBeforeBounds.getMessage());

    var component =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ [1]['x':] }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, component.category());
    assertEquals(new SourceLocation(7, 1, 8), component.location().orElseThrow());
    assertEquals("Slice start must be numeric or undefined", component.getMessage());
    assertEquals(
        "Slice stop must be numeric or undefined",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ [1][:'x'] }}").render(Map.of()))
            .getMessage());
    assertEquals(
        "Slice step must be numeric or undefined",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ [1][::true] }}").render(Map.of()))
            .getMessage());
    assertEquals(
        "Slice start must be numeric or undefined",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ [1][1.0:] }}").render(Map.of()))
            .getMessage());
    assertEquals(
        "Slice start must be numeric or undefined",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ [0,1,2][(6 / 2):] }}").render(Map.of()))
            .getMessage());

    var evaluatedBeforeValidation =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ [0,1]['a': missing + 1] }}").render(Map.of()));
    assertEquals(
        "Cannot perform operation + on undefined values", evaluatedBeforeValidation.getMessage());
    assertEquals(new SourceLocation(14, 1, 15), evaluatedBeforeValidation.location().orElseThrow());

    // Upstream throws TypeError: undefined is not iterable (cannot read property
    // Symbol(Symbol.iterator)).
    var undefinedSlice =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'abc'[9][0:1] }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, undefinedSlice.category());
    assertEquals("undefined is not iterable", undefinedSlice.getMessage());
    var undefinedProperty =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'abc'[9][missing + 1] }}").render(Map.of()));
    assertEquals("Cannot perform operation + on undefined values", undefinedProperty.getMessage());
  }

  @Test
  void evaluatesExpressionOperatorsWithUpstreamValueSemantics() {
    assertEquals("03", Template.parse("{{ 0 and 5 }}{{ 3 or 5 }}").render(Map.of()));
    assertEquals(
        "false", Template.parse("{{ false and raise_exception('boom') }}").render(Map.of()));
    assertEquals("true", Template.parse("{{ true or raise_exception('boom') }}").render(Map.of()));
    assertEquals(
        "truefalse", Template.parse("{{ none == missing }}{{ none != missing }}").render(Map.of()));
    assertEquals(
        "falsetruefalsetrue",
        Template.parse("{{ none == 1 }}{{ none != 1 }}{{ missing == 1 }}{{ missing != 1 }}")
            .render(Map.of()));
    assertEquals(
        "Cannot perform operation on null values",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ 1 + none }}").render(Map.of()))
            .getMessage());
    assertEquals("true", Template.parse("{{ '1' == true }}").render(Map.of()));
    assertEquals(
        "truetruetruetrue",
        Template.parse("{{ 1 == true }}{{ 0 == false }}{{ true == 1 }}{{ false == 0 }}")
            .render(Map.of()));
    assertEquals(
        "truetruefalse",
        Template.parse("{{ true == true }}{{ false == false }}{{ true != true }}")
            .render(Map.of()));
    assertEquals(
        "truefalse",
        Template.parse("{{ 'abc'[5] == missing }}{{ 'abc'[5] != missing }}").render(Map.of()));
    assertEquals(
        "3|3.0|ab|true|true",
        Template.parse(
                "{{ 1 + 2 }}|{{ 6.0 / 2 }}|{{ 'a' ~ 'b' }}|{{ 'a' in 'cat' }}|{{ 'a' in {'a': 1}"
                    + " }}")
            .render(Map.of()));
    assertEquals("[1, 2]", Template.parse("{{ [1] + [2] }}").render(Map.of()));
    assertEquals(
        "[1, 2, 3]|true", Template.parse("{{ (1, 2) + [3] }}|{{ 1 in (1, 2) }}").render(Map.of()));
    assertEquals(
        "[object Map]|1.0|[object Map]",
        Template.parse("{{ {'a': 1} ~ '' }}|{{ [1.0] ~ '' }}|{{ '' + {'a': 1} }}")
            .render(Map.of()));
    assertEquals("1|1.0", Template.parse("{{ 1.0 ~ '' }}|{{ [1.0] ~ '' }}").render(Map.of()));
    assertEquals("undefined", Template.parse("{{ [none] ~ '' }}").render(Map.of()));
    var directUndefined =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'abc'[5] ~ '' }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, directUndefined.category());
    assertEquals(
        "Cannot read properties of undefined (reading 'toString')", directUndefined.getMessage());
    var nestedTuple =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ [(1, 2)] ~ '' }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, nestedTuple.category());
    assertEquals("Cannot convert to JSON: TupleValue", nestedTuple.getMessage());
    assertEquals(
        "truetruetrue",
        Template.parse("{{ 1 < 1.5 }}{{ 1.0 <= 1 }}{{ 1.0 in [1] }}").render(Map.of()));
    assertEquals("false", Template.parse("{{ 'x' in missing }}").render(Map.of()));
    assertEquals("true", Template.parse("{{ 'x' not in missing }}").render(Map.of()));
    var arrayNeedle =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ (1, 2) in [(1, 2)] }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, arrayNeedle.category());
    assertEquals(
        "Unknown operator \"in\" between TupleValue and ArrayValue", arrayNeedle.getMessage());
  }

  @Test
  void evaluatesSelectedFiltersAndTestsWithPinnedRuntimeSemantics() {
    assertEquals(
        "x|true|x|abc|3|1-true-x|12|1.25|fallback|{\"b\": 1}",
        Template.parse(
                "{{ missing | default('x') }}|{{ none | default('x') is none }}|"
                    + "{{ false | default('x', true) }}|{{ '  AbC  ' | trim | lower }}|"
                    + "{{ 'ABC' | lower | length }}|{{ [1, true, 'x'] | join('-') }}|"
                    + "{{ '12.9x' | int }}|{{ '1.25x' | float }}|"
                    + "{{ 'bad' | int(default='fallback') }}|{{ {'b': 1} | tojson }}")
            .render(Map.of()));
    assertEquals(
        "truetruetruetruetruetruetruetruefalse",
        Template.parse(
                "{{ missing is undefined }}{{ none is defined }}{{ none is none }}"
                    + "{{ true is boolean }}{{ 1 is number }}{{ 'x' is string }}"
                    + "{{ [1] is iterable }}{{ {'x': 1} is sequence }}{{ (1, 2) is iterable }}")
            .render(Map.of()));
    assertEquals(
        "undefined|false|true",
        Template.parse(
                "{{ 'abc'[5] | default('fallback') }}|{{ 'abc'[5] is undefined }}|"
                    + "{{ 'abc'[5] is defined }}")
            .render(Map.of()));
    assertEquals("a-b-c", Template.parse("{{ 'abc' | join('-') }}").render(Map.of()));
    var emptyFirst =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{% if [] | first %}x{% endif %}").render(Map.of()));
    assertEquals(
        "Cannot read properties of undefined (reading '__bool__')", emptyFirst.getMessage());
    assertEquals("null", Template.parse("{{ (1.0 / 0.0) | tojson }}").render(Map.of()));
    assertEquals(
        "1|1.0|1|-2|Infinity|Infinity",
        Template.parse(
                "{{ true | int }}|{{ true | float }}|{{ 1.9 | int }}|{{ (0 - 1.9) | int }}|"
                    + "{{ (1.0 / 0.0) | int }}|{{ (1.0 / 0.0) | float }}")
            .render(Map.of()));
  }

  @Test
  void categorizesFilterAndTestFailures() {
    var unknownFilter =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'x' | no_such_filter }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, unknownFilter.category());
    assertTrue(unknownFilter.location().isPresent());
    var receiver =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 1 | lower }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, receiver.category());
    var arity =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'x' | trim(1) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, arity.category());
    assertEquals("{\n}", Template.parse("{{ {} | tojson(indent=2) }}").render(Map.of()));
    var defaultFlag =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ missing | default('x', 1) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, defaultFlag.category());
    var defaultIdentifier =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'x' | default }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, defaultIdentifier.category());
    assertEquals("x", Template.parse("{{ 'x' | join(other='-') }}").render(Map.of()));
    var unknownTest =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 1 is no_such_test }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, unknownTest.category());
    var deferredEquality =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 1 is equalto }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, deferredEquality.category());
  }

  @Test
  void pinsFilterEdgeCasesAgainstNodeOracle() {
    assertEquals("abc", Template.parse("{{ '\u00a0abc\u00a0' | trim }}").render(Map.of()));
    assertEquals(
        "Infinity|Infinity|0.0",
        Template.parse("{{ 'Infinity' | float }}|{{ 'Infinityx' | float }}|{{ 'NaN' | float }}")
            .render(Map.of()));
    assertEquals("1,2-3,4", Template.parse("{{ [[1,2], [3,4]] | join('-') }}").render(Map.of()));
    assertEquals(
        "1--x-", Template.parse("{{ [1, missing, 'x', none] | join('-') }}").render(Map.of()));
    assertEquals(
        "null|[null]|{\"a\": null}",
        Template.parse(
                "{{ missing | tojson }}|{{ [missing] | tojson }}|"
                    + "{{ {'a': missing} | tojson }}")
            .render(Map.of()));
  }

  @Test
  void distinguishesUnaryRawTruthinessAndOperatorErrors() {
    assertEquals("falsefalse", Template.parse("{{ not [] }}{{ not {} }}").render(Map.of()));
    assertEquals(
        "falsefalse",
        Template.parse(
                "{% if [] %}true{% else %}false{% endif %}{% if {} %}true{% else %}false{% endif"
                    + " %}")
            .render(Map.of()));
    assertEquals(
        "0.0|-Infinity",
        Template.parse("{{ 0.0 * (0.0 - 1.0) }}|{{ 1.0 / (0.0 * (0.0 - 1.0)) }}").render(Map.of()));
    assertEquals(
        "Infinity|-Infinity",
        Template.parse("{{ 1.0 / 0.0 }}|{{ 1.0 / (0.0 * (0.0 - 1.0)) }}").render(Map.of()));
    assertEquals("false", Template.parse("{{ none == 0 }}").render(Map.of()));
    var unsupported =
        assertThrows(
            TemplateRenderException.class, () -> Template.parse("{{ true + 1 }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, unsupported.category());
  }

  @Test
  void rendersCoreLiteralsCallsAndAssignments() {
    assertEquals("12", Template.parse("{% set (a, b) = (1, 2) %}{{ a }}{{ b }}").render(Map.of()));
    assertEquals("[0, 1, 2]", Template.parse("{{ range(3) }}").render(Map.of()));
  }

  @Test
  void preservesTupleRenderQuirk() {
    var error =
        assertThrows(
            TemplateRenderException.class, () -> Template.parse("{{ (1, 2) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
  }

  @Test
  void rendersLoopsMembersAndAliases() {
    assertEquals(
        "123",
        Template.parse("{% for x in range(3) %}{{ loop.index }}{% endfor %}").render(Map.of()));
    assertEquals(
        "2",
        Template.parse("{% set y = x %}{% set y.a = 2 %}{{ x.a }}").render(Map.of("x", Map.of())));
    assertEquals("[0, 1, 2]", Template.parse("{% set r = range %}{{ r(3) }}").render(Map.of()));
    assertEquals(
        "EMPTY",
        Template.parse("{% for x in [1, 2] if false %}{{ x }}{% else %}EMPTY{% endfor %}")
            .render(Map.of()));
    assertEquals(
        "1|3",
        Template.parse(
                "{% set xs = [{'x': ''}, {'x': false}, {'x': []}, {'x': 'a'}] %}"
                    + "{{ xs|selectattr('x')|length }}|{{ xs|rejectattr('x')|length }}")
            .render(Map.of()));
  }

  @Test
  void retainsAssignmentsAcrossIfBodiesAndExposesCompleteLoopMetadata() {
    assertEquals(
        "1", Template.parse("{% if true %}{% set x = 1 %}{% endif %}{{ x }}").render(Map.of()));
    assertEquals(
        "12",
        Template.parse(
                "{% for i in [1,2] %}{% if true %}{% set s = i %}{% endif %}{{ s }}{% endfor %}")
            .render(Map.of()));
    assertEquals(
        "-2;1-;",
        Template.parse("{% for i in [1,2] %}{{ loop.previtem }}-{{ loop.nextitem }};{% endfor %}")
            .render(Map.of()));
    assertEquals(
        "D11",
        Template.parse(
                "{% for i in [1,2,3] %}{% break %}{% else %}D{{ loop.index }}{{ i }}{% endfor %}")
            .render(Map.of()));
    assertEquals(
        "23",
        Template.parse(
                "{% for i in [1,2,3] %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% endfor %}")
            .render(Map.of()));
    assertEquals(
        "D",
        Template.parse("{% for i in [1,2,3] %}{% continue %}{% else %}D{% endfor %}")
            .render(Map.of()));
  }

  @Test
  void distinguishesInvalidMemberAndObjectKeyTypes() {
    assertEquals("", Template.parse("{{ 'abc'.foo }}").render(Map.of()));
    var objectProperty =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ obj[1] }}").render(Map.of("obj", Map.of("a", 1))));
    assertEquals(ErrorCategory.TYPE, objectProperty.category());
    var objectKey =
        assertThrows(
            TemplateRenderException.class, () -> Template.parse("{{ {1: 'a'} }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, objectKey.category());
    var raised =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ raise_exception(none) }}").render(Map.of()));
    assertEquals(ErrorCategory.EXPLICIT_RAISE, raised.category());
  }

  @Test
  void rangeMatchesJavaScriptMissingAndCoercedArguments() {
    assertEquals("[]", Template.parse("{{ range() }}").render(Map.of()));
    assertEquals("[\"3\"]", Template.parse("{{ range('3', 5, 1, 99) }}").render(Map.of()));
  }

  @Test
  void rangeIsBoundedBeforeItAllocatesAndLoopNamesCanShadowMetadata() {
    var options = RenderOptions.builder().maxLoopIterations(10).build();
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{% for x in range(20) %}{% break %}{% endfor %}")
                    .render(Map.of(), options));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
    assertEquals(
        "1;2;", Template.parse("{% for loop in [1,2] %}{{ loop }};{% endfor %}").render(Map.of()));
  }

  @Test
  void repeatedHostMapsDoNotAliasAfterTemplateMutation() {
    var shared = Map.of("k", 1);
    assertEquals(
        "{\"k\": 2}{\"k\": 1}",
        Template.parse("{% set a.k = 2 %}{{ a }}{{ b }}").render(Map.of("a", shared, "b", shared)));
  }

  @Test
  void rangeTreatsNoneAndUndefinedAsJavaScriptUndefined() {
    assertEquals("[0, 1, 2, 3, 4]", Template.parse("{{ range(0, 5, none) }}").render(Map.of()));
    assertEquals(
        "[0, 1, 2, 3, 4]", Template.parse("{{ range(0, 5, undefinedvar) }}").render(Map.of()));
    assertEquals("[]", Template.parse("{{ range(none, 3) }}").render(Map.of()));
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ range(0, 5, 0.0) }}").render(Map.of()));
    assertEquals(ErrorCategory.VALUE, error.category());
  }

  @Test
  void globalsIgnoreExtraArgumentsAndRenderStableTimeDigits() {
    assertEquals(
        "",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ raise_exception() }}").render(Map.of()))
            .getMessage());
    assertEquals(
        "a",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ raise_exception('a', 'b') }}").render(Map.of()))
            .getMessage());
    assertEquals(
        "1,2",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ raise_exception([1, 2]) }}").render(Map.of()))
            .getMessage());
    var options =
        RenderOptions.builder()
            .clock(Clock.fixed(Instant.parse("2026-08-21T09:05:00Z"), ZoneId.of("UTC")))
            .zoneId(ZoneId.of("UTC"))
            .build();
    assertEquals(
        "2026", Template.parse("{{ strftime_now('%Y', 'ignored') }}").render(Map.of(), options));
    assertEquals(
        "a\0%", Template.parse("{{ strftime_now(fmt) }}").render(Map.of("fmt", "a\0%%"), options));
    var previous = Locale.getDefault(Locale.Category.FORMAT);
    try {
      Locale.setDefault(Locale.Category.FORMAT, Locale.forLanguageTag("ar-SA"));
      assertEquals(
          "2026-08-21 09:05",
          Template.parse("{{ strftime_now('%Y-%m-%d %H:%M') }}").render(Map.of(), options));
    } finally {
      Locale.setDefault(Locale.Category.FORMAT, previous);
    }
  }

  @Test
  void strftimeNowRejectsMissingOrKeywordOnlyFormatWithType() {
    // {{ strftime_now() }}: no arguments at all, so `a.isEmpty()` is true and the
    // `a.isEmpty() || a.get(0) instanceof Value.KeywordArgumentsValue` guard in
    // Interpreter.strftime throws directly -- argument(a, 0) is never called for this case.
    var missingArgumentError =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ strftime_now() }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, missingArgumentError.category());
    assertEquals("strftime_now() expected one string argument", missingArgumentError.getMessage());

    // {% call strftime_now() %}...{% endcall %}: evaluateCallStatement appends a
    // KeywordArgumentsValue bag as the last element of `arguments`, landing at index 0 only
    // because this call has no positionals (even with no keywords), so the same guard's
    // `a.get(0) instanceof Value.KeywordArgumentsValue` check catches it directly -- argument(a,
    // 0) is never reached, exactly like the keyword-only case below.
    var callBlockError =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{% call strftime_now() %}{% endcall %}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, callBlockError.category());
    assertEquals("strftime_now() expected one string argument", callBlockError.getMessage());

    // strftime_now(fmt='%Y'): a keyword-only call expression. evaluateCallExpression's `call()`
    // only pushes the KeywordArgumentsValue bag when keywords are non-empty, but that is exactly
    // the case here, so arguments.get(0) is again the bag and the same guard intercepts it before
    // argument(a, 0) is ever called.
    var keywordOnlyError =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ strftime_now(fmt='%Y') }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, keywordOnlyError.category());
    assertEquals("strftime_now() expected one string argument", keywordOnlyError.getMessage());
  }

  @Test
  void strftimeNowMissingFormatGuardNormalizesNamespaceValuesToType() {
    // `namespace` returns its keyword bag verbatim when given keywords (Environment.namespace),
    // so `strftime_now(namespace(a=1))` reaches Interpreter.strftime as `a=[KeywordArgumentsValue],
    // k=false` -- structurally identical to `{% call strftime_now() %}` even though a real
    // positional argument was supplied. The missing-format guard cannot tell these apart (see the
    // comment above Interpreter.strftime's guard), and normalizes the ambiguity to TYPE to match
    // upstream's TypeError family. This test pins that intentional category decision.
    var namespaceWithKeywordsError =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ strftime_now(namespace(a=1)) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, namespaceWithKeywordsError.category());
    assertEquals(
        "strftime_now() expected one string argument", namespaceWithKeywordsError.getMessage());

    // Contrast: `namespace()` with no keywords returns a fresh ObjectValue, not the bag, so it
    // does not trip the guard and instead falls through to the TYPE check.
    var namespaceEmptyError =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ strftime_now(namespace()) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, namespaceEmptyError.category());
    assertEquals("strftime_now() format must be a string", namespaceEmptyError.getMessage());
  }

  @Test
  void strftimeNowRequiresBothClockAndZoneIndependently() {
    // Both cases use Template.parse(source).render(context, options) directly (not
    // raisedMessage, which has no options parameter) with a RenderOptions.builder() that sets
    // only one of clock/zoneId: raisedMessage cannot express a partially-built options object.
    var missingClockError =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{{ strftime_now('%Y') }}")
                    .render(Map.of(), RenderOptions.builder().zoneId(ZoneId.of("UTC")).build()));
    assertEquals(ErrorCategory.VALUE, missingClockError.category());
    assertEquals("strftime_now requires clock and zone", missingClockError.getMessage());

    var missingZoneError =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{{ strftime_now('%Y') }}")
                    .render(
                        Map.of(),
                        RenderOptions.builder()
                            .clock(
                                Clock.fixed(
                                    Instant.parse("2026-08-21T09:05:00Z"), ZoneId.of("UTC")))
                            .build()));
    assertEquals(ErrorCategory.VALUE, missingZoneError.category());
    assertEquals("strftime_now requires clock and zone", missingZoneError.getMessage());
  }

  @Test
  void strftimeNowConvertsClockIntoSuppliedZone() {
    // Every other strftime_now fixture in this file uses the same zone for both the Clock and
    // zoneId (UTC), so none of them would notice if
    // `ZonedDateTime.now(o.clock().get()).withZoneSameInstant(o.zoneId().get())` in
    // Interpreter.strftime had its zone conversion deleted or its argument swapped. This mirrors
    // the oracle-verified `self.strftime-date-boundary` corpus record byte-for-byte (same
    // instant, zone, template, and expected text) to prove hfjinja's own interpreter performs
    // that cross-zone conversion, not just the pinned Node oracle.
    var options =
        RenderOptions.builder()
            .clock(Clock.fixed(Instant.parse("2026-01-01T00:30:00Z"), ZoneId.of("UTC")))
            .zoneId(ZoneId.of("America/Los_Angeles"))
            .build();
    assertEquals(
        "2025-12-31 16:30",
        Template.parse("{{ strftime_now('%Y-%m-%d %H:%M') }}").render(Map.of(), options));
  }

  // The message "strftime_now() format must be a string" is deliberately distinct from the
  // missing-format message ("strftime_now() expected one string argument") asserted above: a
  // message-based assertion -- such as `getMessage()` and `raisedMessage(...)` here -- can only
  // distinguish the two TYPE branches if their texts differ, not merely their ErrorCategory.
  // `none` and an undefined-backed value like `x.missing` both pass the missing-format guard (`a`
  // is
  // non-empty and `a.get(0)` is not a KeywordArgumentsValue) but then fail the
  // `instanceof Value.StringValue` check in Interpreter.strftime, so both land on the TYPE
  // branch below.
  //
  // The two sub-cases are wrapped in their own assertAll() lambdas rather than run as plain
  // sequential statements: a bare `assertEquals` failure throws immediately and would abort the
  // method, silently skipping the `x.missing` sub-case if the `none` sub-case were to regress.
  // assertAll executes every lambda and aggregates failures, so a regression in either sub-case
  // is independently observed instead of being masked by the other.
  @Test
  void strftimeNowRejectsNonStringPresentFormatWithType() {
    assertAll(
        () -> {
          var noneError =
              assertThrows(
                  TemplateRenderException.class,
                  () -> Template.parse("{{ strftime_now(none) }}").render(Map.of()));
          assertEquals(ErrorCategory.TYPE, noneError.category());
          assertEquals("strftime_now() format must be a string", noneError.getMessage());
        },
        () -> {
          var undefinedBackedError =
              assertThrows(
                  TemplateRenderException.class,
                  () ->
                      Template.parse("{{ strftime_now(x.missing) }}")
                          .render(Map.of("x", Map.of())));
          assertEquals(ErrorCategory.TYPE, undefinedBackedError.category());
          assertEquals("strftime_now() format must be a string", undefinedBackedError.getMessage());
        });
  }

  @Test
  void matchesRuntimeExceptionStringificationAndCallEvaluationOrder() {
    assertEquals("1,2", raisedMessage("{{ raise_exception((1, 2)) }}", Map.of()));
    assertEquals("1", raisedMessage("{{ raise_exception(1.0) }}", Map.of()));
    assertEquals(
        "[object Map]", raisedMessage("{{ raise_exception(obj) }}", Map.of("obj", Map.of("a", 1))));
    assertEquals("undefined", raisedMessage("{{ raise_exception([none]) }}", Map.of()));
    assertEquals("[1, 2]", raisedMessage("{{ raise_exception([[1, 2]]) }}", Map.of()));
    assertEquals("boom", raisedMessage("{{ nofn(raise_exception('boom')) }}", Map.of()));
    assertEquals(
        "boom", raisedMessage("{{ obj[1](raise_exception('boom')) }}", Map.of("obj", Map.of())));
  }

  @Test
  void usesCallableSourceForCoercionAndExceptionMessages() {
    var source = Template.parse("{{ range }}").render(Map.of());
    assertEquals(
        source + "!|" + source + "!|" + source,
        Template.parse("{{ range ~ '!' }}|{{ range + '!' }}|{{ [range]|join(',') }}")
            .render(Map.of()));
    assertEquals(source, raisedMessage("{{ raise_exception(range) }}", Map.of()));
    assertEquals(source, raisedMessage("{{ raise_exception([range]) }}", Map.of()));
  }

  @Test
  void retainsTheKnownMarkerForUnportedCallableForms() {
    assertEquals(
        "<function>!|<function>!|<function>!|<function>!|<function>!|<function>!|<function>!",
        Template.parse(
                "{{ namespace ~ '!' }}|{{ 'ab'.upper ~ '!' }}|{{ 'ab'.startswith ~ '!' }}|{{"
                    + " ({'a':1}).keys ~ '!' }}|{{ ({'a':1}).items ~ '!' }}|{% macro f() %}x{%"
                    + " endmacro %}{{ f ~ '!' }}|{% macro wrap() %}{{ caller ~ '!' }}{% endmacro"
                    + " %}{% call wrap() %}x{% endcall %}")
            .render(Map.of()));
    assertEquals("<function>", raisedMessage("{{ raise_exception(namespace) }}", Map.of()));
  }

  @Test
  void rendersNamespaceKeywordArgumentsAndRejectsCyclicValues() {
    assertEquals("1", Template.parse("{% set ns = namespace(x=1) %}{{ ns.x }}").render(Map.of()));
    assertEquals(
        "2",
        Template.parse("{% set ns = namespace(x=1) %}{% set ns.x = 2 %}{{ ns.x }}")
            .render(Map.of()));
    assertEquals(
        "a", Template.parse("{% for k in namespace(a=1) %}{{ k }}{% endfor %}").render(Map.of()));
    assertEquals(
        "`namespace` expects either zero arguments or a single object argument",
        raisedMessage("{% set ns = namespace(x, y=1) %}{{ ns.a }}", Map.of("x", Map.of("a", 9))));
    assertEquals(
        "Cannot convert to JSON: KeywordArgumentsValue",
        raisedMessage("{% set ns = namespace(a=1,b=2) %}{{ ns }}", Map.of()));
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{% set ns = namespace() %}{% set ns.self = ns %}{{ ns }}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
  }

  @Test
  void rangeReclassifiesIntegralFloatStarts() {
    assertEquals(
        "2;3;4;",
        Template.parse("{% for i in range(2.0, 5) %}{{ i }};{% endfor %}").render(Map.of()));
  }

  @Test
  void rangeUsesJavaScriptStepBranchingAndNumberCoercion() {
    assertEquals("[3]", Template.parse("{{ range(3, 0, 'x') }}").render(Map.of()));
    assertEquals("[0, 1, 2]", Template.parse("{{ range(0, '0x3') }}").render(Map.of()));
    assertEquals("[]", Template.parse("{{ range(0, '1d') }}").render(Map.of()));
    assertEquals("[0, 1, 2]", Template.parse("{{ range(0, '\u00a03\u00a0') }}").render(Map.of()));
    assertEquals("[0, 1, 2]", Template.parse("{{ range(0, '\u20073\u2007') }}").render(Map.of()));
    assertEquals("[]", Template.parse("{{ range(0, '\u001c3\u001c') }}").render(Map.of()));
  }

  @Test
  void rejectsPositionalArgumentsAfterKeywordsBeforeEvaluation() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ namespace(a=1, raise_exception('boom')) }}").render(Map.of()));
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals("Positional arguments must come before keyword arguments", error.getMessage());
  }

  @Test
  void invalidAssignmentUsesJsonAstDiagnostic() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{% set [1] = 2 %}").render(Map.of()));
    assertEquals(
        "Invalid LHS inside assignment expression:"
            + " {\"type\":\"ArrayLiteral\",\"value\":[{\"type\":\"IntegerLiteral\",\"value\":1}]}",
        error.getMessage());
  }

  @Test
  void loopIterableUsesItsFreshScopeNamespaceBinding() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse(
                        "{% set namespace = [1] %}{% for x in namespace %}{{ x }}{% endfor %}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
  }

  @Test
  void renderBudgetsLimitStepsAndOutput() {
    var stepError =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{{ 1 }}{{ 2 }}")
                    .render(Map.of(), RenderOptions.builder().maxSteps(1).build()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, stepError.category());
    var outputError =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{{ 'ab' }}")
                    .render(Map.of(), RenderOptions.builder().maxOutputLength(1).build()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, outputError.category());
  }

  private static String raisedMessage(String source, Map<String, ?> context) {
    return assertThrows(TemplateRenderException.class, () -> Template.parse(source).render(context))
        .getMessage();
  }

  @Test
  void quoteEscapesUnpairedSurrogates() {
    assertEquals("\"x\\ud800y\"", JsFormat.quote("x\uD800y"));
  }

  @Test
  void outOfRangeStringIndexReturnsUndefinedString() {
    assertEquals("undefined", Template.parse("{{ 'abc'[5] }}").render(Map.of()));
    assertEquals(
        "no", Template.parse("{% if 'abc'[5] %}yes{% else %}no{% endif %}").render(Map.of()));
    assertEquals("[]", Template.parse("{{ ['abc'[5]] }}").render(Map.of()));
    assertEquals("[undefined]", Template.parse("{{ [missing] }}").render(Map.of()));
    assertEquals("", Template.parse("{{ {'undefined': 1}['abc'[5]] }}").render(Map.of()));
    assertEquals("[]", Template.parse("{{ range(0, 'abc'[5]) }}").render(Map.of()));
    assertEquals("[0, 1, 2]", Template.parse("{{ range(0, 3, 'abc'[5]) }}").render(Map.of()));
    assertEquals("", raisedMessage("{{ raise_exception('abc'[5]) }}", Map.of()));
    assertEquals("{undefined: 1}", Template.parse("{{ {'abc'[5]: 1} }}").render(Map.of()));
    assertEquals(
        "1", Template.parse("{% set o = {'abc'[5]: 1} %}{{ o['abc'[5]] }}").render(Map.of()));
  }

  @Test
  void spreadExpandsArraysAndTuplesAsPositionalCallArguments() {
    assertEquals("[1, 2, 3]", Template.parse("{{ range(*[1,4]) }}").render(Map.of()));
    assertEquals("[1, 2, 3]", Template.parse("{{ range(*(1,4)) }}").render(Map.of()));
  }

  @Test
  void spreadExpandsFilterArguments() {
    assertEquals("1-2", Template.parse("{{ [1,2] | join(*['-']) }}").render(Map.of()));
  }

  @Test
  void spreadExpandsExactlyOneLevel() {
    assertEquals("[]", Template.parse("{{ range(*[[1,4]]) }}").render(Map.of()));
  }

  @Test
  void spreadArgumentsPreserveUpstreamOrderingSemantics() {
    assertEquals("[1]", Template.parse("{{ range(*[1,4], 9) }}").render(Map.of()));
    assertEquals("[1]", Template.parse("{{ range(*[1,4], *[9]) }}").render(Map.of()));
    assertEquals("[]", Template.parse("{{ range(1, stop=4, *[2]) }}").render(Map.of()));
  }

  @Test
  void spreadAfterFilterKeywordBypassesPositionalAfterKeywordRejection() {
    assertEquals(
        "1+2", Template.parse("{{ [1,2] | join(separator='-', *['+']) }}").render(Map.of()));
  }

  @Test
  void filterArgumentsStillRejectOrdinaryPositionalAfterKeyword() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 1 | int(a=1, 2) }}").render(Map.of()));
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals("Positional arguments must come before keyword arguments", error.getMessage());
  }

  @Test
  void spreadArgumentEvaluationPrecedesCalleeValidation() {
    var validCallee =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ range(*[raise_exception('boom')]) }}").render(Map.of()));
    assertEquals(ErrorCategory.EXPLICIT_RAISE, validCallee.category());
    var invalidCallee =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ nofn(*[raise_exception('boom')]) }}").render(Map.of()));
    assertEquals(ErrorCategory.EXPLICIT_RAISE, invalidCallee.category());
  }

  @Test
  void spreadRejectsNonArrayNonTupleReceiversWithLocatedTypeError() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ range(*'14') }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
    assertEquals("Cannot unpack non-iterable type: StringValue", error.getMessage());
    assertEquals(new SourceLocation(9, 1, 10), error.location().orElseThrow());

    assertSpreadReceiverTypeError("{{ range(*{'a':1}) }}", "ObjectValue");
    assertSpreadReceiverTypeError("{{ range(*none) }}", "NullValue");
    assertSpreadReceiverTypeError("{{ range(*missing) }}", "UndefinedValue");
  }

  private static void assertSpreadReceiverTypeError(String source, String valueType) {
    var error =
        assertThrows(TemplateRenderException.class, () -> Template.parse(source).render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
    assertEquals("Cannot unpack non-iterable type: " + valueType, error.getMessage());
  }

  @Test
  void interpreterSourcePreservesEvaluatorKeywordOrder() throws Exception {
    var source =
        Files.readString(
            Path.of("src/main/java/se/alipsa/jmlx/jinja/internal/runtime/Interpreter.java"));
    int start = source.indexOf("private static EvaluatedArguments evaluateArguments(");
    int end = source.indexOf("\n  private static TemplateRenderException filterReceiver", start);
    assertTrue(start >= 0 && end > start);
    var body = source.substring(start, end);
    assertTrue(body.contains("var keywords = new LinkedHashMap<String, Value>();"));
    assertTrue(body.contains("Collections.unmodifiableMap(keywords)"));
    assertFalse(body.contains("Map.copyOf(keywords)"));
  }

  @Test
  void callFormJoinValidatesReceiverBeforeArguments() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 5 | join(*none) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
    assertEquals("Cannot apply filter \"join\" to type: IntegerValue", error.getMessage());
  }

  @Test
  void joinIgnoresSurplusSpreadArguments() {
    assertEquals("1-2", Template.parse("{{ [1,2] | join(*['-','+']) }}").render(Map.of()));
  }

  @Test
  void defaultIgnoresSurplusSpreadArguments() {
    assertEquals(
        "a", Template.parse("{{ missing | default(*['a', true, 'c']) }}").render(Map.of()));
  }

  @Test
  void intIgnoresSurplusSpreadArguments() {
    assertEquals("1", Template.parse("{{ 'x' | int(*[1,2,3]) }}").render(Map.of()));
  }

  @Test
  void floatIgnoresSurplusSpreadArguments() {
    assertEquals("1", Template.parse("{{ 'x' | float(*[1,2,3]) }}").render(Map.of()));
  }

  @Test
  void intIgnoresUnknownKeywords() {
    assertEquals("1", Template.parse("{{ '1' | int(a=1, b=2, c=3) }}").render(Map.of()));
  }

  @Test
  void tojsonIgnoresPositionalArguments() {
    assertEquals("[1, 2]", Template.parse("{{ [1,2] | tojson(*[9]) }}").render(Map.of()));
  }

  @Test
  void supportsMappingTestAndTojsonCallOptions() {
    assertEquals(
        "false|true|true|true|true|true|true|false|true|true|false|true",
        Template.parse(
                "{{ none is mapping }}|{{ {} is mapping }}|{% set empty = namespace() %}{{ empty is"
                    + " mapping }}|{% set populated = namespace(x=1) %}{{ populated is mapping"
                    + " }}|{{ [] is not mapping }}|{{ (1, 2) is not mapping }}|{{ 'x' is not"
                    + " mapping }}|{{ {} is iterable }}|{{ [] is iterable }}|{{ 'x' is iterable"
                    + " }}|{{ (1, 2) is iterable }}|{{ populated is sequence }}")
            .render(Map.of()));
    assertEquals(
        "{\"k\\u00e4\": \"v\\u007f\\ud83d\\ude00\"}",
        Template.parse("{{ {'k\u00e4': 'v\u007f😀'} | tojson(ensure_ascii=True) }}")
            .render(Map.of()));
    assertEquals(
        "{\"a\": 1}", Template.parse("{{ {'a': 1} | tojson(indent=0) }}").render(Map.of()));
    assertEquals(
        "{\n  \"a\": [\n    \n  ],\n  \"b\": {\n  }\n}",
        Template.parse("{{ {'a': [], 'b': {}} | tojson(indent=2) }}").render(Map.of()));
    assertEquals(
        "{\n  \"a\"=1;\n  \"b\"=2\n}",
        Template.parse("{{ {'a': 1, 'b': 2} | tojson(indent=2, separators=(';', '=')) }}")
            .render(Map.of()));
    assertEquals(
        "{\"_\": 5, \"a\": 3, \"ä\": 2, \"B\": 4, \"z\": 1}",
        Template.parse("{{ {'z': 1, 'ä': 2, 'a': 3, 'B': 4, '_': 5} | tojson(sort_keys=true) }}")
            .render(Map.of()));
    assertEquals(
        "{\"á\": 1, \"á\": 2}",
        Template.parse("{{ {'\u00e1': 1, 'a\u0301': 2} | tojson(sort_keys=true) }}")
            .render(Map.of()));
    assertEquals(
        "{\"a\": 2, undefined: 1}",
        Template.parse("{{ {'abc'[9]: 1, 'a': 2} | tojson(sort_keys=true) }}").render(Map.of()));
    assertEquals(
        "[, 1]|[1,2]|{\"a\"undefined1}",
        Template.parse(
                "{{ ['abc'[9], 1] | tojson }}|"
                    + "{{ [1, 2] | tojson(separators=('abc'[9], ':')) }}|"
                    + "{{ {'a': 1} | tojson(separators=(',', 'abc'[9])) }}")
            .render(Map.of()));
    assertEquals(
        "\"x\"|5|null",
        Template.parse(
                "{{ 'x' | tojson(indent=-1) }}|{{ 5 | tojson(indent=-1) }}|{{ none |"
                    + " tojson(indent=-1) }}")
            .render(Map.of()));
    assertEquals(
        "{\"a\": 1, \"b\": 2}|[1, 2]|{\"a\": [1]}",
        Template.parse(
                "{{ {'a': 1, 'b': 2} | tojson(indent=0 % 0) }}|"
                    + "{{ [1, 2] | tojson(indent=0 % 0) }}|"
                    + "{{ {'a': [1]} | tojson(indent=0 % 0) }}")
            .render(Map.of()));
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ {} | tojson(indent=-1) }}").render(Map.of()));
    assertEquals(ErrorCategory.VALUE, error.category());
    assertEquals("Invalid count value: -1", error.getMessage());
    var hugeIndent =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ {'a': 1} | tojson(indent=4294967298) }}").render(Map.of()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, hugeIndent.category());
    var nestedIndent =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{{ {'a': {'b': 1}} | tojson(indent=10) }}")
                    .render(Map.of(), RenderOptions.builder().maxOutputLength(20).build()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, nestedIndent.category());
    var typeError =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ {} | tojson(ensure_ascii=1) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, typeError.category());
  }

  @Test
  void macroBindsPositionalAndKeywordArgumentsWithDefaults() {
    assertEquals(
        "5-1",
        Template.parse("{% macro f(a,b=1) %}{{ a }}-{{ b }}{% endmacro %}{{ f(5) }}")
            .render(Map.of()));
  }

  @Test
  void macroMissingPositionalArgumentIsArity() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{% macro f(a) %}{{ a }}{% endmacro %}{{ f() }}").render(Map.of()));
    assertEquals(ErrorCategory.ARITY, error.category());
    assertEquals("Missing positional argument: a", error.getMessage());
  }

  @Test
  void macroIgnoresExtraPositionalArgumentsAndUnknownKeywords() {
    assertEquals(
        "1",
        Template.parse("{% macro f(a) %}{{ a }}{% endmacro %}{{ f(1,2,3) }}").render(Map.of()));
    assertEquals(
        "1",
        Template.parse("{% macro f(a=1) %}{{ a }}{% endmacro %}{{ f(z=9) }}").render(Map.of()));
  }

  @Test
  void macroPositionalArgumentWinsOverKeyword() {
    assertEquals(
        "5",
        Template.parse("{% macro f(a=1) %}{{ a }}{% endmacro %}{{ f(5, a=9) }}").render(Map.of()));
  }

  @Test
  void macroDefaultArgumentsEvaluateAtCallTimeInCallerScope() {
    assertEquals(
        "Hi shadowed",
        Template.parse(
                "{% set x='outer' %}{% macro g(name=x) %}Hi {{ name }}{% endmacro %}"
                    + "{% set x='shadowed' %}{{ g() }}")
            .render(Map.of()));
  }

  @Test
  void macroLaterDefaultArgumentSeesEarlierBoundParameter() {
    assertEquals(
        "1-1",
        Template.parse("{% macro f(a,b=a) %}{{ a }}-{{ b }}{% endmacro %}{{ f(1) }}")
            .render(Map.of()));
  }

  @Test
  void macroSpreadParameterIsSyntaxError() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{% macro f(*x) %}{{ x }}{% endmacro %}{{ f(1,2) }}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals("Unknown argument type: SpreadExpression", error.getMessage());
  }

  @Test
  void wp7Slice4MacroAndCallControlFlowParity() {
    assertAll(
        () ->
            assertEquals(
                "23",
                Template.parse(
                        "{% macro f(i) %}{% if i == 1 %}{% continue %}{% endif %}{% endmacro %}"
                            + "{% for i in [1,2,3] %}{{ i }}{{ f(i) }}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-continue-crosses-call-boundary"),
        () ->
            assertEquals(
                "",
                Template.parse(
                        "{% macro f(i) %}{% if i == 1 %}{% break %}{% endif %}{% endmacro %}"
                            + "{% for i in [1,2,3] %}{{ i }}{{ f(i) }}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-crosses-call-boundary"),
        () ->
            assertEquals(
                "23",
                Template.parse(
                        "{% macro f() %}{{ caller() }}{% endmacro %}"
                            + "{% for i in [1,2,3] %}{{ i }}{% call f() %}"
                            + "{% if i == 1 %}{% continue %}{% endif %}{% endcall %}{% endfor %}")
                    .render(Map.of()),
                "wp7.call-block-continue-crosses-caller-boundary"),
        () ->
            assertEquals(
                "",
                Template.parse(
                        "{% macro f() %}{{ caller() }}{% endmacro %}"
                            + "{% for i in [1,2,3] %}{{ i }}{% call f() %}"
                            + "{% if i == 1 %}{% break %}{% endif %}{% endcall %}{% endfor %}")
                    .render(Map.of()),
                "wp7.call-block-break-crosses-caller-boundary"),
        () ->
            assertEquals(
                "1M",
                Template.parse(
                        "{% macro f(i) %}M{% if i == 2 %}{% break %}{% endif %}{% endmacro %}"
                            + "{% for i in [1,2,3] %}{{ i }}{{ f(i) }}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-preserves-prior-iterations"),
        () ->
            assertEquals(
                "E",
                Template.parse(
                        "{% macro f() %}{% break %}{% endmacro %}"
                            + "{% for i in [1,2] %}{{ f() }}{% else %}E{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-loop-else"),
        () ->
            assertEquals(
                "E",
                Template.parse(
                        "{% macro f() %}{% continue %}{% endmacro %}"
                            + "{% for i in [1,2] %}{{ f() }}{% else %}E{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-continue-loop-else"),
        () ->
            assertEquals(
                "AZAZ",
                Template.parse(
                        "{% macro g(z) %}{% if z == 1 %}{% break %}{% endif %}{% endmacro %}{%"
                            + " macro f() %}{% for z in [1,2] %}{{ z }}{{ g(z) }}{% endfor %}Z{%"
                            + " endmacro %}{% for i in ['A','A'] %}{{ i }}{{ f() }}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-stops-at-inner-loop"),
        () ->
            assertEquals(
                "A2ZA2Z",
                Template.parse(
                        "{% macro g(z) %}{% if z == 1 %}{% continue %}{% endif %}{% endmacro %}{%"
                            + " macro f() %}{% for z in [1,2] %}{{ z }}{{ g(z) }}{% endfor %}Z{%"
                            + " endmacro %}{% for i in ['A','A'] %}{{ i }}{{ f() }}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-continue-stops-at-inner-loop"),
        () ->
            assertEquals(
                "AZAZ",
                Template.parse(
                        "{% macro g() %}{% break %}{% endmacro %}{% macro f() %}{% for z in [1,2]"
                            + " %}{{ caller(z) }}{% endfor %}Z{% endmacro %}{% for i in ['A','A']"
                            + " %}{{ i }}{% call(z) f() %}{{ z }}{{ g() }}{% endcall %}{% endfor"
                            + " %}")
                    .render(Map.of()),
                "wp7.call-block-break-stops-at-inner-loop"),
        () ->
            assertEquals(
                "A2ZA2Z",
                Template.parse(
                        "{% macro g(z) %}{% if z == 1 %}{% continue %}{% endif %}{% endmacro %}{%"
                            + " macro f() %}{% for z in [1,2] %}{{ caller(z) }}{% endfor %}Z{%"
                            + " endmacro %}{% for i in ['A','A'] %}{{ i }}{% call(z) f() %}{{ z"
                            + " }}{{ g(z) }}{% endcall %}{% endfor %}")
                    .render(Map.of()),
                "wp7.call-block-continue-stops-at-inner-loop"),
        () ->
            assertEquals(
                "",
                Template.parse(
                        "{% macro f() %}{% break %}{% endmacro %}{% for i in [1,2] %}{% filter"
                            + " upper %}{{ i }}{{ f() }}{% endfilter %}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-crosses-filter-block"),
        () ->
            assertEquals(
                "",
                Template.parse(
                        "{% macro f() %}{% break %}{% endmacro %}{% for i in [1,2] %}{% set x %}{{"
                            + " i }}{{ f() }}{% endset %}[{{ x }}]{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-crosses-set-capture"),
        () ->
            assertEquals(
                "",
                Template.parse(
                        "{% macro f() %}{% break %}{% endmacro %}"
                            + "{% for i in [1,2] %}{{ i }}{% call f() %}x{% endcall %}{% endfor %}")
                    .render(Map.of()),
                "wp7.call-block-callee-break-crosses-call-boundary"),
        () ->
            assertEquals(
                "",
                Template.parse(
                        "{% macro f() %}{% break %}{% endmacro %}{% for o in [1,2] %}O{% for i in"
                            + " [1,2] if f() %}{{ i }}{% endfor %}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-in-for-predicate"),
        () ->
            assertEquals(
                "",
                Template.parse(
                        "{% macro f() %}{% break %}{% endmacro %}{% for o in [1,2] %}O{% for i in"
                            + " [] %}{% else %}{{ f() }}{% endfor %}{% endfor %}")
                    .render(Map.of()),
                "wp7.macro-break-in-for-else"),
        () -> {
          var template =
              Template.parse(
                  "{% macro g(i) %}{% if i == 1 %}{% continue %}{% endif %}{% endmacro %}"
                      + "{% macro f(i,a=g(i)) %}{{ i }}{% endmacro %}"
                      + "{% for i in [1,2] %}{{ f(i) }}{% endfor %}");
          assertEquals("2", template.render(Map.of()), "wp7.macro-default-control-balances-depth");
          assertEquals(
              "2",
              template.render(Map.of(), RenderOptions.builder().maxMacroDepth(2).build()),
              "wp7.macro-default-control-balances-depth maxMacroDepth=2");
        });
  }

  @Test
  void macroBareBreak_hasCorpusComparableSyntaxContract() {
    // The pinned upstream exposes a raw blank-message BreakControl; the versioned classifier maps
    // that diagnostic family to SYNTAX, which makes this Java contract corpus-comparable.
    var source = "{% macro f() %}{% break %}{% endmacro %}{{ f() }}";
    var error =
        assertThrows(TemplateRenderException.class, () -> Template.parse(source).render(Map.of()));
    var call = source.indexOf("{{ f() }}") + 3;
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals("break or continue outside a for loop", error.getMessage());
    assertEquals(new SourceLocation(call, 1, call + 1), error.location().orElseThrow());
  }

  @Test
  void macroBareContinue_isKnownDivergenceFromUpstream() {
    var source = "{% macro f() %}{% continue %}{% endmacro %}{{ f() }}";
    var error =
        assertThrows(TemplateRenderException.class, () -> Template.parse(source).render(Map.of()));
    var call = source.indexOf("{{ f() }}") + 3;
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals("break or continue outside a for loop", error.getMessage());
    assertEquals(new SourceLocation(call, 1, call + 1), error.location().orElseThrow());
  }

  @Test
  void macroRedeclarationDoesNotThrow() {
    assertEquals(
        "2",
        Template.parse("{% macro f() %}1{% endmacro %}{% macro f() %}2{% endmacro %}{{ f() }}")
            .render(Map.of()));
  }

  @Test
  void callBlockInvokesCallerAndBindsCallerArguments() {
    assertEquals(
        "[body]",
        Template.parse(
                "{% macro f() %}[{{ caller() }}]{% endmacro %}{% call f() %}body{% endcall %}")
            .render(Map.of()));
    assertEquals(
        "[1-2]",
        Template.parse(
                "{% macro f() %}[{{ caller(1,2) }}]{% endmacro %}"
                    + "{% call(a,b) f() %}{{ a }}-{{ b }}{% endcall %}")
            .render(Map.of()));
  }

  @Test
  void callBlockBodySeesMacroLocalStateSetBeforeCallerRuns() {
    assertEquals(
        "[x=5]",
        Template.parse(
                "{% macro f() %}{% set x=5 %}[{{ caller() }}]{% endmacro %}"
                    + "{% call f() %}x={{ x }}{% endcall %}")
            .render(Map.of()));
  }

  @Test
  void callBlockNonIdentifierCallerParameterIsSyntaxError() {
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse(
                        "{% macro f() %}{{ caller(1) }}{% endmacro %}"
                            + "{% call(a.b) f() %}{{ a }}{% endcall %}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals(
        "Caller parameter must be an identifier, got MemberExpression", error.getMessage());
  }

  @Test
  void callerOutsideCallBlockIsUnboundIdentifier() {
    var error =
        assertThrows(
            TemplateRenderException.class, () -> Template.parse("{{ caller() }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
    assertEquals(
        "Cannot call something that is not a function: got UndefinedValue", error.getMessage());
  }

  @Test
  void reportsOriginalSourceLocationThroughDefaultPreprocessing() {
    // https://github.com/Alipsa/hfjinja/issues/27 — trim_blocks removes the newline after %}, so
    // the preprocessed position of `nope` drifts one character and one line from the original.
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{% set a = 1 %}\n{{ nope() }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
    assertEquals(new SourceLocation(19, 2, 4), error.location().orElseThrow());
  }

  @Test
  void callBlockBareBreak_isKnownDivergenceFromUpstream() {
    // See macroBareBreak_hasCorpusComparableSyntaxContract.
    var source =
        "{% macro f() %}[{{ caller() }}]{% endmacro %}" + "{% call f() %}{% break %}{% endcall %}";
    var error =
        assertThrows(TemplateRenderException.class, () -> Template.parse(source).render(Map.of()));
    var call = source.indexOf("{{ caller() }}") + 3;
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals("break or continue outside a for loop", error.getMessage());
    assertEquals(new SourceLocation(call, 1, call + 1), error.location().orElseThrow());
  }

  @Test
  void callBlockBareContinue_isKnownDivergenceFromUpstream() {
    var source =
        "{% macro f() %}[{{ caller() }}]{% endmacro %}"
            + "{% call f() %}{% continue %}{% endcall %}";
    var error =
        assertThrows(TemplateRenderException.class, () -> Template.parse(source).render(Map.of()));
    var call = source.indexOf("{{ caller() }}") + 3;
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertEquals("break or continue outside a for loop", error.getMessage());
    assertEquals(new SourceLocation(call, 1, call + 1), error.location().orElseThrow());
  }

  @Test
  void callBlockPushesKeywordArgumentsBagUnconditionallyForNonMacroCallees() {
    assertEquals("[]", Template.parse("{% call range(3) %}x{% endcall %}").render(Map.of()));
    assertEquals(
        "1[]2[]",
        Template.parse(
                "{% for i in [1,2] %}{{ i }}{% call range(2) %}{% break %}{% endcall %}{% endfor"
                    + " %}")
            .render(Map.of()));
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{% call namespace() %}x{% endcall %}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
  }

  @Test
  void callBlockHostFunctionDoesNotObserveExtraTrailingEmptyMapArgument() {
    // The unconditional keyword-arguments-bag push in evaluateCallStatement (needed to match the
    // oracle for range()/namespace()/macro callees) would otherwise leak into host functions as
    // an extra trailing empty map; HostFunctions.invoke strips it since host functions have no
    // such upstream contract and no template-visible way to construct one on their own.
    var seen = new java.util.ArrayList<java.util.List<Object>>();
    var options =
        se.alipsa.jmlx.jinja.RenderOptions.builder()
            .hostFunction(
                "record",
                arguments -> {
                  seen.add(arguments);
                  return "";
                })
            .build();
    Template.parse("{% call record(1) %}x{% endcall %}").render(Map.of(), options);
    Template.parse("{% call record() %}x{% endcall %}").render(Map.of(), options);
    assertEquals(2, seen.size());
    assertEquals(java.util.List.of(1L), seen.get(0));
    assertEquals(java.util.List.of(), seen.get(1));
  }

  @Test
  void deepMacroRecursionFailsWithResourceLimitNotStackOverflow_isKnownDivergenceFromUpstream() {
    // Which of the two guards fires first — RenderBudget's macroDepth counter or
    // Interpreter.render's StackOverflowError backstop — is inherently environment-dependent, and
    // this exact template proves it: verified locally between -Xss1024k (backstop fires,
    // "Maximum interpreter recursion depth exceeded") and -Xss2048k (macroDepth guard fires,
    // "Maximum macro call depth exceeded"), with no other AST nesting involved beyond the single
    // {% if %} per level the recursion needs to terminate. This is exactly what broke CI after the
    // backstop was added: this project's own dev machine happens to give test threads a stack
    // comfortably above that boundary, GitHub Actions' runners apparently do not, and an earlier
    // version of this test asserted the specific message, which is not a portable property of the
    // code. Upstream has no equivalent guard at all and instead produces a toolchain-dependent,
    // nonsensical error (or worse, a wrong answer) at its own unguarded stack limit — see this
    // plan's Known Gaps section. What both of Java's guards agree on, and what upstream cannot
    // match, is ErrorCategory.RESOURCE_LIMIT from a documented TemplateRenderException rather than
    // a raw escaped Error — that is the property this test pins.
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse(
                        "{% macro f(n) %}{% if n <= 0 %}done{% else %}{{ f(n-1) }}{% endif %}"
                            + "{% endmacro %}{{ f(5000) }}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
  }

  @Test
  void macroRecursionNestedInsideControlFlowFailsWithResourceLimitNotStackOverflow() {
    // maxMacroDepth bounds macro/call-block *invocation* count, not total interpreter recursion
    // depth. Nesting extra control-flow frames inside a recursive macro body (here, two nested
    // {% for %} loops per level) multiplies the JVM stack consumed per invocation, so a recursion
    // this deep (100000 levels) exhausts the native stack long before hitting maxMacroDepth's
    // default of 500 invocations — verified empirically to throw a raw StackOverflowError before
    // Interpreter.render's top-level catch was added. That catch is the backstop pinned here.
    //
    // maxMacroDepth is deliberately raised out of contention (to 1_000_000, far above what any
    // JVM stack tolerates for this body shape): at the default of 500, whichever of the two
    // guards fires first depends on the running thread's stack size, which is environment-
    // dependent, not a property of this code. Verified deterministic across -Xss512k..64m with
    // maxMacroDepth this high; macroDepthLimitIsExactlyEnforcedAtTheConfiguredBoundary is what
    // pins the counter itself.
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse(
                        "{% macro f(n) %}{% if n <= 0 %}done{% else %}{% for i in [1] %}{% for j in"
                            + " [1] %}{{ f(n-1) }}{% endfor %}{% endfor %}{% endif %}{% endmacro"
                            + " %}{{ f(100000) }}")
                    .render(Map.of(), RenderOptions.builder().maxMacroDepth(1_000_000).build()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
    assertEquals("Maximum interpreter recursion depth exceeded", error.getMessage());
  }

  @Test
  void macroRecursionThroughDefaultArgumentExpressionIsGuarded() {
    // A default-argument expression is evaluated before the macro's body — enterMacro must guard
    // the whole binding loop, not just evaluateBlock(n.body(), ...), or recursion hidden inside a
    // default argument bypasses maxMacroDepth entirely and overflows the native JVM stack instead
    // of failing with a clean RESOURCE_LIMIT error.
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{% macro m(a=m()) %}x{{ a }}{% endmacro %}{{ m() }}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
    assertEquals("Maximum macro call depth exceeded", error.getMessage());
  }

  @Test
  void nonRecursiveSequentialMacroCallsDoNotAccumulateDepth() {
    var builder = new StringBuilder("{% macro f() %}x{% endmacro %}");
    builder.append("{{ f() }}".repeat(600));
    assertEquals("x".repeat(600), Template.parse(builder.toString()).render(Map.of()));
  }

  @Test
  void macroDepthLimitIsExactlyEnforcedAtTheConfiguredBoundary() {
    // f(9) nests exactly 10 macro invocations (f(9), f(8), ..., f(0), each still active while
    // the next is evaluated). This pins the boundary itself: neither `>=` in place of `>` in
    // RenderBudget.enterMacro, nor an off-by-one in maxMacroDepth's threshold, could pass both
    // assertions below.
    var template =
        Template.parse(
            "{% macro f(n) %}{% if n <= 0 %}done{% else %}{{ f(n-1) }}{% endif %}{% endmacro %}"
                + "{{ f(9) }}");
    assertEquals(
        "done", template.render(Map.of(), RenderOptions.builder().maxMacroDepth(10).build()));
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> template.render(Map.of(), RenderOptions.builder().maxMacroDepth(9).build()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
    assertEquals("Maximum macro call depth exceeded", error.getMessage());
  }

  @Test
  void defaultMacroDepthLimitAllowsOrdinaryNestedRecursion() {
    // Pins a floor on RenderOptions.DEFAULT's maxMacroDepth (500) without hardcoding the exact
    // value: nested macro recursion is an ordinary template pattern, and a regression collapsing
    // the effective default down to a handful of levels must fail this, even though the existing
    // f(5000)-scale test above cannot (any limit under 5000 also fails that one).
    assertEquals(
        "done",
        Template.parse(
                "{% macro f(n) %}{% if n <= 0 %}done{% else %}{{ f(n-1) }}{% endif %}"
                    + "{% endmacro %}{{ f(99) }}")
            .render(Map.of()));
  }

  @Test
  void filterBlockRendersBodyThenAppliesNamedFilter() {
    assertEquals("HI", Template.parse("{% filter upper %}hi{% endfilter %}").render(Map.of()));
    assertEquals(
        "1-2",
        Template.parse(
                "{% filter join('-') %}{% for i in [1,2] %}{{ i }}{% endfor %}{% endfilter %}")
            .render(Map.of()));
  }

  @Test
  void filterBlockPropagatesBreakToEnclosingLoop() {
    assertEquals(
        "",
        Template.parse(
                "{% for i in [1,2,3] %}{{ i }}{% filter upper %}{% break %}{% endfilter %}{% endfor"
                    + " %}")
            .render(Map.of()));
  }

  @Test
  void filterBlockChargesOutputForBodyAndFilteredResultSeparately() {
    // maxOutputLength bounds cumulative rendered characters, not final output size: the 6-char
    // body is charged once inside evaluateBlock, then the 6-char filtered result is charged
    // again, so a limit of 10 is exceeded even though the visible output is only 6 characters.
    // Matches evaluateSet's pre-existing {% set x %}...{% endset %} block-capture behavior — not
    // a regression introduced by filter/call blocks.
    var options = se.alipsa.jmlx.jinja.RenderOptions.builder().maxOutputLength(10).build();
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{% filter upper %}abcdef{% endfilter %}")
                    .render(Map.of(), options));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
  }

  @Test
  void callBlockChargesOutputForBodyAndCalleeResultSeparately() {
    var options = se.alipsa.jmlx.jinja.RenderOptions.builder().maxOutputLength(10).build();
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse(
                        "{% macro wrap() %}<{{ caller() }}>{% endmacro %}"
                            + "{% call wrap() %}abcdef{% endcall %}")
                    .render(Map.of(), options));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, error.category());
  }

  @Test
  void macroCannotBeUsedAsAFilter() {
    // Filters are resolved from a fixed table, never from the variable/macro namespace: `f` is
    // not found as a filter even though it is a perfectly callable macro.
    var error =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{% macro f(x) %}{{ x|upper }}{% endmacro %}{{ 'hi' | f }}")
                    .render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
    assertEquals("Unknown StringValue filter: f", error.getMessage());
    assertEquals(
        "Unknown ArrayValue filter: f",
        assertThrows(
                TemplateRenderException.class,
                () ->
                    Template.parse("{% macro f(x) %}{{ x|upper }}{% endmacro %}{{ [1] | f }}")
                        .render(Map.of()))
            .getMessage());
  }

  @Test
  void matchesPinnedCallFormFilterDispatch() {
    assertAll(
        () ->
            assertEquals(
                "Unknown StringValue filter: safe",
                filterError("{{ 'abc' | safe(raise_exception('sentinel')) }}").getMessage()),
        () ->
            assertEquals(
                "Cannot unpack non-iterable type: StringValue",
                filterError("{{ {'a': 1} | items(*'ab') }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ArrayValue filter: first",
                filterError("{{ [1, 2] | first() }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ArrayValue filter: first",
                filterError("{{ (1, 2) | first(1) }}").getMessage()),
        () ->
            assertEquals(
                "1-2|12",
                Template.parse("{{ (1, 2) | join('-') }}|{{ (1, 2) | join() }}").render(Map.of())),
        () -> assertEquals("[]", Template.parse("{{ [] | selectattr() }}").render(Map.of())),
        () ->
            assertEquals(
                "`selectattr` can only be applied to array of objects",
                filterError("{{ [1, 2] | selectattr() }}").getMessage()),
        () ->
            assertEquals(
                "`selectattr` filter requires at least one argument",
                filterError("{{ [{'a': 1}] | selectattr() }}").getMessage()),
        () ->
            assertEquals(
                ErrorCategory.ARITY, filterError("{{ [{'a': 1}] | selectattr() }}").category()),
        () ->
            assertEquals(
                "Cannot apply filter \"abs\" to type: FloatValue",
                filterError("{{ 1.5 | abs() }}").getMessage()),
        () ->
            assertEquals(
                "2.9|1.5|1",
                Template.parse("{{ 2.9 | int() }}|{{ 1.5 | int(0) }}|{{ 1 | float() }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "Cannot apply filter \"join\" to type: IntegerValue",
                filterError("{{ 1 | join() }}").getMessage()),
        () ->
            assertEquals(
                "Cannot apply filter \"frob\" to type: NullValue",
                filterError("{{ none | frob }}").getMessage()));
  }

  @Test
  void matchesPinnedBareUnknownFilterDiagnostics() {
    assertAll(
        () ->
            assertEquals(
                "Unknown NumericValue filter: frob", filterError("{{ 1 | frob }}").getMessage()),
        () ->
            assertEquals(
                "Unknown NumericValue filter: frob", filterError("{{ 1.5 | frob }}").getMessage()),
        () ->
            assertEquals(
                "Unknown BooleanValue filter: frob", filterError("{{ true | frob }}").getMessage()),
        () ->
            assertEquals(
                "Cannot apply filter \"frob\" to type: UndefinedValue",
                filterError("{{ nope | frob }}").getMessage()),
        () ->
            assertEquals(
                "Cannot apply filter \"frob\" to type: FunctionValue",
                filterError("{{ range | frob }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ArrayValue filter: frob", filterError("{{ (1, 2) | frob }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ObjectValue filter: frob",
                filterError("{{ {'a': 1} | frob }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ObjectValue filter: length",
                filterError("{{ {'a': 1} | length() }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ArrayValue filter: selectattr",
                filterError("{{ [1] | selectattr }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ArrayValue filter: rejectattr",
                filterError("{{ [1] | rejectattr }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ArrayValue filter: map", filterError("{{ [1] | map }}").getMessage()),
        () ->
            assertEquals(
                "Unknown StringValue filter: replace",
                filterError("{{ 'abc' | replace }}").getMessage()),
        () ->
            assertEquals(
                "`map` expressions without `attribute` set are not currently supported.",
                filterError("{{ [1, 2] | map() }}").getMessage()),
        () ->
            assertEquals(
                "`map` expressions without `attribute` set are not currently supported.",
                filterError("{{ [1, 2] | map('x') }}").getMessage()),
        () ->
            assertEquals(
                "Unknown ArrayValue filter: default",
                filterError("{{ [1] | default }}").getMessage()),
        () ->
            assertEquals(
                "Cannot apply filter \"first\" to type: StringValue",
                filterError("{{ 'abc' | first }}").getMessage()));
  }

  @Test
  void supportsMistralSelectattrFiltersAndSharedTests() {
    assertEquals(
        "a|b|",
        Template.parse(
                "{% set xs = [{'x': 'a'}, {'x': 1}, {'x': 'b'}] %}"
                    + "{% for x in xs|selectattr('x', 'string') %}{{ x.x }}|{% endfor %}")
            .render(Map.of()));
    assertEquals(
        "2",
        Template.parse("{% set xs = ({'x': 'a'}, {'x': 'b'}) %}{{ xs|selectattr('x')|length }}")
            .render(Map.of()));
    assertEquals(
        "true",
        Template.parse(
                "{{ [{'n': '1'}, {'n': 1}, {'n': true}]|selectattr('n', 'equalto', '1')|length == 1"
                    + " }}")
            .render(Map.of()));
  }

  @Test
  void wp7Slice3FilterInvocationParity() {
    assertAll(
        () ->
            assertEquals(
                "[{\"a\": \"x\"}]",
                Template.parse("{{ [{'a':'x'}]|selectattr('a','string','x','ignored') }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "arguments of `selectattr` must be strings",
                raisedMessage("{{ [{'a':1}]|selectattr('a'~'') }}", Map.of())),
        () ->
            assertEquals(
                "replace() arguments must be strings",
                raisedMessage("{{ 'abc'.replace('a',count=1) }}", Map.of())),
        () ->
            assertEquals(
                "ZZa", Template.parse("{{ 'aaa'|replace('a','Z',count=2) }}").render(Map.of())),
        () ->
            assertEquals(
                "Cannot convert to JSON: KeywordArgumentsValue",
                raisedMessage("{{ {'a':1}|get('z',nope=1) }}", Map.of())),
        () ->
            assertEquals(
                "replace() arguments must be strings",
                raisedMessage("{{ 'ab'|replace('a') }}", Map.of())),
        () ->
            assertEquals(
                "Cannot convert to JSON: KeywordArgumentsValue",
                raisedMessage("{{ {'a':1}.get('z',default='d') }}", Map.of())),
        () ->
            assertEquals(
                "maxsplit argument must be a number",
                raisedMessage("{{ 'a b c'.split(' ',maxsplit=1) }}", Map.of())),
        () ->
            assertEquals(
                "sep argument must be a string or null",
                raisedMessage("{{ 'a b'.split(sep=' ') }}", Map.of())),
        () ->
            assertEquals(
                "Object key must be a string: got KeywordArgumentsValue",
                raisedMessage("{{ {'a':1}.get(foo=1) }}", Map.of())),
        () ->
            assertEquals(
                "Positional arguments must come before keyword arguments",
                raisedMessage("{% macro m(a=1) %}{{ a }}{% endmacro %}{{ m(a=1,2) }}", Map.of())),
        () ->
            assertEquals(
                "replace() requires at least two arguments",
                raisedMessage("{{ 'abc'|replace() }}", Map.of())),
        () ->
            assertEquals(
                "replace() requires at least two arguments",
                raisedMessage("{{ 'ab'|replace(old='a',new='x') }}", Map.of())),
        () ->
            assertEquals(
                "Zbc", Template.parse("{{ 'abc'|replace('a','Z',1,99) }}").render(Map.of())));
  }

  @Test
  void wp7CorpusSelectattrAstAndFilterBlockCases() {
    assertAll(
        () ->
            assertEquals(
                "arguments of `selectattr` must be strings",
                raisedMessage("{{ [{'a':1}]|selectattr(*['a']) }}", Map.of())),
        () ->
            assertEquals(
                "arguments of `selectattr` must be strings",
                raisedMessage("{{ [{'a':1}]|selectattr(a='a') }}", Map.of())),
        () ->
            assertEquals(
                "1", Template.parse("{% filter int(1,2) %}x{% endfilter %}").render(Map.of())),
        () ->
            assertEquals(
                "replace() arguments must be strings",
                raisedMessage("{% filter replace('a') %}abc{% endfilter %}", Map.of())),
        () ->
            assertEquals(
                "wp7-eager-sentinel",
                raisedMessage(
                    "{% filter int(1,raise_exception('wp7-eager-sentinel')) %}x{% endfilter %}",
                    Map.of())));
  }

  @Test
  void wp7CorpusGetFilterWithoutKeywords() {
    assertEquals("", Template.parse("{{ {'a':1}|get('z') }}").render(Map.of()));
    assertEquals("1", Template.parse("{{ {'a':1}|get('a') }}").render(Map.of()));
  }

  @Test
  void wp7CorpusEagerSelectedFilterArguments() {
    assertAll(
        () -> assertEagerSentinel("{{ 'x'|int(1,raise_exception('wp7-eager-sentinel')) }}"),
        () -> assertEagerSentinel("{{ 'x'|float(1,raise_exception('wp7-eager-sentinel')) }}"),
        () ->
            assertEagerSentinel(
                "{{ [2,1]|sort(false,false,none,raise_exception('wp7-eager-sentinel')) }}"),
        () -> assertEagerSentinel("{{ [1,2]|join('-',raise_exception('wp7-eager-sentinel')) }}"),
        () ->
            assertEagerSentinel(
                "{{ 'x'|indent(2,false,false,raise_exception('wp7-eager-sentinel')) }}"),
        () ->
            assertEagerSentinel(
                "{{ [{'a':1}]|map(attribute='a',unused=raise_exception('wp7-eager-sentinel')) }}"),
        () -> assertEagerSentinel("{{ [1]|tojson(raise_exception('wp7-eager-sentinel')) }}"),
        () ->
            assertEagerSentinel(
                "{{ missing|default('x',false,raise_exception('wp7-eager-sentinel')) }}"),
        () ->
            assertEagerSentinel(
                "{{ 'abc'|replace('a','z',1,raise_exception('wp7-eager-sentinel')) }}"));
  }

  private static void assertEagerSentinel(String source) {
    assertEquals("wp7-eager-sentinel", raisedMessage(source, Map.of()));
  }

  @Test
  void supportsMistralListStringAndObjectItems() {
    assertEquals(
        "[1, 2]|true",
        Template.parse("{{ [1, 2]|list }}|{{ (1, 2)|list is sequence }}").render(Map.of()));
    assertEquals(
        "a=1,b=2,",
        Template.parse(
                "{% for key, value in {'a': 1, 'b': 2}.items() %}{{ key }}={{ value }},{% endfor"
                    + " %}")
            .render(Map.of()));
    var error =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ {'a': 1}|string }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, error.category());
    var tupleError =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ (1, 2)|string }}").render(Map.of()));
    assertEquals("Cannot convert to JSON: TupleValue", tupleError.getMessage());
    assertEquals(
        "true", Template.parse("{% set o = {'a': 1} %}{{ o.items == o.items }}").render(Map.of()));
  }

  @Test
  void rendersRetainedModelTemplateResources() throws Exception {
    var mistral = resource("mistral-7b-instruct-v0.3.jinja");
    assertEquals(
        "<s>[INST] Hello![/INST] Hi!</s>",
        Template.parse(mistral)
            .render(
                Map.of(
                    "bos_token",
                    "<s>",
                    "eos_token",
                    "</s>",
                    "messages",
                    java.util.List.of(
                        Map.of("role", "user", "content", "Hello!"),
                        Map.of("role", "assistant", "content", "Hi!")))));
    var function =
        orderedMap(
            "name",
            "get_weather",
            "description",
            "Get weather",
            "parameters",
            orderedMap(
                "type",
                "object",
                "properties",
                orderedMap("city", orderedMap("type", "string")),
                "required",
                java.util.List.of("city")),
            "return",
            orderedMap("type", "string"));
    assertEquals(
        "<s>[INST] What is the weather?[/INST][TOOL_CALLS] [{\"name\": \"get_weather\","
            + " \"arguments\": {\"city\": \"Paris\"}, \"id\": \"abcdefghi\"}]</s>[TOOL_RESULTS]"
            + " {\"content\": sunny, \"call_id\": \"abcdefghi\"}[/TOOL_RESULTS] It is"
            + " sunny.</s>[AVAILABLE_TOOLS] [{\"type\": \"function\", \"function\": {\"name\":"
            + " \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\":"
            + " \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\":"
            + " [\"city\"]}}}][/AVAILABLE_TOOLS][INST] What is the weather?[/INST]",
        Template.parse(mistral)
            .render(
                Map.of(
                    "bos_token",
                    "<s>",
                    "eos_token",
                    "</s>",
                    "tools",
                    java.util.List.of(orderedMap("type", "function", "function", function)),
                    "messages",
                    java.util.List.of(
                        orderedMap("role", "user", "content", "What is the weather?"),
                        orderedMap(
                            "role",
                            "assistant",
                            "tool_calls",
                            java.util.List.of(
                                orderedMap(
                                    "id",
                                    "abcdefghi",
                                    "function",
                                    orderedMap(
                                        "name",
                                        "get_weather",
                                        "arguments",
                                        orderedMap("city", "Paris"))))),
                        orderedMap("role", "tool", "content", "sunny", "tool_call_id", "abcdefghi"),
                        orderedMap("role", "assistant", "content", "It is sunny."),
                        orderedMap("role", "user", "content", "What is the weather?")))));
    var qwen = resource("qwen2.5-32b-instruct.jinja");
    assertEquals(
        "<|im_start|>system\n"
            + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n"
            + "Hello!<|im_end|>\n"
            + "<|im_start|>assistant\n",
        Template.parse(qwen)
            .render(
                Map.of(
                    "add_generation_prompt",
                    true,
                    "messages",
                    java.util.List.of(Map.of("role", "user", "content", "Hello!")))));
  }

  @Test
  void rendersStep3MacroHeavyTemplate() throws Exception {
    var tool =
        orderedMap(
            "type",
            "function",
            "function",
            orderedMap(
                "name",
                "get_weather",
                "description",
                "Météo",
                "parameters",
                orderedMap(
                    "type",
                    "object",
                    "properties",
                    orderedMap("city", orderedMap("type", "string")),
                    "required",
                    java.util.List.of("city"))));
    var context =
        orderedMap(
            "bos_token",
            "<s>",
            "tools",
            java.util.List.of(tool),
            "messages",
            java.util.List.of(
                orderedMap("role", "system", "content", "You are helpful."),
                orderedMap("role", "tool_description", "content", "Use tools."),
                orderedMap(
                    "role",
                    "user",
                    "content",
                    java.util.List.of(
                        orderedMap("type", "text", "text", "What is the weather?"),
                        orderedMap("type", "image"))),
                orderedMap(
                    "role",
                    "assistant",
                    "content",
                    "Checking",
                    "tool_calls",
                    java.util.List.of(
                        orderedMap(
                            "type",
                            "function",
                            "function",
                            orderedMap(
                                "name",
                                "get_weather",
                                "arguments",
                                orderedMap("city", "Paris", "unit", "C"))))),
                orderedMap(
                    "role",
                    "tool_response",
                    "content",
                    java.util.List.of(orderedMap("text", "sunny")))));
    assertEquals(
        resource("step3-tooluse.expected.txt"),
        Template.parse(resource("step3.jinja")).render(context));
  }

  @Test
  void rendersPrimaryQwen38MlxFixtureWithPinnedGoldens() throws Exception {
    var template = resource("qwen3.8-27b-4bit.jinja");
    assertEquals(8952, template.getBytes(StandardCharsets.UTF_8).length);
    assertEquals(
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", sha256(template));

    assertEquals(
        resource("qwen3.8-normal.expected.txt"),
        Template.parse(template)
            .render(
                Map.of(
                    "add_generation_prompt",
                    true,
                    "enable_thinking",
                    false,
                    "reasoning_effort",
                    "medium",
                    "messages",
                    java.util.List.of(Map.of("role", "user", "content", "Hello!")))));
    assertEquals(
        resource("qwen3.8-vision.expected.txt"),
        Template.parse(template)
            .render(
                Map.of(
                    "add_generation_prompt", false,
                    "add_vision_id", true,
                    "enable_thinking", false,
                    "messages",
                        java.util.List.of(
                            Map.of(
                                "role",
                                "user",
                                "content",
                                java.util.List.of(
                                    Map.of("type", "text", "text", "Describe "),
                                    Map.of("type", "image"),
                                    Map.of("type", "text", "text", " then "),
                                    Map.of("type", "video")))))));
    var tool =
        orderedMap(
            "type",
            "function",
            "function",
            orderedMap(
                "name",
                "get_weather",
                "description",
                "Get weather",
                "parameters",
                orderedMap(
                    "type",
                    "object",
                    "properties",
                    orderedMap("city", orderedMap("type", "string")),
                    "required",
                    java.util.List.of("city"))));
    assertEquals(
        resource("qwen3.8-tooluse.expected.txt"),
        Template.parse(template)
            .render(
                orderedMap(
                    "add_generation_prompt",
                    true,
                    "enable_thinking",
                    true,
                    "reasoning_effort",
                    "low",
                    "tools",
                    java.util.List.of(tool),
                    "messages",
                    java.util.List.of(
                        orderedMap("role", "user", "content", "What is the weather?"),
                        orderedMap(
                            "role",
                            "assistant",
                            "content",
                            "Checking",
                            "tool_calls",
                            java.util.List.of(
                                orderedMap(
                                    "function",
                                    orderedMap(
                                        "name",
                                        "get_weather",
                                        "arguments",
                                        orderedMap(
                                            "city",
                                            "Paris",
                                            "units",
                                            java.util.List.of("metric", "celsius")))))),
                        orderedMap("role", "tool", "content", "Sunny")))));
  }

  @Test
  void supportsQwen38StringBoundariesAndSafeFilter() {
    assertEquals(
        "truetruetruetrue",
        Template.parse(
                "{{ 'abc'.startswith('a') }}{{ 'abc'.endswith('c') }}"
                    + "{{ 'abc'.startswith(('a', 'x')) }}{{ 'abc'.endswith(['x', 'c']) }}")
            .render(Map.of()));
    assertEquals("[1, 2]", Template.parse("{{ [1, 2] | safe }}").render(Map.of()));
    var missing =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'abc'.startswith() }}").render(Map.of()));
    assertEquals(ErrorCategory.ARITY, missing.category());
    assertEquals("startswith() requires at least one argument", missing.getMessage());
    var argument =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'abc'.endswith(1) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, argument.category());
    assertEquals("endswith() argument must be a string or tuple of strings", argument.getMessage());
    var tuple =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'abc'.startswith(('x', 1)) }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, tuple.category());
    assertEquals("startswith() tuple elements must be strings", tuple.getMessage());
    assertEquals(
        "Unknown StringValue filter: safe",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ 'abc' | safe(1) }}").render(Map.of()))
            .getMessage());
    assertEquals("", Template.parse("{{ 'abc'['xy'[9]] }}").render(Map.of()));
    assertEquals("3", Template.parse("{{ 'abc'.length }}").render(Map.of()));
  }

  @Test
  void validatesItemsFilterArgumentsAndReceiver() {
    assertEquals("[[\"a\", 1]]", Template.parse("{{ {'a': 1} | items }}").render(Map.of()));
    assertEquals("[[\"a\", 1]]", Template.parse("{{ {'a': 1} | items(1) }}").render(Map.of()));
    var receiver =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ [1] | items }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, receiver.category());
  }

  @Test
  void supportsRemainingPinnedRuntimeFiltersTestsAndMembers() {
    var source =
        "{{ [3,1,2]|first }}|{{ [3,1,2]|last }}|{{ [3,1,2]|reverse }}|"
            + "{{ [3,1,2]|unique }}|{{ [3,1,2]|sort }}|{{ xs|map(attribute='n', default=0) }}|"
            + "{{ 'hello world'|title }}|{{ 'hello'|capitalize }}|{{ 'a\\nb'|indent(2) }}|"
            + "{{ 'abab'|replace('a','x',1) }}|{{ -2|abs }}|{{ true|bool }}|"
            + "{{ ' a b '.split() }}|{{ 'a-b-c'.replace('-','/',1) }}|{{ o.get('x','d') }}|"
            + "{{ o.keys() }}|{{ o.values() }}|{{ o.dictsort() }}|{{ 3 is odd }}{{ 4 is even }}"
            + "{{ 4 is integer }}{{ 'ABC' is upper }}{{ 'abc' is lower }}{{ range is callable }}";
    assertEquals(
        "3|2|[2, 1, 3]|[3, 1, 2]|[1, 2, 3]|[2, 0]|Hello World|Hello|a\n"
            + "  b|xbab|2|true|[\"a\", \"b\"]|a/b-c|d|[\"b\", \"a\"]|[2, 1]|[[\"a\", 1], [\"b\","
            + " 2]]|truetruetruetruetruetrue",
        Template.parse(source)
            .render(
                Map.of(
                    "xs", java.util.List.of(Map.of("n", 2), Map.of()),
                    "o", orderedMap("b", 2, "a", 1))));
  }

  @Test
  void supportsObjectBuiltinFilterFormsAndValueSort() {
    assertEquals(
        "[\"b\", \"a\"]|[2, 1]|[[\"a\", 1], [\"b\", 2]]|d",
        Template.parse(
                "{% set o = {'b': 2, 'a': 1} %}{{ o|keys }}|{{ o|values }}|{{ o|dictsort }}|{{"
                    + " o|get('x','d') }}")
            .render(Map.of()));
    assertEquals(
        "[[\"b\", 1], [\"a\", 2]]",
        Template.parse("{{ {'a': 2, 'b': 1}|dictsort(by='value') }}").render(Map.of()));
  }

  @Test
  void supportsSequenceAndStringFilterOptions() {
    assertAll(
        () -> assertEquals("a\n    b", Template.parse("{{ 'a\\nb'|indent }}").render(Map.of())),
        () -> assertEquals("2.5", Template.parse("{{ (0.0 - 2.5)|abs }}").render(Map.of())),
        () ->
            assertEquals(
                "[[2, 0], [3, 1]]",
                Template.parse("{{ [[3,1], [2,0]]|sort(attribute='0') }}").render(Map.of())),
        () ->
            assertEquals(
                "[3, 4]",
                Template.parse("{{ [{'a': [2,3]}, {'a': [1,4]}]|map(attribute='a.1') }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "[\"a\", \"b-c\"]", Template.parse("{{ 'a-b-c'.split('-', 1) }}").render(Map.of())),
        () ->
            assertEquals(
                "[1, 2]", Template.parse("{{ [2,1]|sort(attribute=none) }}").render(Map.of())),
        () ->
            assertEquals("[null, null]", Template.parse("{{ [none,none]|sort }}").render(Map.of())),
        () ->
            assertEquals(
                "[\"a b \"]", Template.parse("{{ ' a b '.split(none, 0) }}").render(Map.of())),
        () -> assertEquals("A_b 3d", Template.parse("{{ 'a_b 3d'|title }}").render(Map.of())),
        () ->
            assertEquals(
                "a b|Hello World|Hello",
                Template.parse(
                        "{{ ' a b '.strip() }}|{{ 'hello world'.title() }}|{{ 'hello'.capitalize()"
                            + " }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "xbab|xbab",
                Template.parse(
                        "{{ 'abab'|replace('a','x',count=1) }}|{{ 'abab'.replace('a','x',count=1)"
                            + " }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "falsefalsefalsefalsefalsefalse",
                Template.parse(
                        "{{ 3 is even }}{{ 4 is odd }}{{ 'abc' is upper }}{{ 'ABC' is lower }}"
                            + "{{ 1.0 is integer }}{{ 1 is callable }}")
                    .render(Map.of())));
    var mapReceiver =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ [{'a': 1}, 2]|map(attribute='a') }}").render(Map.of()));
    assertEquals(ErrorCategory.TYPE, mapReceiver.category());
    assertEquals(
        "[[2, 0], [3, 1]]",
        Template.parse("{{ [[3,1], [2,0]]|sort(attribute=0) }}").render(Map.of()));
    assertAll(
        () ->
            assertEquals(
                ErrorCategory.TYPE,
                assertThrows(
                        TemplateRenderException.class,
                        () -> Template.parse("{{ 5|keys }}").render(Map.of()))
                    .category()),
        () ->
            assertEquals(
                ErrorCategory.TYPE,
                assertThrows(
                        TemplateRenderException.class,
                        () -> Template.parse("{{ [1]|sort(attribute=true) }}").render(Map.of()))
                    .category()),
        () ->
            assertEquals(
                ErrorCategory.TYPE,
                assertThrows(
                        TemplateRenderException.class,
                        () -> Template.parse("{{ 'a'|replace('a','b','c') }}").render(Map.of()))
                    .category()));
  }

  @Test
  void matchesUpstreamUndefinedBackedFailures() {
    assertAll(
        () ->
            assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ []|first }}|{{ []|last }}").render(Map.of())),
        () ->
            assertThrows(
                TemplateRenderException.class,
                () ->
                    Template.parse("{% set u = {('ab'[9]): 1, 'z': 2} %}{{ u.dictsort() }}")
                        .render(Map.of())),
        () ->
            assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ 'ab'[9] is lower }}").render(Map.of())));
  }

  @Test
  void coversRemainingRuntimeArgumentPaths() {
    assertAll(
        () ->
            assertEquals(
                "[[\"b\", 2], [\"a\", 1]]",
                Template.parse("{{ {'b': 2, 'a': 1}.dictsort(true, 'value', true) }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "[1, undefined]",
                Template.parse("{{ [{'a':1}, {}]|map(attribute='a') }}").render(Map.of())),
        () ->
            assertEquals(
                "[undefined, undefined]",
                Template.parse("{{ [{'a':[1]}, {'a':[2]}]|map(attribute='a.x') }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "[undefined, undefined]",
                Template.parse("{{ [{},{}]|map(attribute='x')|sort }}").render(Map.of())),
        () ->
            assertEquals(
                "3|2|[3, 1, 2]|[2, 1, 3]",
                Template.parse(
                        "{{ (3,1,2)|first }}|{{ (3,1,2)|last }}|{{ (3,1,2)|unique }}|{{"
                            + " (3,1,2)|reverse }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "[\"a\", \"b,c\"]",
                Template.parse("{{ 'a,b,c'.split(',', -2) }}").render(Map.of())),
        () ->
            assertEquals(
                "[\"a b c \"]",
                Template.parse("{{ ' a b c '.split(none, -2) }}").render(Map.of())));
    assertAll(
        () ->
            assertEquals(
                ErrorCategory.TYPE,
                assertThrows(
                        TemplateRenderException.class,
                        () -> Template.parse("{{ [1]|first(1) }}").render(Map.of()))
                    .category()),
        () ->
            assertEquals(
                ErrorCategory.TYPE,
                assertThrows(
                        TemplateRenderException.class,
                        () -> Template.parse("{{ [1]|last(1) }}").render(Map.of()))
                    .category()),
        () ->
            assertEquals(
                ErrorCategory.TYPE,
                assertThrows(
                        TemplateRenderException.class,
                        () -> Template.parse("{{ [1]|unique(1) }}").render(Map.of()))
                    .category()),
        () ->
            assertEquals(
                ErrorCategory.TYPE,
                assertThrows(
                        TemplateRenderException.class,
                        () -> Template.parse("{{ {'a': 1}|get(key='a') }}").render(Map.of()))
                    .category()));
  }

  @Test
  void matchesPinnedRuntimeEdgesForNewlyPortedFeatures() {
    assertAll(
        () ->
            assertEquals(
                "[0, 1, true]", Template.parse("{{ [1, true, 0] | sort }}").render(Map.of())),
        () ->
            assertEquals(
                "éLan Vital", Template.parse("{{ 'élan vital' | title }}").render(Map.of())),
        () ->
            assertEquals("-😀-", Template.parse("{{ '😀' | replace('', '-') }}").render(Map.of())),
        () ->
            assertEquals("[\"a\", \"b\"]", Template.parse("{{ 'a b'.split() }}").render(Map.of())),
        () -> assertEquals("a", Template.parse("{{ 'a  '.rstrip() }}").render(Map.of())),
        () -> assertEquals("a", Template.parse("{{ '  a'.lstrip() }}").render(Map.of())),
        () ->
            assertEquals(
                "[\"a\", \"b \"]", Template.parse("{{ ' a b '.split(none, 1) }}").render(Map.of())),
        () ->
            assertEquals(
                "[, \"z\"]|2",
                Template.parse(
                        "{% set u = {('ab'[9]): 1, 'z': 2} %}{{ u.keys() }}|{{ u|items|length }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "[[, 1]]",
                Template.parse("{% set u = {('ab'[9]): 1} %}{{ u.dictsort() }}").render(Map.of())),
        () -> assertEquals("2", Template.parse("{{ [1,2].length }}").render(Map.of())),
        () ->
            assertEquals(
                "  a\n  b", Template.parse("{{ 'a\\nb'|indent(2, []) }}").render(Map.of())),
        () ->
            assertEquals(
                "  a\n  b", Template.parse("{{ 'a\\nb'|indent(2, {}) }}").render(Map.of())),
        () ->
            assertEquals(
                "a\n  b\n  \n  c",
                Template.parse("{{ 'a\\nb\\n\\nc'|indent(2, false, []) }}").render(Map.of())),
        () ->
            assertEquals(
                "[\"a\", \"b\"]", Template.parse("{{ 'a\uFEFFb'.split() }}").render(Map.of())),
        () ->
            assertEquals(
                "[\"a\u0085b\"]", Template.parse("{{ 'a\u0085b'.split() }}").render(Map.of())),
        () ->
            assertEquals(
                "[\"b\", \"a\", \"A\"]",
                Template.parse("{{ ['b', 'A', 'a']|sort(case_sensitive=true, reverse=true) }}")
                    .render(Map.of())),
        () ->
            assertEquals(
                "  a\n  b", Template.parse("{{ 'a\\nb'|indent(2, true) }}").render(Map.of())),
        () ->
            assertEquals(
                "a\n  b\n  \n  c",
                Template.parse("{{ 'a\\nb\\n\\nc'|indent(2, false, true) }}").render(Map.of())));
    var negative =
        assertThrows(
            TemplateRenderException.class,
            () -> Template.parse("{{ 'a\\nb' | indent(-1) }}").render(Map.of()));
    assertEquals(ErrorCategory.VALUE, negative.category());
    var bounded =
        assertThrows(
            TemplateRenderException.class,
            () ->
                Template.parse("{{ 'a\\nb' | indent(400000000) }}")
                    .render(Map.of(), RenderOptions.builder().maxOutputLength(100).build()));
    assertEquals(ErrorCategory.RESOURCE_LIMIT, bounded.category());
    assertEquals(
        "replace() requires at least two arguments",
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ 'ab' | replace(old='a', new='x') }}").render(Map.of()))
            .getMessage());
    assertEquals(
        ErrorCategory.TYPE,
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ 'x'|abs }}").render(Map.of()))
            .category());
    assertEquals(
        ErrorCategory.TYPE,
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ 1|bool }}").render(Map.of()))
            .category());
    assertEquals(
        ErrorCategory.TYPE,
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ [1, 'x']|sort }}").render(Map.of()))
            .category());
    assertEquals(
        ErrorCategory.TYPE,
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ 'a'.split('') }}").render(Map.of()))
            .category());
    assertEquals(
        ErrorCategory.TYPE,
        assertThrows(
                TemplateRenderException.class,
                () -> Template.parse("{{ {}.dictsort(by='other') }}").render(Map.of()))
            .category());
  }

  @Test
  void evaluateExpressionAssertsUnreachableForParserOnlyExpressionShapes() {
    // Handing one of these three node types directly to evaluateExpression, as this test does,
    // is the only way to reach their arms at all -- see the comment above
    // Interpreter.evaluateExpression's three matching cases for why the parser itself never can.
    var location = new SourceLocation(0, 1, 1);
    var env = new Environment(null);
    var budget = new RenderBudget(RenderOptions.builder().build());
    var slice = new Expression.SliceExpression(null, null, null, location);
    var keywordArgument =
        new Expression.KeywordArgumentExpression(
            new Expression.Identifier("k", location),
            new Expression.IntegerLiteral(1, location),
            location);
    var spread =
        new Expression.SpreadExpression(new Expression.Identifier("x", location), location);
    assertAll(
        () ->
            assertEquals(
                "unreachable: SliceExpression at " + location,
                assertThrows(
                        AssertionError.class,
                        () -> Interpreter.evaluateExpression(slice, env, budget))
                    .getMessage()),
        () ->
            assertEquals(
                "unreachable: KeywordArgumentExpression at " + location,
                assertThrows(
                        AssertionError.class,
                        () -> Interpreter.evaluateExpression(keywordArgument, env, budget))
                    .getMessage()),
        () ->
            assertEquals(
                "unreachable: SpreadExpression at " + location,
                assertThrows(
                        AssertionError.class,
                        () -> Interpreter.evaluateExpression(spread, env, budget))
                    .getMessage()));
  }

  private String resource(String name) throws Exception {
    try (InputStream input = getClass().getResourceAsStream("/model-templates/" + name)) {
      if (input == null) throw new AssertionError("Missing model template resource: " + name);
      return new String(input.readAllBytes(), StandardCharsets.UTF_8);
    }
  }

  private static TemplateRenderException filterError(String template) {
    return assertThrows(
        TemplateRenderException.class, () -> Template.parse(template).render(Map.of()));
  }

  private static String sha256(String text) throws Exception {
    var digest = MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
    var hex = new StringBuilder();
    for (var value : digest) hex.append(String.format("%02x", value));
    return hex.toString();
  }

  private static Map<String, Object> orderedMap(Object... entries) {
    var result = new LinkedHashMap<String, Object>();
    for (int i = 0; i < entries.length; i += 2) result.put((String) entries[i], entries[i + 1]);
    return result;
  }
}
