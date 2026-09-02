package se.alipsa.jmlx.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TemplateFormatTest {
  @Test
  void formatsPinnedNodeGoldenCases() {
    var template = Template.parse("{% if a %}{{ x }}{% endif %}");
    assertEquals("{%- if a -%}\n\t{{- x -}}\n{%- endif -%}", template.format());
    assertEquals("{%- if a -%}\n{{- x -}}\n{%- endif -%}", template.format(-0.5));
    assertEquals("{%- if a -%}\n  {{- x -}}\n{%- endif -%}", template.format(2.7));
    assertEquals("{#  a  #}", Template.parse("{# a #}").format());
    assertEquals(
        "{{- 1 -}}\n{{- \"|\" -}}\n{{- 2.5 -}}", Template.parse("{{ 1.0 }}|{{ 2.50 }}").format());
    assertEquals(
        "{%- macro f(x) -%}\n"
            + "  {{- x | upper -}}\n"
            + "{%- endmacro -%}\n"
            + "{%- call f(\"a\") -%}\n"
            + "  {{- \"x\" -}}\n"
            + "{%- endcall -%}",
        Template.parse("{% macro f(x) %}{{ x|upper }}{% endmacro %}{% call f('a') %}x{% endcall %}")
            .format("  "));
    assertEquals(
        "{{- {\"a\": [1, 2, 3]}[\"a\"][1:] | join(\",\") -}}",
        Template.parse("{{ {'a': [1, 2, 3]}['a'][1:]|join(',') }}").format());
  }

  @Test
  void numericIndentationDefaultsAndRejectsInvalidCounts() {
    var template = Template.parse("x");
    assertEquals("{{- \"x\" -}}", template.format(0));
    assertEquals("{{- \"x\" -}}", template.format(Double.NaN));
    assertEquals(
        "Invalid indentation count: -1",
        assertThrows(IllegalArgumentException.class, () -> template.format(-1)).getMessage());
    assertEquals(
        "Invalid indentation count: Infinity",
        assertThrows(
                IllegalArgumentException.class, () -> template.format(Double.POSITIVE_INFINITY))
            .getMessage());
    assertEquals(
        "Invalid indentation count: 536870889",
        assertThrows(IllegalArgumentException.class, () -> template.format(536_870_889d))
            .getMessage());
  }
}
