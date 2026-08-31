package se.alipsa.jmlx.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Map;
import org.junit.jupiter.api.Test;

class PublicApiTest {
  @Test
  void sourceLocationsAreValidated() {
    assertThrows(IllegalArgumentException.class, () -> new SourceLocation(-1, 1, 1));
    assertEquals(new SourceLocation(0, 1, 1), new SourceLocation(0, 1, 1));
  }

  @Test
  void conversionCategoryCanHaveNoLocation() {
    var error = new TemplateRenderException("bad input", ErrorCategory.HOST_CONVERSION, null);
    assertEquals(ErrorCategory.HOST_CONVERSION, error.category());
    assertFalse(error.location().isPresent());
  }

  @Test
  void exceptionsWithLocationsAreSerializable() throws Exception {
    var original = new TemplateSyntaxException("bad template", new SourceLocation(3, 1, 4));
    var bytes = new ByteArrayOutputStream();
    try (var output = new ObjectOutputStream(bytes)) {
      output.writeObject(original);
    }
    TemplateSyntaxException restored;
    try (var input = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
      restored = (TemplateSyntaxException) input.readObject();
    }
    assertEquals(original.getMessage(), restored.getMessage());
    assertEquals(original.category(), restored.category());
    assertEquals(original.location(), restored.location());
  }

  @Test
  void parsedTemplateRendersLiteralText() {
    var template = Template.parse("literal");
    assertEquals("literal", template.render(Map.of()));
  }

  @Test
  void convertedHostCallableSharesTheBuiltinFunctionSource() {
    // JSONL corpus fixtures cannot inject RenderOptions host functions, so this public-boundary
    // assertion covers the Java-only host-function registration path.
    var options = RenderOptions.builder().hostFunction("host", arguments -> null).build();
    var rendered = Template.parse("{{ range }}|{{ host }}").render(Map.of(), options);
    var functions = rendered.split("\\|", -1);
    assertEquals(2, functions.length);
    assertEquals(functions[0], functions[1]);
  }

  @Test
  void formattingOverloadsValidateInputsAndDoNotMutateTheTemplate() {
    var template = Template.parse("{% if a %}{{ x }}{% endif %}");
    var context = Map.of("a", true, "x", "x");
    final var beforeFormatting = template.render(context);
    assertThrows(NullPointerException.class, () -> template.format((String) null));
    assertEquals(template.format(9), template.format('\t'));
    assertEquals("{%- if a -%}\n\t{{- x -}}\n{%- endif -%}", template.format(""));
    assertEquals(beforeFormatting, template.render(context));
  }

  @Test
  void parseRejectsMalformedTemplatesAtThePublicBoundary() {
    var error = assertThrows(TemplateSyntaxException.class, () -> Template.parse("{% if x %}"));
    assertEquals(ErrorCategory.SYNTAX, error.category());
    assertTrue(error.location().isPresent());
  }

  @Test
  void renderOptionsAreImmutableAndRejectInvalidFunctionRegistrations() {
    var clock = Clock.fixed(Instant.parse("2025-01-02T03:04:05Z"), ZoneOffset.UTC);
    HostFunction formatter = arguments -> "formatted";
    var options =
        RenderOptions.builder()
            .clock(clock)
            .zoneId(ZoneOffset.UTC)
            .hostFunction("format_tool", formatter)
            .build();

    assertEquals(clock, options.clock().orElseThrow());
    assertEquals(ZoneOffset.UTC, options.zoneId().orElseThrow());
    assertEquals(formatter, options.hostFunctions().get("format_tool"));
    assertThrows(
        UnsupportedOperationException.class, () -> options.hostFunctions().put("other", formatter));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            RenderOptions.builder()
                .hostFunction("duplicate", formatter)
                .hostFunction("duplicate", formatter)
                .build());
    assertThrows(
        IllegalArgumentException.class,
        () -> RenderOptions.builder().hostFunction("range", formatter).build());
    assertThrows(
        IllegalArgumentException.class, () -> RenderOptions.builder().hostFunction(" ", formatter));
    assertThrows(
        IllegalArgumentException.class,
        () -> RenderOptions.builder().hostFunction("format-tool", formatter));
    assertThrows(
        IllegalArgumentException.class,
        () -> RenderOptions.builder().hostFunction("two words", formatter));
    assertThrows(
        IllegalArgumentException.class,
        () -> RenderOptions.builder().hostFunction("1st", formatter));
    assertThrows(
        NullPointerException.class,
        () -> RenderOptions.builder().hostFunction("format_tool", null));
  }

  @Test
  void renderOptionsRejectNonpositiveResourceLimits() {
    assertThrows(IllegalArgumentException.class, () -> RenderOptions.builder().maxSteps(0));
    assertThrows(IllegalArgumentException.class, () -> RenderOptions.builder().maxSteps(-1));
    assertThrows(
        IllegalArgumentException.class, () -> RenderOptions.builder().maxLoopIterations(0));
    assertThrows(
        IllegalArgumentException.class, () -> RenderOptions.builder().maxLoopIterations(-1));
    assertThrows(IllegalArgumentException.class, () -> RenderOptions.builder().maxOutputLength(0));
    assertThrows(IllegalArgumentException.class, () -> RenderOptions.builder().maxOutputLength(-1));
    assertThrows(IllegalArgumentException.class, () -> RenderOptions.builder().maxMacroDepth(0));
    assertThrows(IllegalArgumentException.class, () -> RenderOptions.builder().maxMacroDepth(-1));
  }

  @Test
  void templateOptionsHaveSaneDefaultsAndRejectInvalidLimits() {
    assertEquals(1_048_576, TemplateOptions.DEFAULT.maxSourceLength());
    assertEquals(200_000, TemplateOptions.DEFAULT.maxTokenCount());
    assertEquals(256, TemplateOptions.DEFAULT.maxAstDepth());
    assertTrue(TemplateOptions.DEFAULT.trimBlocks());
    assertTrue(TemplateOptions.DEFAULT.lstripBlocks());
    assertEquals(
        TemplateOptions.DEFAULT.trimBlocks(), TemplateOptions.builder().build().trimBlocks());
    assertEquals(
        TemplateOptions.DEFAULT.lstripBlocks(), TemplateOptions.builder().build().lstripBlocks());

    var options =
        TemplateOptions.builder()
            .maxSourceLength(10)
            .maxTokenCount(20)
            .trimBlocks(true)
            .lstripBlocks(true)
            .build();
    assertEquals(10, options.maxSourceLength());
    assertEquals(20, options.maxTokenCount());
    assertEquals(256, options.maxAstDepth());
    assertTrue(options.trimBlocks());
    assertTrue(options.lstripBlocks());

    assertThrows(
        IllegalArgumentException.class, () -> TemplateOptions.builder().maxSourceLength(0));
    assertThrows(
        IllegalArgumentException.class, () -> TemplateOptions.builder().maxSourceLength(-1));
    assertThrows(IllegalArgumentException.class, () -> TemplateOptions.builder().maxTokenCount(0));
    assertThrows(IllegalArgumentException.class, () -> TemplateOptions.builder().maxTokenCount(-1));
    assertThrows(IllegalArgumentException.class, () -> TemplateOptions.builder().maxAstDepth(0));
    assertThrows(IllegalArgumentException.class, () -> TemplateOptions.builder().maxAstDepth(-1));
  }
}
