package se.alipsa.jmlx.jinja;

import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.jinja.internal.TemplateFormatter;
import se.alipsa.jmlx.jinja.internal.Value;
import se.alipsa.jmlx.jinja.internal.Values;
import se.alipsa.jmlx.jinja.internal.ast.Statement;
import se.alipsa.jmlx.jinja.internal.lexer.Lexer;
import se.alipsa.jmlx.jinja.internal.parser.Parser;
import se.alipsa.jmlx.jinja.internal.runtime.Interpreter;

/** An immutable parsed template. */
public final class Template {
  private final Statement.Program program;

  private Template(Statement.Program program) {
    this.program = program;
  }

  /**
   * Parses a template with {@link TemplateOptions#DEFAULT}.
   *
   * @param source the template source
   * @return the parsed template
   */
  public static Template parse(String source) {
    return parse(source, TemplateOptions.DEFAULT);
  }

  /**
   * Parses a template.
   *
   * @param source the template source
   * @param options parse-time limits and syntax options
   * @return the parsed template
   */
  public static Template parse(String source, TemplateOptions options) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(options, "options");
    return new Template(Parser.parse(Lexer.tokenize(source, options), options));
  }

  /**
   * Formats this parsed template using the pinned upstream's tab indentation default.
   *
   * <p>The result reproduces the pinned upstream's canonical bytes and is not guaranteed to parse
   * or render equivalently when formatted again.
   *
   * @return the canonical template text
   */
  public String format() {
    return TemplateFormatter.format(program, "\t");
  }

  /**
   * Formats this parsed template using a string indentation unit; an empty unit selects a tab.
   *
   * @param indent the indentation unit
   * @return the canonical template text
   * @throws NullPointerException if {@code indent} is null
   */
  public String format(String indent) {
    Objects.requireNonNull(indent, "indent");
    return TemplateFormatter.format(program, indent.isEmpty() ? "\t" : indent);
  }

  /**
   * Formats this parsed template using a numeric space indentation count.
   *
   * <p>A character argument such as {@code format('\t')} widens to its numeric code point (nine);
   * use {@link #format(String)} for a text indentation unit.
   *
   * @param indent the requested number of spaces per nesting level
   * @return the canonical template text
   * @throws IllegalArgumentException if the normalized count is negative or exceeds the pinned Node
   *     string-length limit
   */
  public String format(double indent) {
    if (Double.isNaN(indent) || indent == 0d) {
      return format();
    }
    return TemplateFormatter.format(
        program, " ".repeat(TemplateFormatter.validateIndentCount(indent)));
  }

  /**
   * Renders this template with {@link RenderOptions#DEFAULT}.
   *
   * @param context the top-level template variables
   * @return the rendered output
   */
  public String render(Map<String, ?> context) {
    return render(context, RenderOptions.DEFAULT);
  }

  /**
   * Renders this template.
   *
   * @param context the top-level template variables
   * @param options render-time clock/zone and host-function options
   * @return the rendered output
   */
  public String render(Map<String, ?> context, RenderOptions options) {
    Objects.requireNonNull(context, "context");
    Objects.requireNonNull(options, "options");
    var output = new StringBuilder();
    render(context, output, options);
    return output.toString();
  }

  /**
   * Renders this template with {@link RenderOptions#DEFAULT}, buffering the complete result before
   * appending it once to {@code output}.
   *
   * @param context the top-level template variables
   * @param output the destination for rendered output
   */
  public void render(Map<String, ?> context, Appendable output) {
    render(context, output, RenderOptions.DEFAULT);
  }

  /**
   * Renders this template, buffering the complete result before appending it once to {@code
   * output}.
   *
   * @param context the top-level template variables
   * @param output the destination for rendered output
   * @param options render-time clock/zone and host-function options
   */
  public void render(Map<String, ?> context, Appendable output, RenderOptions options) {
    Objects.requireNonNull(context, "context");
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(options, "options");
    var value = Values.fromHost(context);
    Interpreter.render(program, (Value.ObjectValue) value, options, output);
  }
}
