package se.alipsa.jmlx.jinja.internal;

import java.util.List;
import se.alipsa.jmlx.jinja.internal.ast.Expression;
import se.alipsa.jmlx.jinja.internal.ast.Statement;

/** Canonical serializer for the pinned upstream template AST. */
public final class TemplateFormatter {
  static final int MAX_PINNED_NODE_STRING_LENGTH = 536_870_888;
  private static final String NL = "\n";

  private TemplateFormatter() {}

  /**
   * Serializes a parsed program using an already validated indentation unit.
   *
   * @param program the parsed template program
   * @param indent the indentation unit
   * @return canonical template text
   */
  public static String format(Statement.Program program, String indent) {
    return statements(program.body(), 0, indent).replaceFirst("\\n$", "");
  }

  /**
   * Normalizes a truthy numeric indent before constructing a space string.
   *
   * <p>Callers must apply JavaScript's NaN/zero defaulting first; this helper intentionally maps
   * NaN to zero through the same floor conversion used for finite fractional values.
   *
   * @param value the requested numeric count
   * @return the normalized nonnegative count
   * @throws IllegalArgumentException if the normalized count is too large or negative
   */
  public static int validateIndentCount(double value) {
    double count = value < 0 ? Math.ceil(value) : Math.floor(value);
    if (count < 0 || count > MAX_PINNED_NODE_STRING_LENGTH) {
      throw invalidCount(value);
    }
    return (int) count;
  }

  static void validateRepeatedIndentLength(int unitLength, int depth) {
    if ((long) unitLength * depth > MAX_PINNED_NODE_STRING_LENGTH) {
      throw new IllegalArgumentException("Indentation exceeds pinned Node string length limit");
    }
  }

  private static IllegalArgumentException invalidCount(double value) {
    return new IllegalArgumentException(
        "Invalid indentation count: " + JsFormat.plainString(value));
  }

  private static String statements(List<Statement> body, int depth, String indent) {
    return body.stream()
        .map(x -> statement(x, depth, indent))
        .reduce((a, b) -> a + NL + b)
        .orElse("");
  }

  private static String pad(int depth, String indent) {
    validateRepeatedIndentLength(indent.length(), depth);
    return indent.repeat(depth);
  }

  private static String tag(String... text) {
    return "{%- " + String.join(" ", text) + " -%}";
  }

  private static String statement(Statement node, int depth, String indent) {
    String pad = pad(depth, indent);
    return switch (node) {
      case Statement.Program x -> statements(x.body(), depth, indent);
      case Statement.If x -> formatIf(x, depth, indent);
      case Statement.For x -> formatFor(x, depth, indent);
      case Statement.SetStatement x -> formatSet(x, depth, indent);
      case Statement.Macro x ->
          pad
              + tag("macro", x.name().value() + "(" + expressions(x.args()) + ")")
              + NL
              + statements(x.body(), depth + 1, indent)
              + NL
              + pad
              + tag("endmacro");
      case Statement.Break ignored -> pad + tag("break");
      case Statement.Continue ignored -> pad + tag("continue");
      case Statement.Comment x -> pad + "{# " + x.value() + " #}";
      case Statement.FilterStatement x ->
          pad
              + tag("filter", filterSpec(x.filter()))
              + NL
              + statements(x.body(), depth + 1, indent)
              + NL
              + pad
              + tag("endfilter");
      case Statement.CallStatement x -> formatCall(x, depth, indent);
      case Expression x -> pad + "{{- " + expression(x, -1) + " -}}";
    };
  }

  private static String formatIf(Statement.If node, int depth, String indent) {
    String pad = pad(depth, indent);
    var out = new StringBuilder();
    Statement.If current = node;
    boolean first = true;
    while (true) {
      if (!first) {
        out.append(NL);
      }
      out.append(pad)
          .append(tag(first ? "if" : "elif", expression(current.test(), -1)))
          .append(NL)
          .append(statements(current.body(), depth + 1, indent));
      if (current.alternate().size() == 1
          && current.alternate().getFirst() instanceof Statement.If next) {
        current = next;
        first = false;
      } else {
        break;
      }
    }
    if (!current.alternate().isEmpty()) {
      out.append(NL)
          .append(pad)
          .append(tag("else"))
          .append(NL)
          .append(statements(current.alternate(), depth + 1, indent));
    }
    return out.append(NL).append(pad).append(tag("endif")).toString();
  }

  private static String formatFor(Statement.For x, int depth, String indent) {
    String iterable =
        x.iterable() instanceof Expression.SelectExpression s
            ? expression(s.lhs(), -1) + " if " + expression(s.test(), -1)
            : expression(x.iterable(), -1);
    String pad = pad(depth, indent);
    var out =
        new StringBuilder(pad)
            .append(tag("for", expression(x.loopVariable(), -1), "in", iterable))
            .append(NL)
            .append(statements(x.body(), depth + 1, indent));
    if (!x.defaultBlock().isEmpty()) {
      out.append(NL)
          .append(pad)
          .append(tag("else"))
          .append(NL)
          .append(statements(x.defaultBlock(), depth + 1, indent));
    }
    return out.append(NL).append(pad).append(tag("endfor")).toString();
  }

  private static String formatSet(Statement.SetStatement x, int depth, String indent) {
    String pad = pad(depth, indent);
    String value =
        pad
            + tag(
                "set",
                expression(x.assignee(), -1)
                    + (x.value() == null ? "" : " = " + expression(x.value(), -1)));
    return x.body().isEmpty()
        ? value
        : value + NL + statements(x.body(), depth + 1, indent) + NL + pad + tag("endset");
  }

  private static String formatCall(Statement.CallStatement x, int depth, String indent) {
    String params =
        x.callerArgs() == null || x.callerArgs().isEmpty()
            ? ""
            : "(" + expressions(x.callerArgs()) + ")";
    String pad = pad(depth, indent);
    return pad
        + tag("call" + params, expression(x.call(), -1))
        + NL
        + statements(x.body(), depth + 1, indent)
        + NL
        + pad
        + tag("endcall");
  }

  private static String filterSpec(Expression x) {
    return x instanceof Expression.Identifier i ? i.value() : expression(x, -1);
  }

  private static String expressions(List<Expression> values) {
    return values.stream().map(x -> expression(x, -1)).reduce((a, b) -> a + ", " + b).orElse("");
  }

  private static int precedence(Expression.BinaryExpression x) {
    return switch (x.operator().type()) {
      case MultiplicativeBinaryOperator -> 4;
      case AdditiveBinaryOperator -> 3;
      case ComparisonBinaryOperator -> 2;
      case Identifier ->
          x.operator().value().equals("and")
              ? 1
              : (x.operator().value().equals("in") || x.operator().value().equals("not in"))
                  ? 2
                  : 0;
      default -> 0;
    };
  }

  private static String expression(Expression node, int parentPrecedence) {
    return switch (node) {
      case Expression.Identifier x -> x.value();
      case Expression.IntegerLiteral x -> JsFormat.plainString(x.value());
      case Expression.FloatLiteral x -> JsFormat.plainString(x.value());
      case Expression.StringLiteral x -> JsFormat.quote(x.value());
      case Expression.SpreadExpression x -> "*" + expression(x.argument(), -1);
      case Expression.BinaryExpression x -> binary(x, parentPrecedence);
      case Expression.UnaryExpression x ->
          x.operator().value()
              + (x.operator().value().equals("not") ? " " : "")
              + expression(x.argument(), Integer.MAX_VALUE);
      case Expression.CallExpression x ->
          expression(x.callee(), -1) + "(" + expressions(x.args()) + ")";
      case Expression.MemberExpression x -> member(x);
      case Expression.FilterExpression x ->
          expression(x.operand(), Integer.MAX_VALUE) + " | " + filterSpec(x.filter());
      case Expression.SelectExpression x ->
          expression(x.lhs(), -1) + " if " + expression(x.test(), -1);
      case Expression.TestExpression x ->
          expression(x.operand(), -1) + " is" + (x.negate() ? " not" : "") + " " + x.test().value();
      case Expression.ArrayLiteral x -> "[" + expressions(x.value()) + "]";
      case Expression.TupleLiteral x -> "(" + expressions(x.value()) + ")";
      case Expression.ObjectLiteral x ->
          "{"
              + x.value().stream()
                  .map(e -> expression(e.key(), -1) + ": " + expression(e.value(), -1))
                  .reduce((a, b) -> a + ", " + b)
                  .orElse("")
              + "}";
      case Expression.SliceExpression x ->
          (x.start() == null ? "" : expression(x.start(), -1))
              + ":"
              + (x.stop() == null ? "" : expression(x.stop(), -1))
              + (x.step() == null ? "" : ":" + expression(x.step(), -1));
      case Expression.KeywordArgumentExpression x ->
          x.key().value() + "=" + expression(x.value(), -1);
      case Expression.Ternary x -> ternary(x, parentPrecedence);
    };
  }

  private static String binary(Expression.BinaryExpression x, int parent) {
    int p = precedence(x);
    String value =
        expression(x.left(), p) + " " + x.operator().value() + " " + expression(x.right(), p + 1);
    return p < parent ? "(" + value + ")" : value;
  }

  private static String ternary(Expression.Ternary x, int parent) {
    String value =
        expression(x.trueExpr(), -1)
            + " if "
            + expression(x.condition(), 0)
            + " else "
            + expression(x.falseExpr(), -1);
    return parent > -1 ? "(" + value + ")" : value;
  }

  private static String member(Expression.MemberExpression x) {
    String object = expression(x.object(), -1);
    if (!(x.object() instanceof Expression.Identifier
        || x.object() instanceof Expression.MemberExpression
        || x.object() instanceof Expression.CallExpression
        || x.object() instanceof Expression.StringLiteral
        || x.object() instanceof Expression.IntegerLiteral
        || x.object() instanceof Expression.FloatLiteral
        || x.object() instanceof Expression.ArrayLiteral
        || x.object() instanceof Expression.TupleLiteral
        || x.object() instanceof Expression.ObjectLiteral)) {
      object = "(" + object + ")";
    }
    String property = expression(x.property(), -1);
    if (!x.computed()
        && !(x.property() instanceof Expression.Identifier
            || x.property() instanceof Expression.IntegerLiteral)) {
      property = "(" + property + ")";
    }
    return x.computed() ? object + "[" + property + "]" : object + "." + property;
  }
}
