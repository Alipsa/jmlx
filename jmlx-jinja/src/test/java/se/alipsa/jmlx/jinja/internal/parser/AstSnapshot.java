package se.alipsa.jmlx.jinja.internal.parser;

import java.util.List;
import se.alipsa.jmlx.jinja.internal.JsFormat;
import se.alipsa.jmlx.jinja.internal.ast.Expression;
import se.alipsa.jmlx.jinja.internal.ast.Statement;

/** Test-only deterministic AST serializer used by the Node differential test. */
final class AstSnapshot {
  private AstSnapshot() {}

  static String of(Statement node) {
    var out = new StringBuilder();
    emit(node, "", out);
    return out.toString();
  }

  private static void line(String text, String indent, StringBuilder out) {
    out.append(indent).append(text).append('\n');
  }

  private static void list(
      String name, List<? extends Statement> values, String indent, StringBuilder out) {
    line(name, indent, out);
    if (values == null) line("-", indent + "  ", out);
    else for (var value : values) emit(value, indent + "  ", out);
  }

  private static void value(String name, Statement value, String indent, StringBuilder out) {
    line(name, indent, out);
    if (value == null) line("-", indent + "  ", out);
    else emit(value, indent + "  ", out);
  }

  private static void emit(Statement n, String i, StringBuilder o) {
    switch (n) {
      case Statement.Program x -> {
        line("Program", i, o);
        list("body", x.body(), i + "  ", o);
      }
      case Statement.If x -> {
        line("If", i, o);
        line("test", i + "  ", o);
        emit(x.test(), i + "    ", o);
        list("body", x.body(), i + "  ", o);
        list("alternate", x.alternate(), i + "  ", o);
      }
      case Statement.For x -> {
        line("For", i, o);
        line("loopvar", i + "  ", o);
        emit(x.loopVariable(), i + "    ", o);
        line("iterable", i + "  ", o);
        emit(x.iterable(), i + "    ", o);
        list("body", x.body(), i + "  ", o);
        list("defaultBlock", x.defaultBlock(), i + "  ", o);
      }
      case Statement.Break x -> line("Break", i, o);
      case Statement.Continue x -> line("Continue", i, o);
      case Statement.Comment x -> {
        line("Comment", i, o);
        line("value", i + "  ", o);
        line(q(x.value()), i + "    ", o);
      }
      case Statement.SetStatement x -> {
        line("Set", i, o);
        value("assignee", x.assignee(), i + "  ", o);
        value("value", x.value(), i + "  ", o);
        list("body", x.body(), i + "  ", o);
      }
      case Statement.Macro x -> {
        line("Macro", i, o);
        line("name", i + "  ", o);
        emit(x.name(), i + "    ", o);
        list("args", x.args(), i + "  ", o);
        list("body", x.body(), i + "  ", o);
      }
      case Statement.FilterStatement x -> {
        line("FilterStatement", i, o);
        line("filter", i + "  ", o);
        emit(x.filter(), i + "    ", o);
        list("body", x.body(), i + "  ", o);
      }
      case Statement.CallStatement x -> {
        line("CallStatement", i, o);
        line("call", i + "  ", o);
        emit(x.call(), i + "    ", o);
        list("callerArgs", x.callerArgs(), i + "  ", o);
        list("body", x.body(), i + "  ", o);
      }
      case Expression.MemberExpression x -> {
        line("MemberExpression", i, o);
        line("object", i + "  ", o);
        emit(x.object(), i + "    ", o);
        line("property", i + "  ", o);
        emit(x.property(), i + "    ", o);
        line("computed", i + "  ", o);
        line(Boolean.toString(x.computed()), i + "    ", o);
      }
      case Expression.CallExpression x -> {
        line("CallExpression", i, o);
        line("callee", i + "  ", o);
        emit(x.callee(), i + "    ", o);
        list("args", x.args(), i + "  ", o);
      }
      case Expression.Identifier x -> scalar("Identifier", q(x.value()), i, o);
      case Expression.IntegerLiteral x -> scalar("IntegerLiteral", number(x.value()), i, o);
      case Expression.FloatLiteral x -> scalar("FloatLiteral", number(x.value()), i, o);
      case Expression.StringLiteral x -> scalar("StringLiteral", q(x.value()), i, o);
      case Expression.ArrayLiteral x -> {
        line("ArrayLiteral", i, o);
        list("value", x.value(), i + "  ", o);
      }
      case Expression.TupleLiteral x -> {
        line("TupleLiteral", i, o);
        list("value", x.value(), i + "  ", o);
      }
      case Expression.ObjectLiteral x -> {
        line("ObjectLiteral", i, o);
        line("value", i + "  ", o);
        for (var e : x.value()) {
          line("key", i + "    ", o);
          emit(e.key(), i + "      ", o);
          line("value", i + "    ", o);
          emit(e.value(), i + "      ", o);
        }
      }
      case Expression.BinaryExpression x -> {
        line("BinaryExpression", i, o);
        line("operator", i + "  ", o);
        token(x.operator(), i + "    ", o);
        line("left", i + "  ", o);
        emit(x.left(), i + "    ", o);
        line("right", i + "  ", o);
        emit(x.right(), i + "    ", o);
      }
      case Expression.FilterExpression x ->
          binary("FilterExpression", "operand", x.operand(), "filter", x.filter(), i, o);
      case Expression.SelectExpression x ->
          binary("SelectExpression", "lhs", x.lhs(), "test", x.test(), i, o);
      case Expression.TestExpression x -> {
        line("TestExpression", i, o);
        line("operand", i + "  ", o);
        emit(x.operand(), i + "    ", o);
        line("negate", i + "  ", o);
        line(Boolean.toString(x.negate()), i + "    ", o);
        line("test", i + "  ", o);
        emit(x.test(), i + "    ", o);
      }
      case Expression.UnaryExpression x -> {
        line("UnaryExpression", i, o);
        line("operator", i + "  ", o);
        token(x.operator(), i + "    ", o);
        line("argument", i + "  ", o);
        emit(x.argument(), i + "    ", o);
      }
      case Expression.SliceExpression x -> {
        line("SliceExpression", i, o);
        value("start", x.start(), i + "  ", o);
        value("stop", x.stop(), i + "  ", o);
        value("step", x.step(), i + "  ", o);
      }
      case Expression.KeywordArgumentExpression x ->
          binary("KeywordArgumentExpression", "key", x.key(), "value", x.value(), i, o);
      case Expression.SpreadExpression x -> {
        line("SpreadExpression", i, o);
        line("argument", i + "  ", o);
        emit(x.argument(), i + "    ", o);
      }
      case Expression.Ternary x -> {
        line("Ternary", i, o);
        line("condition", i + "  ", o);
        emit(x.condition(), i + "    ", o);
        line("trueExpr", i + "  ", o);
        emit(x.trueExpr(), i + "    ", o);
        line("falseExpr", i + "  ", o);
        emit(x.falseExpr(), i + "    ", o);
      }
    }
  }

  private static void scalar(String type, String value, String i, StringBuilder o) {
    line(type, i, o);
    line("value", i + "  ", o);
    line(value, i + "    ", o);
  }

  private static void token(
      se.alipsa.jmlx.jinja.internal.lexer.Token token, String i, StringBuilder o) {
    line(token.type().toString(), i, o);
    line("value", i + "  ", o);
    line(q(token.value()), i + "    ", o);
  }

  private static void binary(
      String type, String a, Statement av, String b, Statement bv, String i, StringBuilder o) {
    line(type, i, o);
    line(a, i + "  ", o);
    emit(av, i + "    ", o);
    line(b, i + "  ", o);
    emit(bv, i + "    ", o);
  }

  private static String q(String value) {
    return JsFormat.quote(value);
  }

  /** Delegates JSON-compatible number formatting to the shared runtime formatter. */
  private static String number(double value) {
    return JsFormat.jsonString(value);
  }
}
