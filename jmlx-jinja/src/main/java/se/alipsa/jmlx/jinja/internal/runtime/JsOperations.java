package se.alipsa.jmlx.jinja.internal.runtime;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.internal.JsFormat;
import se.alipsa.jmlx.jinja.internal.Value;

/** Stateless JavaScript-value coercion and operator helpers used by the interpreter. */
@SuppressWarnings("doclint:missing")
final class JsOperations {
  private JsOperations() {}

  static Value add(Value left, Value right, SourceLocation location) {
    if (left instanceof Value.StringValue || right instanceof Value.StringValue)
      return new Value.StringValue(payloadText(left, location) + payloadText(right, location));
    return numericAdd(left, right);
  }

  static Value arithmetic(String operator, Value left, Value right) {
    double a = number(left);
    double b = number(right);
    boolean floating = left instanceof Value.FloatValue || right instanceof Value.FloatValue;
    return switch (operator) {
      case "+" -> numericAdd(left, right);
      case "-" -> numeric(a - b, floating);
      case "*" -> numeric(a * b, floating);
      case "/" -> new Value.FloatValue(a / b);
      case "%" -> numeric(a % b, floating);
      default -> throw new IllegalArgumentException("Not an arithmetic operator: " + operator);
    };
  }

  static boolean compare(String operator, Value left, Value right) {
    double a = number(left);
    double b = number(right);
    return switch (operator) {
      case "<" -> a < b;
      case ">" -> a > b;
      case "<=" -> a <= b;
      case ">=" -> a >= b;
      default -> throw new IllegalArgumentException("Not a comparison operator: " + operator);
    };
  }

  static Value concatenate(List<Value> left, List<Value> right) {
    var values = new ArrayList<Value>(left);
    values.addAll(right);
    return new Value.ArrayValue(values);
  }

  static boolean contains(Value needle, Value.ArrayValue haystack) {
    return haystack.values().stream().anyMatch(value -> strictValueEquals(value, needle));
  }

  static boolean contains(Value.StringValue needle, Value.StringValue haystack) {
    return haystack.value().contains(needle.value());
  }

  static boolean contains(Value.StringValue needle, Value.ObjectValue haystack) {
    return haystack.values().containsKey(needle.undefinedBacked() ? needle : needle.value());
  }

  /** Returns the upstream payload stringification used by string {@code +}. */
  static String payloadText(Value value, SourceLocation location) {
    return switch (value) {
      case Value.StringValue x -> x.undefinedBacked() ? undefinedPayload(location) : x.value();
      case Value.IntegerValue x -> JsFormat.plainString(x.value());
      case Value.FloatValue x -> JsFormat.plainString(x.value());
      case Value.BooleanValue x -> Boolean.toString(x.value());
      case Value.NullValue ignored -> "undefined";
      case Value.UndefinedValue ignored -> "undefined";
      case Value.DeferredUndefinedValue ignored -> undefinedPayload(location);
      case Value.ArrayValue x -> arrayPayloadText(x.values(), location);
      case Value.TupleValue x -> arrayPayloadText(x.values(), location);
      case Value.ObjectValue ignored -> "[object Map]";
      case Value.KeywordArgumentsValue ignored -> "[object Map]";
      case Value.CallableValue x -> x.renderedText();
    };
  }

  static double toNumber(Value value) {
    if (value instanceof Value.IntegerValue x) return x.value();
    if (value instanceof Value.FloatValue x) return x.value();
    if (value instanceof Value.NullValue) return 0;
    if (value instanceof Value.BooleanValue x) return x.value() ? 1 : 0;
    if (value instanceof Value.StringValue x) return stringNumber(x.value());
    return Double.NaN;
  }

  static boolean looseEquals(Value left, Value right) {
    if (nilLike(left) || nilLike(right)) return nilLike(left) && nilLike(right);
    if (left instanceof Value.BooleanValue a && right instanceof Value.BooleanValue b)
      return a.value() == b.value();
    if (numeric(left) && right instanceof Value.BooleanValue x)
      return number(left) == (x.value() ? 1 : 0);
    if (numeric(right) && left instanceof Value.BooleanValue x)
      return number(right) == (x.value() ? 1 : 0);
    if (left instanceof Value.BooleanValue x) return looseNumberEquals(x.value() ? 1 : 0, right);
    if (right instanceof Value.BooleanValue x) return looseNumberEquals(x.value() ? 1 : 0, left);
    if (numeric(left) && right instanceof Value.StringValue x)
      return number(left) == stringNumber(x.value());
    if (numeric(right) && left instanceof Value.StringValue x)
      return number(right) == stringNumber(x.value());
    return strictValueEquals(left, right);
  }

  static boolean strictEquals(Value left, Value right) {
    if (nilLike(left) || nilLike(right)) return nilLike(left) && nilLike(right);
    return strictValueEquals(left, right);
  }

  static boolean rawTruthy(Value value) {
    return switch (value) {
      case Value.NullValue ignored -> false;
      case Value.UndefinedValue ignored -> false;
      case Value.DeferredUndefinedValue ignored ->
          throw new TemplateRenderException(
              "Cannot read properties of undefined (reading '__bool__')", ErrorCategory.TYPE, null);
      case Value.BooleanValue x -> x.value();
      case Value.IntegerValue x -> x.value() != 0 && !Double.isNaN(x.value());
      case Value.FloatValue x -> x.value() != 0 && !Double.isNaN(x.value());
      case Value.StringValue x -> !x.undefinedBacked() && !x.value().isEmpty();
      default -> true;
    };
  }

  static boolean numeric(Value value) {
    return value instanceof Value.IntegerValue || value instanceof Value.FloatValue;
  }

  static boolean nilLike(Value value) {
    return value instanceof Value.NullValue
        || value instanceof Value.UndefinedValue
        || value instanceof Value.StringValue string && string.undefinedBacked();
  }

  private static boolean looseNumberEquals(double number, Value other) {
    return numeric(other)
        ? number == number(other)
        : other instanceof Value.StringValue x && number == stringNumber(x.value());
  }

  static boolean strictValueEquals(Value left, Value right) {
    if (numeric(left) && numeric(right)) return number(left) == number(right);
    if (left instanceof Value.StringValue a && right instanceof Value.StringValue b)
      return a.undefinedBacked() == b.undefinedBacked() && a.value().equals(b.value());
    if (left instanceof Value.BooleanValue a && right instanceof Value.BooleanValue b)
      return a.value() == b.value();
    return left == right;
  }

  private static Value numeric(double value, boolean floating) {
    return floating ? new Value.FloatValue(value) : new Value.IntegerValue(value);
  }

  private static Value numericAdd(Value left, Value right) {
    double result = toNumber(left) + toNumber(right);
    boolean floating = left instanceof Value.FloatValue || right instanceof Value.FloatValue;
    return numeric(result, floating);
  }

  private static double number(Value value) {
    return value instanceof Value.IntegerValue x ? x.value() : ((Value.FloatValue) value).value();
  }

  private static String arrayPayloadText(List<Value> values, SourceLocation location) {
    var text = new StringBuilder();
    for (int i = 0; i < values.size(); i++) {
      if (i > 0) text.append(',');
      var value = values.get(i);
      text.append(wrapperText(value, location));
    }
    return text.toString();
  }

  private static String wrapperText(Value value, SourceLocation location) {
    return switch (value) {
      case Value.ArrayValue ignored -> JsFormat.runtimeJson(value, location);
      case Value.ObjectValue ignored -> JsFormat.runtimeJson(value, location);
      case Value.TupleValue ignored -> JsFormat.runtimeJson(value, location);
      case Value.StringValue x -> x.undefinedBacked() ? "undefined" : x.value();
      case Value.FloatValue x -> JsFormat.floatString(x.value());
      default -> payloadText(value, location);
    };
  }

  private static String undefinedPayload(SourceLocation location) {
    throw new TemplateRenderException(
        "Cannot read properties of undefined (reading 'toString')", ErrorCategory.TYPE, location);
  }

  private static double stringNumber(String value) {
    var text = trimEcmaWhitespace(value);
    if (text.isEmpty()) return 0;
    if (text.equals("Infinity") || text.equals("+Infinity")) return Double.POSITIVE_INFINITY;
    if (text.equals("-Infinity")) return Double.NEGATIVE_INFINITY;
    if (text.matches("0[xX][0-9a-fA-F]+"))
      return new BigInteger(text.substring(2), 16).doubleValue();
    if (text.matches("0[oO][0-7]+")) return new BigInteger(text.substring(2), 8).doubleValue();
    if (text.matches("0[bB][01]+")) return new BigInteger(text.substring(2), 2).doubleValue();
    if (!text.matches("[+-]?(?:(?:\\d+\\.?\\d*)|(?:\\.\\d+))(?:[eE][+-]?\\d+)?")) return Double.NaN;
    return Double.parseDouble(text);
  }

  static String trimEcmaWhitespace(String value) {
    int start = 0;
    int end = value.length();
    while (start < end && isEcmaWhitespace(value.charAt(start))) start++;
    while (end > start && isEcmaWhitespace(value.charAt(end - 1))) end--;
    return value.substring(start, end);
  }

  static String trimStartEcmaWhitespace(String value) {
    int start = 0;
    while (start < value.length() && isEcmaWhitespace(value.charAt(start))) start++;
    return value.substring(start);
  }

  static String trimEndEcmaWhitespace(String value) {
    int end = value.length();
    while (end > 0 && isEcmaWhitespace(value.charAt(end - 1))) end--;
    return value.substring(0, end);
  }

  static boolean isEcmaWhitespace(char value) {
    return Character.getType(value) == Character.SPACE_SEPARATOR
        || value == '\u0009'
        || value == '\u000B'
        || value == '\u000C'
        || value == '\n'
        || value == '\r'
        || value == '\u2028'
        || value == '\u2029'
        || value == '\uFEFF';
  }
}
