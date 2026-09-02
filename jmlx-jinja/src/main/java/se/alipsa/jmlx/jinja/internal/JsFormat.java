package se.alipsa.jmlx.jinja.internal;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.text.Collator;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateRenderException;

/** JavaScript-compatible number and JSON string formatting helpers. */
@SuppressWarnings("doclint:missing")
public final class JsFormat {
  private JsFormat() {}

  /** Options accepted by the {@code tojson} filter. */
  public record JsonOptions(
      Double indent,
      boolean ensureAscii,
      boolean sortKeys,
      JsonSeparators separators,
      int maxOutputLength) {
    /** Returns the default JSON formatting options. */
    public static JsonOptions defaults() {
      return new JsonOptions(null, false, false, null, Integer.MAX_VALUE);
    }
  }

  /**
   * Custom item and key separators for {@code tojson}; either value may be JavaScript undefined.
   */
  public record JsonSeparators(String item, String key) {}

  /**
   * Formats a number for JSON.
   *
   * @param value the number to format
   * @return a JSON number, or {@code null} for a non-finite value
   */
  public static String jsonString(double value) {
    return Double.isFinite(value) ? shortest(value) : "null";
  }

  /**
   * Formats a number using JavaScript's ordinary string form.
   *
   * @param value the number to format
   * @return the JavaScript-compatible representation
   */
  public static String plainString(double value) {
    if (Double.isNaN(value)) {
      return "NaN";
    }
    if (value == Double.POSITIVE_INFINITY) {
      return "Infinity";
    }
    if (value == Double.NEGATIVE_INFINITY) {
      return "-Infinity";
    }
    return shortest(value);
  }

  /** Formats a float runtime value, retaining a fractional zero for integral floats. */
  public static String floatString(double value) {
    if (value % 1 == 0 && Math.abs(value) < 1e21) {
      return new BigDecimal(value).setScale(1, RoundingMode.UNNECESSARY).toPlainString();
    }
    return plainString(value);
  }

  /** Renders a runtime value using the project's JSON-compatible value representation. */
  public static String runtimeJson(Value value, SourceLocation location) {
    return runtimeJson(value, location, false);
  }

  /** Renders a runtime value using the project's JSON-compatible value representation. */
  public static String runtimeJson(
      Value value, SourceLocation location, boolean convertUndefinedToNull) {
    return runtimeJson(value, location, convertUndefinedToNull, JsonOptions.defaults());
  }

  /** Renders a runtime value using the project's JSON-compatible value representation. */
  public static String runtimeJson(
      Value value, SourceLocation location, boolean convertUndefinedToNull, JsonOptions options) {
    var rendered =
        runtimeJsonValue(
            value, location, new IdentityHashMap<>(), convertUndefinedToNull, options, 0);
    return rendered == null ? "undefined" : rendered;
  }

  private static String runtimeJsonValue(
      Value value,
      SourceLocation location,
      IdentityHashMap<Value, Boolean> visiting,
      boolean convertUndefinedToNull,
      JsonOptions options,
      int depth) {
    return switch (value) {
      case Value.NullValue ignored -> "null";
      case Value.UndefinedValue ignored -> convertUndefinedToNull ? "null" : "undefined";
      case Value.DeferredUndefinedValue ignored ->
          throw new TemplateRenderException(
              "Cannot read properties of undefined (reading 'type')", ErrorCategory.TYPE, location);
      case Value.BooleanValue x -> Boolean.toString(x.value());
      case Value.IntegerValue x -> jsonString(x.value());
      case Value.FloatValue x -> jsonString(x.value());
      case Value.StringValue x ->
          x.undefinedBacked() ? null : quote(x.value(), options.ensureAscii());
      case Value.ArrayValue x ->
          jsonArray(x.values(), location, visiting, x, convertUndefinedToNull, options, depth);
      case Value.ObjectValue x ->
          jsonObject(x.values(), location, visiting, x, convertUndefinedToNull, options, depth);
      case Value.TupleValue ignored -> jsonUnsupported("TupleValue", location);
      case Value.KeywordArgumentsValue ignored ->
          jsonUnsupported("KeywordArgumentsValue", location);
      case Value.CallableValue ignored -> jsonUnsupported("FunctionValue", location);
    };
  }

  private static String jsonUnsupported(String type, SourceLocation location) {
    throw new TemplateRenderException(
        "Cannot convert to JSON: " + type, ErrorCategory.TYPE, location);
  }

  private static String jsonArray(
      List<Value> values,
      SourceLocation location,
      IdentityHashMap<Value, Boolean> visiting,
      Value container,
      boolean convertUndefinedToNull,
      JsonOptions options,
      int depth) {
    enterJson(container, visiting, location);
    try {
      var rendered = new ArrayList<String>();
      for (int i = 0; i < values.size(); i++) {
        var item =
            runtimeJsonValue(
                values.get(i), location, visiting, convertUndefinedToNull, options, depth + 1);
        // Array.join preserves an undefined element as an empty slot, rather than removing it.
        rendered.add(item == null ? "" : item);
      }
      return jsonContainer("[", "]", rendered, options, depth, true, location);
    } finally {
      visiting.remove(container);
    }
  }

  private static String jsonObject(
      Map<?, Value> values,
      SourceLocation location,
      IdentityHashMap<Value, Boolean> visiting,
      Value container,
      boolean convertUndefinedToNull,
      JsonOptions options,
      int depth) {
    enterJson(container, visiting, location);
    try {
      var entries = new ArrayList<>(values.entrySet());
      if (options.sortKeys()) {
        var collator = Collator.getInstance(Locale.US);
        entries.sort(
            Comparator.comparing(entry -> jsonObjectSortKey(entry.getKey(), location), collator));
      }
      var rendered = new ArrayList<String>();
      for (var entry : entries) {
        var item =
            runtimeJsonValue(
                entry.getValue(), location, visiting, convertUndefinedToNull, options, depth + 1);
        rendered.add(
            jsonObjectKey(entry.getKey(), options.ensureAscii())
                + keySeparator(options)
                + (item == null ? "undefined" : item));
      }
      return jsonContainer("{", "}", rendered, options, depth, false, location);
    } finally {
      visiting.remove(container);
    }
  }

  private static String jsonObjectKey(Object key, boolean ensureAscii) {
    return key instanceof Value.StringValue string && string.undefinedBacked()
        ? "undefined"
        : quote((String) key, ensureAscii);
  }

  private static String jsonObjectSortKey(Object key, SourceLocation location) {
    if (key instanceof String string) {
      return string;
    }
    // Known upstream divergence: with three or more keys including undefined, V8's sort callback
    // dereferences undefined and throws. Keep this total ordering rather than reproducing that
    // engine accident; it also keeps all entries representable by the Java object model.
    if (key instanceof Value.StringValue string && string.undefinedBacked()) {
      return string.value();
    }
    throw new TemplateRenderException("Object keys must be strings", ErrorCategory.TYPE, location);
  }

  private static String jsonContainer(
      String open,
      String close,
      List<String> values,
      JsonOptions options,
      int depth,
      boolean array,
      SourceLocation location) {
    if (!hasPrettyIndent(options)) {
      return open + String.join(itemSeparator(options), values) + close;
    }
    int indentWidth = jsonIndent(options, location);
    String itemSeparator = itemSeparator(options);
    long basePaddingLength = 1L + (long) indentWidth * depth;
    long childrenPaddingLength = basePaddingLength + indentWidth;
    long renderedLength = 2L + childrenPaddingLength + basePaddingLength;
    if (values.isEmpty() && !array) {
      renderedLength = 2L + basePaddingLength;
    } else {
      for (var value : values) {
        renderedLength = checkedAdd(renderedLength, value.length(), location);
      }
      renderedLength =
          checkedAdd(
              renderedLength,
              (long) Math.max(0, values.size() - 1)
                  * (itemSeparator.length() + childrenPaddingLength),
              location);
    }
    if (renderedLength > options.maxOutputLength()) {
      throw new TemplateRenderException(
          "Maximum render output length exceeded", ErrorCategory.RESOURCE_LIMIT, location);
    }
    String indent = " ".repeat(indentWidth);
    String basePadding = "\n" + indent.repeat(depth);
    String childrenPadding = basePadding + indent;
    if (values.isEmpty() && !array) {
      return open + basePadding + close;
    }
    return open
        + childrenPadding
        + String.join(itemSeparator + childrenPadding, values)
        + basePadding
        + close;
  }

  private static long checkedAdd(long left, long right, SourceLocation location) {
    try {
      return Math.addExact(left, right);
    } catch (ArithmeticException ex) {
      throw new TemplateRenderException(
          "Maximum render output length exceeded", ex, ErrorCategory.RESOURCE_LIMIT, location);
    }
  }

  private static int jsonIndent(JsonOptions options, SourceLocation location) {
    double indent = options.indent();
    if (indent < 0) {
      throw new TemplateRenderException(
          "Invalid count value: " + plainString(indent), ErrorCategory.VALUE, location);
    }
    if (indent > options.maxOutputLength()) {
      throw new TemplateRenderException(
          "Maximum render output length exceeded", ErrorCategory.RESOURCE_LIMIT, location);
    }
    return (int) indent;
  }

  private static boolean hasPrettyIndent(JsonOptions options) {
    return options.indent() != null && options.indent() != 0 && !Double.isNaN(options.indent());
  }

  private static String itemSeparator(JsonOptions options) {
    if (options.separators() != null) {
      if (options.separators().item() == null) {
        return hasPrettyIndent(options) ? "undefined" : ",";
      }
      return options.separators().item();
    }
    return hasPrettyIndent(options) ? "," : ", ";
  }

  private static String keySeparator(JsonOptions options) {
    if (options.separators() == null) {
      return ": ";
    }
    return options.separators().key() == null ? "undefined" : options.separators().key();
  }

  private static void enterJson(
      Value value, IdentityHashMap<Value, Boolean> visiting, SourceLocation location) {
    if (visiting.put(value, Boolean.TRUE) != null) {
      throw new TemplateRenderException(
          "Cannot convert cyclic value to JSON", ErrorCategory.TYPE, location);
    }
  }

  /**
   * Quotes and escapes a string for JSON.
   *
   * @param value the string to quote
   * @return the JSON string literal
   */
  public static String quote(String value) {
    return quote(value, false);
  }

  private static String quote(String value, boolean ensureAscii) {
    var b = new StringBuilder("\"");
    for (int i = 0; i < value.length(); i++) {
      char c = value.charAt(i);
      switch (c) {
        case '\\' -> b.append("\\\\");
        case '"' -> b.append("\\\"");
        case '\b' -> b.append("\\b");
        case '\f' -> b.append("\\f");
        case '\n' -> b.append("\\n");
        case '\r' -> b.append("\\r");
        case '\t' -> b.append("\\t");
        default -> {
          boolean high = Character.isHighSurrogate(c);
          boolean low = Character.isLowSurrogate(c);
          boolean paired =
              high && i + 1 < value.length() && Character.isLowSurrogate(value.charAt(i + 1))
                  || low && i > 0 && Character.isHighSurrogate(value.charAt(i - 1));
          if (c < 0x20 || (high || low) && !paired) {
            b.append(String.format("\\u%04x", (int) c));
          } else {
            b.append(c);
          }
        }
      }
    }
    var quoted = b.append('"').toString();
    if (!ensureAscii) {
      return quoted;
    }
    var ascii = new StringBuilder(quoted.length());
    for (int i = 0; i < quoted.length(); i++) {
      char c = quoted.charAt(i);
      if (c >= 0x7f) {
        ascii.append(String.format("\\u%04x", (int) c));
      } else {
        ascii.append(c);
      }
    }
    return ascii.toString();
  }

  static String shortest(double value) {
    if (value == 0d) {
      return "0";
    }
    var exact = new BigDecimal(value);
    for (int p = 1; p <= 17; p++) {
      var c = exact.round(new MathContext(p, RoundingMode.HALF_EVEN)).stripTrailingZeros();
      if (Double.parseDouble(c.toString()) == value) {
        return format(c);
      }
    }
    throw new IllegalStateException("No JavaScript round-trip decimal for finite double");
  }

  static String format(BigDecimal decimal) {
    if (decimal.signum() == 0) {
      return "0";
    }
    var n = decimal.stripTrailingZeros();
    int e = n.precision() - n.scale() - 1;
    if (e >= -6 && e < 21) {
      return n.toPlainString();
    }
    String d = n.unscaledValue().abs().toString();
    return (n.signum() < 0 ? "-" : "")
        + (d.length() == 1 ? d : d.charAt(0) + "." + d.substring(1))
        + "e"
        + (e >= 0 ? "+" : "")
        + e;
  }
}
