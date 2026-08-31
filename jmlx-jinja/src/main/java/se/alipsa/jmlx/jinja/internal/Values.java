package se.alipsa.jmlx.jinja.internal;

import static se.alipsa.jmlx.jinja.internal.Value.ArrayValue;
import static se.alipsa.jmlx.jinja.internal.Value.BooleanValue;
import static se.alipsa.jmlx.jinja.internal.Value.CallableValue;
import static se.alipsa.jmlx.jinja.internal.Value.DeferredUndefinedValue;
import static se.alipsa.jmlx.jinja.internal.Value.FloatValue;
import static se.alipsa.jmlx.jinja.internal.Value.IntegerValue;
import static se.alipsa.jmlx.jinja.internal.Value.KeywordArgumentsValue;
import static se.alipsa.jmlx.jinja.internal.Value.NullValue;
import static se.alipsa.jmlx.jinja.internal.Value.ObjectValue;
import static se.alipsa.jmlx.jinja.internal.Value.StringValue;
import static se.alipsa.jmlx.jinja.internal.Value.TupleValue;
import static se.alipsa.jmlx.jinja.internal.Value.UndefinedValue;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.HostFunction;
import se.alipsa.jmlx.jinja.TemplateRenderException;

/** Converts only the explicitly supported Java boundary types into runtime values. */
@SuppressWarnings("doclint:missing")
public final class Values {
  private static final double MAX_SAFE_INTEGER = 9_007_199_254_740_991d;
  private static final int MAX_MUTABLE_COPIES = 100_000;

  private Values() {}

  /**
   * Converts a supported host value into its runtime representation.
   *
   * @param input the host value
   * @return the corresponding runtime value
   */
  public static Value fromHost(Object input) {
    return fromHost(
        input,
        new IdentityHashMap<>(),
        new IdentityHashMap<>(),
        new IdentityHashMap<>(),
        new CopyBudget());
  }

  /**
   * Converts a runtime value for use as a host-function argument.
   *
   * @param value the runtime value
   * @param converted converted containers, retained for identity preservation
   * @param sourceValues source runtime values, retained for return-value identity preservation
   * @param path the diagnostic path of the argument
   * @return an inert host value
   */
  public static Object toHost(
      Value value,
      IdentityHashMap<Value, Object> converted,
      IdentityHashMap<Object, Value> sourceValues,
      HostPath path) {
    Objects.requireNonNull(value, "value");
    Objects.requireNonNull(converted, "converted");
    Objects.requireNonNull(sourceValues, "sourceValues");
    Objects.requireNonNull(path, "path");
    return toHost(value, converted, sourceValues, path, new IdentityHashMap<>());
  }

  private static Object toHost(
      Value value,
      IdentityHashMap<Value, Object> converted,
      IdentityHashMap<Object, Value> sourceValues,
      HostPath path,
      IdentityHashMap<Value, Boolean> visiting) {
    return switch (value) {
      case UndefinedValue ignored ->
          throw new UndefinedHostValueException("undefined value at " + path.describe());
      case DeferredUndefinedValue ignored ->
          throw new UndefinedHostValueException("undefined value at " + path.describe());
      case NullValue ignored -> null;
      case BooleanValue booleanValue -> booleanValue.value();
      case IntegerValue integerValue -> {
        var hostValue = hostInteger(integerValue.value());
        // Safe integers reconstruct faithfully from a Long. Larger runtime integers and -0 do
        // not, so retain their exact source when a function echoes one.
        yield !(Math.abs(integerValue.value()) <= MAX_SAFE_INTEGER)
                || isNegativeZero(integerValue.value())
            ? sourceValue(hostValue, value, sourceValues)
            : hostValue;
      }
      case FloatValue floatValue -> sourceValue(hostFloat(floatValue.value()), value, sourceValues);
      case StringValue stringValue -> {
        if (stringValue.undefinedBacked())
          throw new UndefinedHostValueException("undefined value at " + path.describe());
        yield stringValue.value();
      }
      case ArrayValue arrayValue -> {
        var existing = converted.get(arrayValue);
        if (existing != null) {
          yield existing;
        }
        requireAcyclicValue(arrayValue, visiting);
        try {
          var values = new ArrayList<Object>(arrayValue.values().size());
          for (int index = 0; index < arrayValue.values().size(); index++) {
            var pathLength = path.length();
            path.appendIndex(index);
            try {
              values.add(
                  toHost(arrayValue.values().get(index), converted, sourceValues, path, visiting));
            } finally {
              path.restore(pathLength);
            }
          }
          var hostValue = Collections.unmodifiableList(values);
          converted.put(arrayValue, hostValue);
          sourceValues.putIfAbsent(hostValue, value);
          yield hostValue;
        } finally {
          visiting.remove(arrayValue);
        }
      }
      case TupleValue tupleValue -> {
        var existing = converted.get(tupleValue);
        if (existing != null) yield existing;
        requireAcyclicValue(tupleValue, visiting);
        try {
          var values = new ArrayList<Object>(tupleValue.values().size());
          for (int index = 0; index < tupleValue.values().size(); index++) {
            var pathLength = path.length();
            path.appendIndex(index);
            try {
              values.add(
                  toHost(tupleValue.values().get(index), converted, sourceValues, path, visiting));
            } finally {
              path.restore(pathLength);
            }
          }
          var hostValue = Collections.unmodifiableList(values);
          converted.put(tupleValue, hostValue);
          yield hostValue;
        } finally {
          visiting.remove(tupleValue);
        }
      }
      case ObjectValue objectValue -> {
        var existing = converted.get(objectValue);
        if (existing != null) {
          yield existing;
        }
        requireAcyclicValue(objectValue, visiting);
        try {
          var values = new LinkedHashMap<String, Object>(objectValue.values().size());
          for (var entry : objectValue.values().entrySet()) {
            if (!(entry.getKey() instanceof String key))
              throw conversion("Map keys must be strings");
            var pathLength = path.length();
            path.appendKey(key);
            try {
              values.put(key, toHost(entry.getValue(), converted, sourceValues, path, visiting));
            } finally {
              path.restore(pathLength);
            }
          }
          var hostValue = Collections.unmodifiableMap(values);
          converted.put(objectValue, hostValue);
          sourceValues.putIfAbsent(hostValue, value);
          yield hostValue;
        } finally {
          visiting.remove(objectValue);
        }
      }
      case KeywordArgumentsValue keywordArgumentsValue -> {
        var existing = converted.get(keywordArgumentsValue);
        if (existing != null) yield existing;
        requireAcyclicValue(keywordArgumentsValue, visiting);
        try {
          var values = new LinkedHashMap<String, Object>(keywordArgumentsValue.values().size());
          for (var entry : keywordArgumentsValue.values().entrySet()) {
            var pathLength = path.length();
            path.appendKey(entry.getKey());
            try {
              values.put(
                  entry.getKey(),
                  toHost(entry.getValue(), converted, sourceValues, path, visiting));
            } finally {
              path.restore(pathLength);
            }
          }
          var hostValue = Collections.unmodifiableMap(values);
          converted.put(keywordArgumentsValue, hostValue);
          sourceValues.putIfAbsent(hostValue, value);
          yield hostValue;
        } finally {
          visiting.remove(keywordArgumentsValue);
        }
      }
      case CallableValue ignored ->
          throw new UndefinedHostValueException("callable value at " + path.describe());
    };
  }

  private static void requireAcyclicValue(Value value, IdentityHashMap<Value, Boolean> visiting) {
    if (visiting.put(value, Boolean.TRUE) != null) {
      throw new IllegalStateException("Runtime value graph contains a cycle");
    }
  }

  private static Object sourceValue(
      Object hostValue, Value sourceValue, IdentityHashMap<Object, Value> sourceValues) {
    // Host functions receive ordinary Java scalar types. An exact scalar argument object returned
    // unchanged is the only signal that it should retain its runtime int/float tag; computed boxed
    // values convert by value and must use FloatResult or IntegerResult when that tag matters.
    //
    // This registration is sound only because nothing JVM-cached ever reaches it: callers route
    // safe-range integers straight through without calling sourceValue(), so only wide Longs
    // (always freshly boxed) and Doubles (never cached by the JVM) arrive here. Registering a
    // cached box would let an unrelated host value with the same identity silently alias to the
    // wrong tag.
    assert !isJvmCachedBox(hostValue) : "Refusing to register a JVM-cached box: " + hostValue;
    sourceValues.putIfAbsent(hostValue, sourceValue);
    return hostValue;
  }

  private static boolean isJvmCachedBox(Object value) {
    return value instanceof Long longValue && Long.valueOf(longValue) == value
        || value instanceof Boolean booleanValue && Boolean.valueOf(booleanValue) == value;
  }

  private static Object hostInteger(double value) {
    if (isNegativeZero(value)) {
      return Double.valueOf(value);
    }
    if (Double.isFinite(value) && value >= -0x1.0p63 && value < 0x1.0p63) {
      return Long.valueOf((long) value);
    }
    return Double.valueOf(value);
  }

  private static Double hostFloat(double value) {
    return Double.valueOf(value);
  }

  public static final class UndefinedHostValueException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private UndefinedHostValueException(String message) {
      super(message);
    }
  }

  /** Defers construction of a diagnostic path string until undefined reaches the host boundary. */
  public static final class HostPath {
    private final StringBuilder description;

    static HostPath argument(int index) {
      return new HostPath("argument " + index);
    }

    private HostPath(String description) {
      this.description = new StringBuilder(description);
    }

    int length() {
      return description.length();
    }

    void appendIndex(int index) {
      description.append('[').append(index).append(']');
    }

    void appendKey(String key) {
      description.append('.').append(Objects.requireNonNull(key, "key"));
    }

    void restore(int length) {
      description.setLength(length);
    }

    String describe() {
      return description.toString();
    }
  }

  /**
   * Converts a host-function result while preserving arguments returned by identity.
   *
   * @param input the host-function result
   * @param sourceValues source runtime values supplied to the host function
   * @return the corresponding runtime value
   */
  public static Value fromHostFunctionReturn(
      Object input, IdentityHashMap<Object, Value> sourceValues) {
    return fromHost(
        input,
        new IdentityHashMap<>(),
        new IdentityHashMap<>(),
        new IdentityHashMap<>(),
        sourceValues,
        true,
        new CopyBudget());
  }

  private static Value fromHost(
      Object input,
      IdentityHashMap<Object, Value> converted,
      IdentityHashMap<Object, Boolean> visiting,
      IdentityHashMap<Value, Boolean> containsMutable,
      CopyBudget copyBudget) {
    return fromHost(input, converted, visiting, containsMutable, null, false, copyBudget);
  }

  private static Value fromHost(
      Object input,
      IdentityHashMap<Object, Value> converted,
      IdentityHashMap<Object, Boolean> visiting,
      IdentityHashMap<Value, Boolean> containsMutable,
      IdentityHashMap<Object, Value> sourceValues,
      boolean allowResultMarkers,
      CopyBudget copyBudget) {
    if (sourceValues != null) {
      var sourceValue = sourceValues.get(input);
      if (sourceValue != null) {
        return sourceValue;
      }
    }
    if (input == null) {
      return NullValue.INSTANCE;
    }
    if (allowResultMarkers && input instanceof HostFunction.FloatResult floatResult) {
      if (!Double.isFinite(floatResult.value())) {
        throw conversion("Float result must be finite: " + floatResult.value());
      }
      return new FloatValue(floatResult.value());
    }
    if (allowResultMarkers && input instanceof HostFunction.IntegerResult integerResult) {
      if (!Double.isFinite(integerResult.value())
          || integerResult.value() != Math.rint(integerResult.value())) {
        throw conversion("Integer result must be finite and integral: " + integerResult.value());
      }
      return new IntegerValue(integerResult.value());
    }
    if (input instanceof String string) {
      return new StringValue(string);
    }
    if (input instanceof Boolean bool) {
      return new BooleanValue(bool);
    }
    if (input instanceof Number number) {
      return numberValue(number);
    }
    if (input.getClass().isArray()) {
      var existing = converted.get(input);
      if (existing != null) {
        return copyMutableObjects(existing, containsMutable, copyBudget);
      }
      requireAcyclic(input, visiting);
      try {
        int length = Array.getLength(input);
        var values = new ArrayList<Value>(length);
        for (int index = 0; index < length; index++) {
          values.add(
              fromHost(
                  Array.get(input, index),
                  converted,
                  visiting,
                  containsMutable,
                  sourceValues,
                  allowResultMarkers,
                  copyBudget));
        }
        var value = new ArrayValue(values);
        converted.put(input, value);
        return value;
      } finally {
        visiting.remove(input);
      }
    }
    if (input instanceof List<?> list) {
      var existing = converted.get(input);
      if (existing != null) {
        return copyMutableObjects(existing, containsMutable, copyBudget);
      }
      requireAcyclic(input, visiting);
      try {
        var values = new ArrayList<Value>(list.size());
        for (Object item : list) {
          values.add(
              fromHost(
                  item,
                  converted,
                  visiting,
                  containsMutable,
                  sourceValues,
                  allowResultMarkers,
                  copyBudget));
        }
        var value = new ArrayValue(values);
        converted.put(input, value);
        return value;
      } finally {
        visiting.remove(input);
      }
    }
    if (input instanceof Map<?, ?> map) {
      var existing = converted.get(input);
      if (existing != null) {
        return copyMutableObjects(existing, containsMutable, copyBudget);
      }
      requireAcyclic(input, visiting);
      try {
        var values = new LinkedHashMap<Object, Value>(map.size());
        for (var entry : map.entrySet()) {
          if (!(entry.getKey() instanceof String key)) {
            throw conversion("Map keys must be strings");
          }
          values.put(
              key,
              fromHost(
                  entry.getValue(),
                  converted,
                  visiting,
                  containsMutable,
                  sourceValues,
                  allowResultMarkers,
                  copyBudget));
        }
        var value = new ObjectValue(values);
        converted.put(input, value);
        return value;
      } finally {
        visiting.remove(input);
      }
    }
    throw conversion("Unsupported host value type: " + input.getClass().getName());
  }

  /** Copies only paths containing mutable objects; immutable DAG portions stay shared. */
  private static Value copyMutableObjects(
      Value value, IdentityHashMap<Value, Boolean> containsMutable, CopyBudget copyBudget) {
    if (!containsMutableObjects(value, containsMutable)) return value;
    return copyMutableObjects(value, containsMutable, new IdentityHashMap<>(), copyBudget);
  }

  /**
   * Each repeated host reference receives an independent mutable copy, retaining its internal DAG.
   */
  private static Value copyMutableObjects(
      Value value,
      IdentityHashMap<Value, Boolean> containsMutable,
      IdentityHashMap<Value, Value> copied,
      CopyBudget copyBudget) {
    var existing = copied.get(value);
    if (existing != null) return existing;
    return switch (value) {
      case ObjectValue objectValue -> {
        copyBudget.charge();
        var result = new ObjectValue(new LinkedHashMap<>(objectValue.values().size()));
        copied.put(value, result);
        for (var entry : objectValue.values().entrySet()) {
          result
              .values()
              .put(
                  entry.getKey(),
                  copyMutableObjects(entry.getValue(), containsMutable, copied, copyBudget));
        }
        yield result;
      }
      case ArrayValue arrayValue -> {
        var copy = new ArrayList<Value>(arrayValue.values().size());
        // ArrayValue snapshots its input, so install the result only after its children are copied.
        boolean changed = false;
        for (var item : arrayValue.values()) {
          var itemCopy = copyMutableObjects(item, containsMutable, copied, copyBudget);
          changed |= itemCopy != item;
          copy.add(itemCopy);
        }
        var result = changed ? new ArrayValue(copy) : arrayValue;
        copied.put(value, result);
        yield result;
      }
      case TupleValue tupleValue -> {
        var copy = new ArrayList<Value>(tupleValue.values().size());
        boolean changed = false;
        for (var item : tupleValue.values()) {
          var itemCopy = copyMutableObjects(item, containsMutable, copied, copyBudget);
          changed |= itemCopy != item;
          copy.add(itemCopy);
        }
        var result = changed ? new TupleValue(copy) : tupleValue;
        copied.put(value, result);
        yield result;
      }
      default -> value;
    };
  }

  private static final class CopyBudget {
    private int copies;

    void charge() {
      if (++copies > MAX_MUTABLE_COPIES) {
        throw conversion("Host value graph is too large after mutable copy isolation");
      }
    }
  }

  private static boolean containsMutableObjects(Value value, IdentityHashMap<Value, Boolean> memo) {
    var existing = memo.get(value);
    if (existing != null) return existing;
    boolean result =
        switch (value) {
          case ObjectValue ignored -> true;
          case ArrayValue arrayValue ->
              arrayValue.values().stream().anyMatch(item -> containsMutableObjects(item, memo));
          case TupleValue tupleValue ->
              tupleValue.values().stream().anyMatch(item -> containsMutableObjects(item, memo));
          default -> false;
        };
    memo.put(value, result);
    return result;
  }

  private static Value numberValue(Number number) {
    if ((number instanceof Double || number instanceof Float)
        && !Double.isFinite(number.doubleValue())) {
      throw conversion("Number must be finite: " + Double.toString(number.doubleValue()));
    }
    final String text = number.toString();
    if (text == null) {
      throw conversion(
          "Number does not have a decimal representation: " + number.getClass().getName());
    }
    final BigDecimal inputDecimal;
    try {
      inputDecimal = new BigDecimal(text).stripTrailingZeros();
    } catch (NumberFormatException exception) {
      throw conversion(
          "Number does not have a decimal representation: " + number.getClass().getName());
    }
    final String canonical = JsFormat.format(inputDecimal);

    final double value =
        number instanceof Double
            ? number.doubleValue()
            : Double.parseDouble(inputDecimal.toString());
    if (!Double.isFinite(value)) {
      throw conversion("Number must be finite: " + canonical);
    }
    boolean integral = inputDecimal.scale() <= 0;
    boolean floatingPointInput = number instanceof Float || number instanceof Double;
    if (integral && !floatingPointInput && Math.abs(value) > MAX_SAFE_INTEGER) {
      throw conversion("Integer is outside the JavaScript safe-integer range: " + canonical);
    }
    if (!floatingPointInput && !JsFormat.shortest(value).equals(canonical)) {
      throw conversion("Number is not representable as a JavaScript number: " + canonical);
    }
    if (integral) {
      return new IntegerValue(normalizeHostZero(value));
    }
    return new FloatValue(normalizeHostZero(value));
  }

  /**
   * Host values have no observable signed zero, unlike runtime arithmetic such as {@code 1 / -0}.
   */
  private static double normalizeHostZero(double value) {
    return value == 0d ? 0d : value;
  }

  private static boolean isNegativeZero(double value) {
    return Double.doubleToRawLongBits(value) == Double.doubleToRawLongBits(-0d);
  }

  private static void requireAcyclic(Object input, IdentityHashMap<Object, Boolean> ancestors) {
    if (ancestors.put(input, Boolean.TRUE) != null) {
      throw conversion("Host value graph contains a cycle");
    }
  }

  private static TemplateRenderException conversion(String message) {
    return new TemplateRenderException(message, ErrorCategory.HOST_CONVERSION, null);
  }
}
