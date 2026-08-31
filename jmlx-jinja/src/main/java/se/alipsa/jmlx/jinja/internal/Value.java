package se.alipsa.jmlx.jinja.internal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.internal.runtime.Environment;

/**
 * Internal, closed template value model. This public type is intentionally in an internal package:
 * it is shared by the parser-facing runtime without becoming part of the exported public API.
 */
@SuppressWarnings("doclint:missing")
public sealed interface Value
    permits Value.UndefinedValue,
        Value.DeferredUndefinedValue,
        Value.NullValue,
        Value.BooleanValue,
        Value.IntegerValue,
        Value.FloatValue,
        Value.StringValue,
        Value.ArrayValue,
        Value.TupleValue,
        Value.ObjectValue,
        Value.KeywordArgumentsValue,
        Value.CallableValue {
  /** Materializes a deferred sequence-filter result when it becomes part of another value. */
  static Value materialize(Value value) {
    return value instanceof DeferredUndefinedValue ? UndefinedValue.INSTANCE : value;
  }

  /** The singleton value used for undefined template expressions. */
  enum UndefinedValue implements Value {
    INSTANCE
  }

  /** Undefined returned by a sequence filter and dereferenced only by a subsequent operation. */
  enum DeferredUndefinedValue implements Value {
    INSTANCE
  }

  /** The singleton value used for the JavaScript null value. */
  enum NullValue implements Value {
    INSTANCE
  }

  /** A JavaScript boolean value. */
  record BooleanValue(boolean value) implements Value {}

  /** A JavaScript integer value. */
  record IntegerValue(double value) implements Value {}

  /** A JavaScript floating-point value. */
  record FloatValue(double value) implements Value {}

  /** A string value, optionally backed by JavaScript {@code undefined}. */
  final class StringValue implements Value {
    private final String value;
    private final boolean undefinedBacked;
    // Values are rebuilt from host data for each render, so this render-local cache is not shared
    // across concurrent renders. Keep it lazy: most string values never access a member builtin.
    private Map<String, CallableValue> builtins;

    public StringValue(String value, boolean undefinedBacked) {
      this.value = Objects.requireNonNull(value, "value");
      this.undefinedBacked = undefinedBacked;
    }

    public StringValue(String value) {
      this(value, false);
    }

    public String value() {
      return value;
    }

    public boolean undefinedBacked() {
      return undefinedBacked;
    }

    public Map<String, CallableValue> builtins() {
      if (builtins == null) {
        builtins = new LinkedHashMap<>();
      }
      return builtins;
    }

    /** Returns the string-shaped value produced by out-of-range string indexing. */
    public static StringValue undefined() {
      return new StringValue("undefined", true);
    }

    @Override
    public boolean equals(Object object) {
      return object instanceof StringValue other
          && undefinedBacked == other.undefinedBacked
          && value.equals(other.value);
    }

    @Override
    public int hashCode() {
      return Objects.hash(value, undefinedBacked);
    }
  }

  /** An immutable JavaScript array value. */
  record ArrayValue(List<Value> values) implements Value {
    public ArrayValue {
      values =
          Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(values, "values")));
    }
  }

  /** An immutable tuple value. */
  record TupleValue(List<Value> values) implements Value {
    public TupleValue {
      values =
          Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(values, "values")));
    }
  }

  /** Mutable by design for template member assignment; never use as a hash-based key. */
  final class ObjectValue implements Value {
    private final Map<Object, Value> values;
    // Values are rebuilt from host data for each render, so this cache is never shared by renders.
    private Map<String, CallableValue> builtins;

    public ObjectValue(Map<?, Value> values) {
      this.values = new LinkedHashMap<>(Objects.requireNonNull(values, "values"));
    }

    public Map<Object, Value> values() {
      return values;
    }

    public Map<String, CallableValue> builtins() {
      if (builtins == null) {
        builtins = new LinkedHashMap<>();
      }
      return builtins;
    }

    @Override
    public boolean equals(Object object) {
      return object instanceof ObjectValue other && values.equals(other.values);
    }

    @Override
    public int hashCode() {
      return values.hashCode();
    }

    @Override
    public String toString() {
      return "ObjectValue[values=" + values + "]";
    }
  }

  /** Object-like call keyword arguments, distinct because they are not JSON-renderable upstream. */
  final class KeywordArgumentsValue implements Value {
    private final Map<String, Value> values;
    // Values are rebuilt from host data for each render, so this cache is never shared by renders.
    private Map<String, CallableValue> builtins;

    public KeywordArgumentsValue(Map<String, Value> values) {
      this.values = new LinkedHashMap<>(Objects.requireNonNull(values, "values"));
    }

    public Map<String, Value> values() {
      return values;
    }

    public Map<String, CallableValue> builtins() {
      if (builtins == null) {
        builtins = new LinkedHashMap<>();
      }
      return builtins;
    }
  }

  /**
   * A callable together with the pinned runtime's observable JavaScript source representation.
   *
   * <p>The runtime renders a {@code FunctionValue} by calling JavaScript {@code String} on its
   * backing function. Preserve that text separately from Java's lambda implementation so filters
   * such as {@code safe} and {@code default} do not turn a callable into a Java-specific marker.
   */
  record CallableValue(Callable callable, String renderedText) implements Value {
    /**
     * Pinned 0.5.9 text produced by globals installed through {@code convertToRuntimeValues}.
     *
     * <p>See {@code upstream/vendor/src/runtime.ts}'s host-function conversion and the pinned
     * {@code dist/index.js}. This is observable through interpolation and JavaScript coercion. When
     * updating {@code upstream/upstream-lock.json}, run the Node oracle with a temporary {@code {{
     * range }}} corpus record and replace this literal with its byte-exact output.
     */
    public static final String CONVERTED_FUNCTION_SOURCE =
        """
        (args, _scope) => {
                const result = input(...args.map((x) => x.value)) ?? null;
                return convertToRuntimeValues(result);
              }\
        """;

    // TODO: Close the remaining callable-rendering parity work in "WP7 — pinned-upstream parity
    // closure" of req/implementation-plan.md (namespace, member builtins, macros, and call blocks).
    /**
     * Explicit marker for Java-created callable forms whose exact upstream function source has not
     * yet been ported (namespace, member builtins, macros, and call blocks).
     */
    public static final String JAVA_FUNCTION_MARKER = "<function>";

    public CallableValue {
      Objects.requireNonNull(callable);
      Objects.requireNonNull(renderedText);
    }

    /** Invokes a callable template value. */
    @FunctionalInterface
    public interface Callable {
      /**
       * Invokes this callable.
       *
       * @param arguments the positional arguments, possibly with a trailing {@link
       *     KeywordArgumentsValue} bag
       * @param hasKeywordArguments whether the caller actually supplied keyword arguments
       * @param location the call-site source location, for diagnostics
       * @param environment the call-site environment (upstream: {@code FunctionValue}'s second
       *     {@code scope} argument), not the environment where the callee was defined. Builtins
       *     ignore it; macros and {@code caller()} use it as the parent of their own new scope.
       */
      Value invoke(
          List<Value> arguments,
          boolean hasKeywordArguments,
          SourceLocation location,
          Environment environment);
    }
  }
}
