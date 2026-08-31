package se.alipsa.jmlx.jinja.internal.runtime;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.internal.Value;

// Public so Value.CallableValue.Callable (package se.alipsa.jmlx.jinja.internal) can name this type
// in its invoke(...) signature — module-info.java exports only se.alipsa.jmlx.jinja, so this is not
// a
// public-API change. The constructor stays package-private: nothing outside internal.runtime can
// construct one.
/** Per-render lexical scope of template runtime values. */
public final class Environment {
  private final Environment parent;
  private final Map<String, Value> variables = new LinkedHashMap<>();

  Environment(Environment parent) {
    this.parent = parent;
    variables.put(
        "namespace",
        new Value.CallableValue(this::namespace, Value.CallableValue.JAVA_FUNCTION_MARKER));
  }

  void set(String name, Value value) {
    if (variables.containsKey(name))
      throw new IllegalStateException("Variable already declared: " + name);
    variables.put(name, value);
  }

  void setVariable(String name, Value value) {
    variables.put(name, Value.materialize(value));
  }

  Value lookupVariable(String name) {
    for (var env = this; env != null; env = env.parent) {
      var value = env.variables.get(name);
      if (value != null) return value;
    }
    return Value.UndefinedValue.INSTANCE;
  }

  private Value namespace(
      List<Value> args, boolean keywords, SourceLocation location, Environment environment) {
    if (args.size() > 1
        || (!args.isEmpty()
            && !(args.get(0) instanceof Value.ObjectValue)
            && !(args.get(0) instanceof Value.KeywordArgumentsValue)))
      throw new TemplateRenderException(
          "`namespace` expects either zero arguments or a single object argument",
          ErrorCategory.TYPE,
          location);
    return args.isEmpty() ? new Value.ObjectValue(new LinkedHashMap<>()) : args.get(0);
  }
}
