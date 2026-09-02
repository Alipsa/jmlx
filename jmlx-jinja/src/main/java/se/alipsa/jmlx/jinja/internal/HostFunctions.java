package se.alipsa.jmlx.jinja.internal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.HostFunction;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateRenderException;

/**
 * Dispatches the closed host-function boundary after the interpreter has resolved the callable.
 *
 * <p>Name lookup and non-callable diagnostics, including context-versus-global shadowing, remain
 * interpreter-owned. Argument conversion is intentionally per call; render-scoped accounting and
 * memoization arrive with the WP4 render environment and budget.
 */
@SuppressWarnings("doclint:missing")
public final class HostFunctions {
  private HostFunctions() {}

  /**
   * Invokes a host function after converting its template arguments to host values.
   *
   * @param name the template-visible function name
   * @param function the host function to invoke
   * @param positionalArguments the positional template arguments
   * @param hasKeywordArguments whether the call has keyword arguments
   * @param location the source location of the call
   * @return the converted result
   */
  public static Value invoke(
      String name,
      HostFunction function,
      List<Value> positionalArguments,
      boolean hasKeywordArguments,
      SourceLocation location) {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(function, "function");
    Objects.requireNonNull(positionalArguments, "positionalArguments");
    Objects.requireNonNull(location, "location");
    if (hasKeywordArguments) {
      throw failure(
          "Host function '" + name + "' does not accept keyword arguments", null, location);
    }

    // {% call %} unconditionally appends an empty keyword-arguments bag to every callee's
    // argument list, including non-macro callees, to match upstream's own call-statement-vs-
    // call-expression asymmetry (see Interpreter.evaluateCallStatement). Host functions have no
    // such upstream contract and no template-visible way to construct a trailing empty
    // KeywordArgumentsValue on their own, so strip it here rather than leak an interpreter-
    // internal calling convention across the public HostFunction boundary.
    if (!positionalArguments.isEmpty()
        && positionalArguments.get(positionalArguments.size() - 1)
            instanceof Value.KeywordArgumentsValue kwargs
        && kwargs.values().isEmpty()) {
      positionalArguments = positionalArguments.subList(0, positionalArguments.size() - 1);
    }
    var convertedValues = new IdentityHashMap<Value, Object>();
    var sourceValues = new IdentityHashMap<Object, Value>();
    var arguments =
        hostArguments(positionalArguments, name, location, convertedValues, sourceValues);
    final Object result;
    try {
      result = function.apply(arguments);
    } catch (RuntimeException exception) {
      throw failure("Host function '" + name + "' failed", exception, location);
    }
    try {
      return Values.fromHostFunctionReturn(result, sourceValues);
    } catch (RuntimeException exception) {
      throw failure(
          "Host function '" + name + "' returned an unsupported value", exception, location);
    }
  }

  private static List<Object> hostArguments(
      List<Value> positionalArguments,
      String name,
      SourceLocation location,
      IdentityHashMap<Value, Object> convertedValues,
      IdentityHashMap<Object, Value> sourceValues) {
    var arguments = new ArrayList<Object>(positionalArguments.size());
    for (int index = 0; index < positionalArguments.size(); index++) {
      try {
        var argument =
            Objects.requireNonNull(
                positionalArguments.get(index), "positionalArguments[" + index + "]");
        arguments.add(
            Values.toHost(
                argument, convertedValues, sourceValues, Values.HostPath.argument(index)));
      } catch (Values.UndefinedHostValueException exception) {
        throw failure(
            "Host function '" + name + "' cannot receive " + exception.getMessage(),
            null,
            location);
      } catch (RuntimeException exception) {
        throw failure(
            "Host function '" + name + "' cannot receive argument " + index, exception, location);
      }
    }
    return Collections.unmodifiableList(arguments);
  }

  private static TemplateRenderException failure(
      String message, Throwable cause, SourceLocation location) {
    return cause == null
        ? new TemplateRenderException(message, ErrorCategory.HOST_FUNCTION, location)
        : new TemplateRenderException(message, cause, ErrorCategory.HOST_FUNCTION, location);
  }
}
