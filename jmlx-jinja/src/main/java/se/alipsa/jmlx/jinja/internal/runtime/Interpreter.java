package se.alipsa.jmlx.jinja.internal.runtime;

import java.io.IOException;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.BiPredicate;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.RenderOptions;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.internal.HostFunctions;
import se.alipsa.jmlx.jinja.internal.JsFormat;
import se.alipsa.jmlx.jinja.internal.Value;
import se.alipsa.jmlx.jinja.internal.ast.Expression;
import se.alipsa.jmlx.jinja.internal.ast.Statement;
import se.alipsa.jmlx.jinja.internal.util.JsSlice;
import se.alipsa.jmlx.jinja.internal.util.PosixStrftime;

/** Internal evaluator entry point. */
@SuppressWarnings("doclint:missing")
public final class Interpreter {
  private Interpreter() {}

  /** Private transport for loop control crossing a template callable boundary. */
  private static final class LoopControl extends RuntimeException {
    private final ExecResult result;
    private final SourceLocation location;

    LoopControl(ExecResult result, SourceLocation location) {
      super(null, null, false, false);
      this.result = result;
      this.location = location;
    }

    ExecResult result() {
      return result;
    }

    SourceLocation location() {
      return location;
    }
  }

  /**
   * Renders a parsed program using the supplied context and options.
   *
   * @param program the parsed program
   * @param context the runtime context object
   * @param options render-time options
   * @param output destination for rendered text
   */
  public static void render(
      Statement.Program program,
      Value.ObjectValue context,
      RenderOptions options,
      Appendable output) {
    var env = new Environment(null);
    var budget = new RenderBudget(options);
    try {
      seed(env, options, program.location(), budget);
      for (var e : context.values().entrySet()) {
        if (e.getKey() instanceof String key) env.set(key, e.getValue());
      }
    } catch (IllegalStateException ex) {
      throw new TemplateRenderException(ex.getMessage(), ErrorCategory.VALUE, program.location());
    }
    ExecResult result;
    try {
      result = evaluateBlock(program.body(), env, budget);
    } catch (LoopControl control) {
      throw new TemplateRenderException(
          "break or continue outside a for loop", ErrorCategory.SYNTAX, control.location());
    } catch (StackOverflowError overflow) {
      // RenderBudget.maxMacroDepth bounds macro/call-block *invocation* count, not total
      // interpreter recursion depth: a recursive macro whose body itself nests control-flow
      // constructs (nested {% for %}/{% if %}/{% call %}) consumes several interpreter stack
      // frames per invocation, so a deliberately or accidentally deep-nested body can still
      // exhaust the native JVM stack well below maxMacroDepth invocations. This catch is the
      // backstop that keeps that case inside this project's documented contract (render() only
      // throws TemplateRenderException) instead of letting a bare Error escape to a caller
      // sandboxing untrusted templates. Chaining the original StackOverflowError as the cause
      // matters here specifically: this catch cannot distinguish "template recursed too deep"
      // from a genuine interpreter bug (e.g. an AST cycle) that also exhausts the stack, so the
      // original stack trace must survive for diagnosis rather than being discarded.
      throw new TemplateRenderException(
          "Maximum interpreter recursion depth exceeded",
          overflow,
          ErrorCategory.RESOURCE_LIMIT,
          program.location());
    }
    if (!(result instanceof ExecResult.Normal normal))
      throw new TemplateRenderException(
          "break or continue outside a for loop", ErrorCategory.SYNTAX, program.location());
    try {
      output.append(normal.output());
    } catch (IOException ex) {
      throw new TemplateRenderException(
          "Unable to write rendered output", ex, ErrorCategory.OUTPUT, program.location());
    }
  }

  private static void seed(
      Environment env, RenderOptions o, SourceLocation l, RenderBudget budget) {
    env.set("false", new Value.BooleanValue(false));
    env.set("true", new Value.BooleanValue(true));
    env.set("none", Value.NullValue.INSTANCE);
    env.set("False", new Value.BooleanValue(false));
    env.set("True", new Value.BooleanValue(true));
    env.set("None", Value.NullValue.INSTANCE);
    env.set(
        "range",
        new Value.CallableValue(
            (a, k, x, s) -> range(a, k, x, budget), Value.CallableValue.CONVERTED_FUNCTION_SOURCE));
    env.set(
        "raise_exception",
        new Value.CallableValue(
            (a, k, x, s) -> raise(a, k, x), Value.CallableValue.CONVERTED_FUNCTION_SOURCE));
    env.set(
        "strftime_now",
        new Value.CallableValue(
            (a, k, x, s) -> strftime(a, k, x, o), Value.CallableValue.CONVERTED_FUNCTION_SOURCE));
    for (var e : o.hostFunctions().entrySet())
      env.set(
          e.getKey(),
          new Value.CallableValue(
              (a, k, x, s) -> HostFunctions.invoke(e.getKey(), e.getValue(), a, k, x),
              Value.CallableValue.CONVERTED_FUNCTION_SOURCE));
  }

  static ExecResult evaluateBlock(List<Statement> body, Environment env, RenderBudget budget) {
    var out = new StringBuilder();
    for (var node : body) {
      budget.chargeStep(node.location());
      var result = evaluateStatement(node, env, budget);
      if (!(result instanceof ExecResult.Normal n)) return result;
      out.append(n.output());
    }
    return new ExecResult.Normal(out.toString());
  }

  static ExecResult evaluateStatement(Statement n, Environment env, RenderBudget budget) {
    return switch (n) {
      case Statement.Program p -> evaluateBlock(p.body(), env, budget);
      case Statement.If i -> evaluateIf(i, env, budget);
      case Statement.For f -> evaluateFor(f, env, budget);
      case Statement.Break ignored -> ExecResult.Break.INSTANCE;
      case Statement.Continue ignored -> ExecResult.Continue.INSTANCE;
      case Statement.SetStatement s -> evaluateSet(s, env, budget);
      case Statement.Comment ignored -> new ExecResult.Normal("");
      case Statement.Macro m -> evaluateMacro(m, env, budget);
      case Statement.FilterStatement f -> evaluateFilterStatement(f, env, budget);
      case Statement.CallStatement c -> evaluateCallStatement(c, env, budget);
      case Expression e -> {
        var v = evaluateExpression(e, env, budget);
        String t =
            v instanceof Value.NullValue || v instanceof Value.UndefinedValue
                ? ""
                : renderText(v, e.location());
        budget.chargeOutput(t.length(), e.location());
        yield new ExecResult.Normal(t);
      }
    };
  }

  static Value evaluateExpression(Expression n, Environment env, RenderBudget budget) {
    return switch (n) {
      case Expression.Identifier x -> env.lookupVariable(x.value());
      case Expression.IntegerLiteral x -> new Value.IntegerValue(x.value());
      case Expression.FloatLiteral x -> new Value.FloatValue(x.value());
      case Expression.StringLiteral x -> new Value.StringValue(x.value());
      case Expression.ArrayLiteral x -> new Value.ArrayValue(values(x.value(), env, budget));
      case Expression.TupleLiteral x -> new Value.TupleValue(values(x.value(), env, budget));
      case Expression.ObjectLiteral x -> object(x, env, budget);
      case Expression.MemberExpression x -> member(x, env, budget);
      case Expression.CallExpression x -> call(x, env, budget);
      case Expression.SelectExpression x ->
          truthy(evaluateExpression(x.test(), env, budget), x.test().location())
              ? evaluateExpression(x.lhs(), env, budget)
              : Value.UndefinedValue.INSTANCE;
      case Expression.BinaryExpression x -> binary(x, env, budget);
      case Expression.UnaryExpression x -> unary(x, env, budget);
      case Expression.FilterExpression x -> filter(x, env, budget);
      case Expression.TestExpression x -> test(x, env, budget);
      case Expression.Ternary x -> ternary(x, env, budget);
      // SliceExpression/KeywordArgumentExpression/SpreadExpression are the only three sealed
      // Expression cases with no real evaluation logic here, and that is not a gap: the parser
      // constructs each of them in exactly one place, and every structural position they can
      // occupy is intercepted before it can ever reach this generic dispatch.
      // Parser.parseMemberExpressionArgumentsList is the only site that builds a SliceExpression,
      // and it always becomes the `property` of a computed MemberExpression; member()
      // special-cases `n.computed() && n.property() instanceof SliceExpression` before ever
      // calling evaluateExpression on that property. Parser.parseArgumentsList is the only site
      // that builds a KeywordArgumentExpression or SpreadExpression, feeding a CallExpression's
      // argument list, a Macro's own parameter list, or a {% call %} block's caller parameter
      // list; none of those three consumers ever passes the kwarg/spread node itself to
      // evaluateExpression — evaluateArguments recurses only into keyword.value()/
      // spread.argument(), and the macro and caller-parameter loops reject unexpected node types
      // outright. No valid AST from this parser can hand one of these three node types to this
      // switch directly, so — matching renderText's identical NullValue/UndefinedValue arms below
      // — these throw AssertionError rather than a template-facing exception: reaching here is an
      // interpreter bug, not a template author's mistake.
      case Expression.SliceExpression x -> throw unreachable(x);
      case Expression.KeywordArgumentExpression x -> throw unreachable(x);
      case Expression.SpreadExpression x -> throw unreachable(x);
    };
  }

  private static Value ternary(
      Expression.Ternary expression, Environment env, RenderBudget budget) {
    return truthy(
            evaluateExpression(expression.condition(), env, budget),
            expression.condition().location())
        ? evaluateExpression(expression.trueExpr(), env, budget)
        : evaluateExpression(expression.falseExpr(), env, budget);
  }

  private static AssertionError unreachable(Expression n) {
    return new AssertionError(
        "unreachable: " + n.getClass().getSimpleName() + " at " + n.location());
  }

  private static Value binary(
      Expression.BinaryExpression expression, Environment env, RenderBudget budget) {
    var left = evaluateExpression(expression.left(), env, budget);
    var operator = expression.operator().value();
    if (operator.equals("and"))
      return truthy(left, expression.left().location())
          ? evaluateExpression(expression.right(), env, budget)
          : left;
    if (operator.equals("or"))
      return truthy(left, expression.left().location())
          ? left
          : evaluateExpression(expression.right(), env, budget);

    var right = evaluateExpression(expression.right(), env, budget);
    deferredValue(left, "value", expression.location());
    deferredValue(right, "value", expression.location());
    if (operator.equals("==") || operator.equals("!=")) {
      boolean equal = JsOperations.looseEquals(left, right);
      return new Value.BooleanValue(operator.equals("==") ? equal : !equal);
    }
    if (left instanceof Value.UndefinedValue || right instanceof Value.UndefinedValue) {
      if (right instanceof Value.UndefinedValue
          && (operator.equals("in") || operator.equals("not in")))
        return new Value.BooleanValue(operator.equals("not in"));
      throw operatorUndefined(operator, expression.location());
    }
    if (left instanceof Value.NullValue || right instanceof Value.NullValue)
      throw operatorNull(expression.location());
    if (operator.equals("~"))
      return new Value.StringValue(
          JsOperations.payloadText(left, expression.location())
              + JsOperations.payloadText(right, expression.location()));
    if (JsOperations.numeric(left) && JsOperations.numeric(right)) {
      if (operator.equals("+")) return JsOperations.add(left, right, expression.location());
      if (operator.equals("-")
          || operator.equals("*")
          || operator.equals("/")
          || operator.equals("%")) return JsOperations.arithmetic(operator, left, right);
      if (operator.equals("<")
          || operator.equals(">")
          || operator.equals("<=")
          || operator.equals(">="))
        return new Value.BooleanValue(JsOperations.compare(operator, left, right));
    } else if (arrayLike(left) && arrayLike(right)) {
      if (operator.equals("+"))
        return JsOperations.concatenate(arrayValues(left), arrayValues(right));
    } else if (arrayLike(right)
        && !(left instanceof Value.ArrayValue || left instanceof Value.TupleValue)) {
      if (operator.equals("in") || operator.equals("not in")) {
        boolean present = JsOperations.contains(left, new Value.ArrayValue(arrayValues(right)));
        return new Value.BooleanValue(operator.equals("in") ? present : !present);
      }
    }
    if (left instanceof Value.StringValue || right instanceof Value.StringValue) {
      if (operator.equals("+")) return JsOperations.add(left, right, expression.location());
    }
    if (left instanceof Value.StringValue string && right instanceof Value.StringValue other) {
      if (operator.equals("in") || operator.equals("not in")) {
        if (other.undefinedBacked())
          throw filterType(
              "Cannot read properties of undefined (reading 'includes')", expression.location());
        boolean present = JsOperations.contains(string, other);
        return new Value.BooleanValue(operator.equals("in") ? present : !present);
      }
    }
    if (left instanceof Value.StringValue string && right instanceof Value.ObjectValue object) {
      if (operator.equals("in") || operator.equals("not in")) {
        boolean present = JsOperations.contains(string, object);
        return new Value.BooleanValue(operator.equals("in") ? present : !present);
      }
    }
    if (left instanceof Value.StringValue string
        && right instanceof Value.KeywordArgumentsValue object) {
      if (operator.equals("in") || operator.equals("not in")) {
        boolean present = object.values().containsKey(string.value());
        return new Value.BooleanValue(operator.equals("in") ? present : !present);
      }
    }
    throw operatorUnsupportedTypes(operator, left, right, expression.location());
  }

  private static boolean arrayLike(Value value) {
    return value instanceof Value.ArrayValue || value instanceof Value.TupleValue;
  }

  private static List<Value> arrayValues(Value value) {
    return value instanceof Value.ArrayValue array
        ? array.values()
        : ((Value.TupleValue) value).values();
  }

  private static Value unary(
      Expression.UnaryExpression expression, Environment env, RenderBudget budget) {
    var argument = evaluateExpression(expression.argument(), env, budget);
    deferredValue(argument, "value", expression.location());
    if (expression.operator().value().equals("not"))
      return new Value.BooleanValue(!JsOperations.rawTruthy(argument));
    throw operatorUnsupportedUnary(expression.operator().value(), argument, expression.location());
  }

  private record NamedArguments(String name, List<Value> positional, Map<String, Value> keywords) {}

  private static Value filter(
      Expression.FilterExpression expression, Environment env, RenderBudget budget) {
    var operand = evaluateExpression(expression.operand(), env, budget);
    return applyFilter(operand, expression.filter(), env, budget, expression.location());
  }

  private static Value applyFilter(
      Value operand,
      Expression filterNode,
      Environment env,
      RenderBudget budget,
      SourceLocation location) {
    if (filterNode instanceof Expression.Identifier identifier) {
      if (!absorbsDeferredOperand(identifier.value())) deferredValue(operand, location);
      return applyBareFilter(operand, identifier.value(), location, budget);
    }
    if (!(filterNode instanceof Expression.CallExpression call)
        || !(call.callee() instanceof Expression.Identifier identifier))
      throw filterType("Unknown filter: " + filterNode.getClass().getSimpleName(), location);
    if (!absorbsDeferredOperand(identifier.value())) deferredValue(operand, location);
    return applyCallFilter(operand, identifier.value(), call, env, budget, location);
  }

  private static Value applyBareFilter(
      Value operand, String name, SourceLocation location, RenderBudget budget) {
    var filter = new NamedArguments(name, List.of(), Map.of());
    return switch (filter.name()) {
      case "safe" -> operand;
      case "tojson" -> filterToJson(operand, filter, location, budget);
      case "default" -> throw unknownBareFilter(filter.name(), operand, location);
      case "length" -> filterLength(operand, location);
      case "lower" ->
          filterString(operand, filter, location, value -> value.toLowerCase(Locale.ROOT));
      case "upper" ->
          filterString(operand, filter, location, value -> value.toUpperCase(Locale.ROOT));
      case "trim" -> filterString(operand, filter, location, JsOperations::trimEcmaWhitespace);
      // Upstream's bare string join returns its operand without reading .value. The call form
      // (including |join()) instead builds an array from .value, which matters for
      // undefined-backed strings.
      case "join" ->
          operand instanceof Value.StringValue ? operand : filterJoin(operand, filter, location);
      case "list" -> filterList(operand, location);
      case "items" -> filterItems(operand, location);
      case "first" -> filterFirstLast(operand, filter, location, false);
      case "last" -> filterFirstLast(operand, filter, location, true);
      case "reverse" -> filterReverse(operand, filter, location);
      case "unique" -> filterUnique(operand, filter, location);
      case "sort" -> filterSort(operand, filter, location);
      case "map" -> throw unknownBareFilter(filter.name(), operand, location);
      case "string" -> filterToString(operand, location);
      case "title" -> filterString(operand, filter, location, Interpreter::titleCase);
      case "capitalize" -> filterString(operand, filter, location, Interpreter::capitalize);
      case "abs" -> filterAbs(operand, filter, location);
      case "bool" -> filterBool(operand, filter, location);
      case "indent" -> filterIndent(operand, filter, location, budget);
      case "replace" -> throw unknownBareFilter(filter.name(), operand, location);
      case "keys", "values", "dictsort", "get" -> filterObjectBuiltin(operand, filter, location);
      case "selectattr", "rejectattr" -> throw unknownBareFilter(filter.name(), operand, location);
      case "int" -> filterNumber(operand, filter, location, true);
      case "float" -> filterNumber(operand, filter, location, false);
      default -> throw unknownBareFilter(filter.name(), operand, location);
    };
  }

  /** Filters whose upstream implementation returns the operand before reading its properties. */
  private static boolean absorbsDeferredOperand(String name) {
    return name.equals("default") || name.equals("safe");
  }

  private static Value applyCallFilter(
      Value operand,
      String name,
      Expression.CallExpression call,
      Environment env,
      RenderBudget budget,
      SourceLocation location) {
    return switch (name) {
      case "tojson" ->
          filterToJson(operand, filterArguments(name, call, env, budget), location, budget);
      case "int", "float" -> {
        var arguments = filterArguments(name, call, env, budget);
        if (operand instanceof Value.IntegerValue || operand instanceof Value.FloatValue)
          yield operand;
        yield filterNumber(operand, arguments, location, name.equals("int"));
      }
      case "default" -> filterDefault(operand, filterArguments(name, call, env, budget), location);
      case "join" -> {
        if (!(operand instanceof Value.ArrayValue
            || operand instanceof Value.TupleValue
            || operand instanceof Value.StringValue)) throw filterReceiver(name, operand, location);
        yield filterJoin(operand, filterArguments(name, call, env, budget), location);
      }
      default -> applyCallFilterByReceiver(operand, name, call, env, budget, location);
    };
  }

  private static Value applyCallFilterByReceiver(
      Value operand,
      String name,
      Expression.CallExpression call,
      Environment env,
      RenderBudget budget,
      SourceLocation location) {
    if (operand instanceof Value.ArrayValue array)
      return applyArrayCallFilter(array.values(), operand, name, call, env, budget, location);
    if (operand instanceof Value.TupleValue tuple)
      return applyArrayCallFilter(tuple.values(), operand, name, call, env, budget, location);
    if (operand instanceof Value.StringValue)
      return switch (name) {
        case "indent" ->
            filterIndent(operand, filterArguments(name, call, env, budget), location, budget);
        case "replace" ->
            filterReplace(operand, filterArguments(name, call, env, budget), location);
        default -> throw unknownCallFilter(name, operand, location);
      };
    if (operand instanceof Value.ObjectValue object) {
      if (name.equals("get")
          || name.equals("items")
          || name.equals("keys")
          || name.equals("values")
          || name.equals("dictsort"))
        return filterObjectBuiltin(object, filterArguments(name, call, env, budget), location);
      throw unknownCallFilter(name, operand, location);
    }
    if (operand instanceof Value.KeywordArgumentsValue object) {
      if (name.equals("get")
          || name.equals("items")
          || name.equals("keys")
          || name.equals("values")
          || name.equals("dictsort"))
        return filterObjectBuiltin(object, filterArguments(name, call, env, budget), location);
      throw unknownCallFilter(name, operand, location);
    }
    throw filterReceiver(name, operand, location);
  }

  private static Value applyArrayCallFilter(
      List<Value> values,
      Value operand,
      String name,
      Expression.CallExpression call,
      Environment env,
      RenderBudget budget,
      SourceLocation location) {
    return switch (name) {
      case "sort" -> filterSort(operand, filterArguments(name, call, env, budget), location);
      case "map" -> filterMap(operand, filterArguments(name, call, env, budget), location);
      case "selectattr" -> filterSelectAttrCall(values, name, call, env, budget, location, true);
      case "rejectattr" -> filterSelectAttrCall(values, name, call, env, budget, location, false);
      default -> throw unknownCallFilter(name, operand, location);
    };
  }

  private static Value filterSelectAttrCall(
      List<Value> values,
      String name,
      Expression.CallExpression call,
      Environment env,
      RenderBudget budget,
      SourceLocation location,
      boolean select) {
    if (values.isEmpty() && call.args().isEmpty()) return new Value.ArrayValue(List.of());
    for (var item : values)
      if (!(item instanceof Value.ObjectValue))
        throw filterType("`" + name + "` can only be applied to array of objects", location);
    for (var argument : call.args())
      if (!(argument instanceof Expression.StringLiteral))
        throw new TemplateRenderException(
            "arguments of `" + name + "` must be strings", ErrorCategory.TYPE, location);
    return filterSelectAttr(values, filterArguments(name, call, env, budget), location, select);
  }

  private static NamedArguments filterArguments(
      String name, Expression.CallExpression call, Environment env, RenderBudget budget) {
    var evaluated = evaluateArguments(call.args(), env, budget);
    return new NamedArguments(name, evaluated.positional(), evaluated.keywords());
  }

  private static Value test(
      Expression.TestExpression expression, Environment env, RenderBudget budget) {
    var operand = evaluateExpression(expression.operand(), env, budget);
    String name = expression.test().value();
    if (testReadsDeferredValue(name)) deferredValue(operand, expression.location());
    boolean result = namedTest(name, operand, null, expression.location());
    return new Value.BooleanValue(expression.negate() ? !result : result);
  }

  /** Tests implemented as {@code instanceof} checks do not read an upstream value property. */
  private static boolean testReadsDeferredValue(String name) {
    return switch (name) {
      case "callable", "integer", "mapping", "number", "sequence" -> false;
      default -> true;
    };
  }

  private static Value filterToJson(
      Value operand, NamedArguments filter, SourceLocation location, RenderBudget budget) {
    Value indent = absentFilterArgument(filter.keywords().get("indent"), Value.NullValue.INSTANCE);
    if (!(indent instanceof Value.NullValue || indent instanceof Value.IntegerValue))
      throw new TemplateRenderException(
          "If set, indent must be a number", ErrorCategory.TYPE, location);
    Double indentValue = indent instanceof Value.IntegerValue number ? number.value() : null;

    Value ensureAscii =
        absentFilterArgument(filter.keywords().get("ensure_ascii"), new Value.BooleanValue(false));
    if (!(ensureAscii instanceof Value.BooleanValue ensureAsciiValue))
      throw new TemplateRenderException(
          "If set, ensure_ascii must be a boolean", ErrorCategory.TYPE, location);
    Value sortKeys =
        absentFilterArgument(filter.keywords().get("sort_keys"), new Value.BooleanValue(false));
    if (!(sortKeys instanceof Value.BooleanValue sortKeysValue))
      throw new TemplateRenderException(
          "If set, sort_keys must be a boolean", ErrorCategory.TYPE, location);
    Value separators =
        absentFilterArgument(filter.keywords().get("separators"), Value.NullValue.INSTANCE);
    JsFormat.JsonSeparators separatorValues = null;
    if (separators instanceof Value.ArrayValue array)
      separatorValues = jsonSeparators(array.values(), location);
    else if (separators instanceof Value.TupleValue tuple)
      separatorValues = jsonSeparators(tuple.values(), location);
    else if (!(separators instanceof Value.NullValue))
      throw new TemplateRenderException(
          "If set, separators must be a tuple of two strings", ErrorCategory.TYPE, location);
    return new Value.StringValue(
        JsFormat.runtimeJson(
            operand,
            location,
            true,
            new JsFormat.JsonOptions(
                indentValue,
                ensureAsciiValue.value(),
                sortKeysValue.value(),
                separatorValues,
                budget.remainingOutputLength())));
  }

  private static JsFormat.JsonSeparators jsonSeparators(
      List<Value> values, SourceLocation location) {
    if (values.size() != 2
        || !(values.get(0) instanceof Value.StringValue first)
        || !(values.get(1) instanceof Value.StringValue second))
      throw new TemplateRenderException(
          "separators must be a tuple of two strings", ErrorCategory.TYPE, location);
    return new JsFormat.JsonSeparators(
        first.undefinedBacked() ? null : first.value(),
        second.undefinedBacked() ? null : second.value());
  }

  private static Value filterDefault(
      Value operand, NamedArguments filter, SourceLocation location) {
    var fallback =
        filter.positional().isEmpty()
            ? new Value.StringValue("")
            : absentFilterArgument(filter.positional().get(0), new Value.StringValue(""));
    Value booleanFlag =
        filter.positional().size() > 1
            ? absentFilterArgument(filter.positional().get(1), new Value.BooleanValue(false))
            : absentFilterArgument(filter.keywords().get("boolean"), new Value.BooleanValue(false));
    if (!(booleanFlag instanceof Value.BooleanValue flag))
      throw new TemplateRenderException(
          "`default` filter flag must be a boolean", ErrorCategory.TYPE, location);
    if (operand instanceof Value.DeferredUndefinedValue) {
      if (flag.value()) deferredValue(operand, "__bool__", location);
      deferredValue(operand, location);
    }
    return undefinedLike(operand) || flag.value() && !truthy(operand) ? fallback : operand;
  }

  private static Value filterLength(Value operand, SourceLocation location) {
    int length;
    if (operand instanceof Value.ArrayValue array) length = array.values().size();
    else if (operand instanceof Value.TupleValue tuple) length = tuple.values().size();
    else if (operand instanceof Value.StringValue string && !string.undefinedBacked())
      length = string.value().length();
    else if (operand instanceof Value.ObjectValue object) length = object.values().size();
    else if (operand instanceof Value.KeywordArgumentsValue object) length = object.values().size();
    else throw filterReceiver("length", operand, location);
    return new Value.IntegerValue(length);
  }

  private static Value filterString(
      Value operand,
      NamedArguments filter,
      SourceLocation location,
      java.util.function.UnaryOperator<String> operation) {
    if (!(operand instanceof Value.StringValue string) || string.undefinedBacked())
      throw filterReceiver(filter.name(), operand, location);
    return new Value.StringValue(operation.apply(string.value()));
  }

  private static Value filterJoin(Value operand, NamedArguments filter, SourceLocation location) {
    Value separator =
        filter.positional().isEmpty()
            ? filter.keywords().get("separator")
            : filter.positional().get(0);
    if (separator == null || separator instanceof Value.DeferredUndefinedValue)
      separator = new Value.StringValue("");
    if (!(separator instanceof Value.StringValue string))
      throw new TemplateRenderException("separator must be a string", ErrorCategory.TYPE, location);
    List<Value> values;
    if (operand instanceof Value.ArrayValue array) values = array.values();
    else if (operand instanceof Value.TupleValue tuple) values = tuple.values();
    else if (operand instanceof Value.StringValue value) {
      if (value.undefinedBacked()) throw filterType("undefined is not iterable", location);
      values =
          value
              .value()
              .codePoints()
              .mapToObj(c -> (Value) new Value.StringValue(new String(Character.toChars(c))))
              .toList();
    } else throw filterReceiver("join", operand, location);
    String separatorText = string.undefinedBacked() ? "," : string.value();
    return new Value.StringValue(
        values.stream()
            .map(v -> joinText(v, location))
            .collect(java.util.stream.Collectors.joining(separatorText)));
  }

  private static Value filterList(Value operand, SourceLocation location) {
    if (operand instanceof Value.ArrayValue array) return array;
    if (operand instanceof Value.TupleValue tuple) return tuple;
    throw filterReceiver("list", operand, location);
  }

  private static Value filterItems(Value operand, SourceLocation location) {
    if (operand instanceof Value.ObjectValue object) return itemsOf(object.values());
    if (operand instanceof Value.KeywordArgumentsValue object) return itemsOf(object.values());
    throw filterReceiver("items", operand, location);
  }

  private static List<Value> sequence(Value operand, String name, SourceLocation location) {
    if (operand instanceof Value.ArrayValue array) return array.values();
    if (operand instanceof Value.TupleValue tuple) return tuple.values();
    throw filterReceiver(name, operand, location);
  }

  private static Value filterFirstLast(
      Value operand, NamedArguments filter, SourceLocation location, boolean last) {
    var values = sequence(operand, filter.name(), location);
    if (values.isEmpty()) return Value.DeferredUndefinedValue.INSTANCE;
    return values.get(last ? values.size() - 1 : 0);
  }

  private static boolean lowerTest(Value value, SourceLocation location) {
    if (!(value instanceof Value.StringValue string)) return false;
    if (string.undefinedBacked())
      throw filterType("Cannot read properties of undefined (reading 'toLowerCase')", location);
    return string.value().equals(string.value().toLowerCase(Locale.ROOT));
  }

  private static Value filterReverse(
      Value operand, NamedArguments filter, SourceLocation location) {
    var values = new ArrayList<>(sequence(operand, filter.name(), location));
    Collections.reverse(values);
    return new Value.ArrayValue(values);
  }

  private static Value filterUnique(Value operand, NamedArguments filter, SourceLocation location) {
    var result = new ArrayList<Value>();
    for (var value : sequence(operand, filter.name(), location)) {
      deferredValue(value, location);
      if (result.stream().noneMatch(existing -> JsOperations.strictValueEquals(existing, value)))
        result.add(value);
    }
    return new Value.ArrayValue(result);
  }

  private static Value filterAbs(Value operand, NamedArguments filter, SourceLocation location) {
    if (operand instanceof Value.IntegerValue number)
      return new Value.IntegerValue(Math.abs(number.value()));
    if (operand instanceof Value.FloatValue number)
      return new Value.FloatValue(Math.abs(number.value()));
    throw filterReceiver(filter.name(), operand, location);
  }

  private static Value filterSort(Value operand, NamedArguments filter, SourceLocation location) {
    var values = new ArrayList<>(sequence(operand, filter.name(), location));
    boolean reverse = filterBoolean(filter, 0, "reverse", false, location);
    boolean caseSensitive = filterBoolean(filter, 1, "case_sensitive", false, location);
    Value attribute = filterArgument(filter, 2, "attribute");
    if (attribute != null
        && !(attribute instanceof Value.NullValue
            || attribute instanceof Value.StringValue
            || attribute instanceof Value.IntegerValue))
      throw filterType("attribute must be a string, integer, or null", location);
    values.sort(
        (left, right) -> {
          Value a = sortValue(left, attribute);
          Value b = sortValue(right, attribute);
          int comparison = compareValues(a, b, caseSensitive, location);
          return reverse ? -comparison : comparison;
        });
    return new Value.ArrayValue(values);
  }

  private static Value filterMap(Value operand, NamedArguments filter, SourceLocation location) {
    var attribute = filter.keywords().get("attribute");
    if (attribute == null)
      throw filterType(
          "`map` expressions without `attribute` set are not currently supported.", location);
    if (!(attribute instanceof Value.StringValue string) || string.undefinedBacked())
      throw filterType("attribute must be a string", location);
    Value fallback =
        absentFilterArgument(filter.keywords().get("default"), Value.UndefinedValue.INSTANCE);
    var values = new ArrayList<Value>();
    for (var item : sequence(operand, filter.name(), location)) {
      if (!(item instanceof Value.ObjectValue))
        throw filterType("items in map must be an object", location);
      Value mapped = attribute(item, string.value());
      values.add(mapped instanceof Value.UndefinedValue ? fallback : mapped);
    }
    return new Value.ArrayValue(values);
  }

  private static Value filterIndent(
      Value operand, NamedArguments filter, SourceLocation location, RenderBudget budget) {
    if (!(operand instanceof Value.StringValue string) || string.undefinedBacked())
      throw filterReceiver(filter.name(), operand, location);
    Value width = filterArgument(filter, 0, "width");
    if (width == null) width = new Value.IntegerValue(4);
    if (!(width instanceof Value.IntegerValue number))
      throw filterType("width must be a number", location);
    boolean first = filterTruthy(filterArgument(filter, 1, "first"), false);
    boolean blank = filterTruthy(filterArgument(filter, 2, "blank"), false);
    if (number.value() < 0)
      throw new TemplateRenderException(
          "Invalid count value: " + JsFormat.plainString(number.value()),
          ErrorCategory.VALUE,
          location);
    if (number.value() > budget.remainingOutputLength())
      throw new TemplateRenderException(
          "Maximum render output length exceeded", ErrorCategory.RESOURCE_LIMIT, location);
    var lines = string.value().split("\\n", -1);
    String prefix = " ".repeat((int) number.value());
    for (int index = 0; index < lines.length; index++)
      if ((first || index != 0) && (blank || !lines[index].isEmpty()))
        lines[index] = prefix + lines[index];
    return new Value.StringValue(String.join("\n", lines));
  }

  private static Value filterReplace(
      Value operand, NamedArguments filter, SourceLocation location) {
    if (!(operand instanceof Value.StringValue string) || string.undefinedBacked())
      throw filterReceiver(filter.name(), operand, location);
    var arguments = new ArrayList<>(filter.positional());
    // Filter calls always append their keyword bag, including an empty one. The bag is an ordinary
    // upstream argument slot, so it can be observed as the replacement value.
    arguments.add(new Value.KeywordArgumentsValue(filter.keywords()));
    return stringReplace(string.value(), arguments, location);
  }

  private static Value filterArgument(NamedArguments filter, int index, String key) {
    Value value =
        filter.positional().size() > index
            ? filter.positional().get(index)
            : filter.keywords().get(key);
    return value instanceof Value.DeferredUndefinedValue ? null : value;
  }

  private static boolean filterBoolean(
      NamedArguments filter, int index, String key, boolean fallback, SourceLocation location) {
    Value value = filterArgument(filter, index, key);
    if (value == null) return fallback;
    if (value instanceof Value.BooleanValue booleanValue) return booleanValue.value();
    throw filterType(key + " must be a boolean", location);
  }

  private static boolean filterTruthy(Value value, boolean fallback) {
    // The upstream indent implementation applies JavaScript's raw !value check, rather than
    // Jinja's container-aware truthiness. In particular, [] and {} enable indentation here.
    return value == null ? fallback : JsOperations.rawTruthy(value);
  }

  private static Value absentFilterArgument(Value value, Value fallback) {
    return value == null || value instanceof Value.DeferredUndefinedValue ? fallback : value;
  }

  private static Value filterObjectBuiltin(
      Value operand, NamedArguments filter, SourceLocation location) {
    Map<?, Value> values;
    Map<String, Value.CallableValue> builtins;
    if (operand instanceof Value.ObjectValue object) {
      values = object.values();
      builtins = object.builtins();
    } else if (operand instanceof Value.KeywordArgumentsValue object) {
      values = object.values();
      builtins = object.builtins();
    } else throw filterReceiver(filter.name(), operand, location);
    var arguments = new ArrayList<>(filter.positional());
    if (!filter.keywords().isEmpty())
      arguments.add(new Value.KeywordArgumentsValue(filter.keywords()));
    var builtin =
        filter.name().equals("items")
            ? objectItemsBuiltin(builtins, values)
            : objectBuiltin(builtins, values, filter.name());
    return ((Value.CallableValue) builtin)
        .callable()
        .invoke(arguments, !filter.keywords().isEmpty(), location, null);
  }

  private static Value sortValue(Value value, Value attribute) {
    if (attribute == null || attribute instanceof Value.NullValue) return value;
    String path =
        attribute instanceof Value.StringValue string
            ? string.value()
            : JsFormat.plainString(((Value.IntegerValue) attribute).value());
    return attribute(value, path);
  }

  private static Value attribute(Value value, String path) {
    for (var part : path.split("\\.")) {
      if (value instanceof Value.ObjectValue object)
        value =
            Value.materialize(object.values().getOrDefault(part, Value.UndefinedValue.INSTANCE));
      else if (value instanceof Value.ArrayValue array) {
        try {
          int index = Integer.parseInt(part);
          value =
              index >= 0 && index < array.values().size()
                  ? Value.materialize(array.values().get(index))
                  : Value.UndefinedValue.INSTANCE;
        } catch (NumberFormatException ignored) {
          return Value.UndefinedValue.INSTANCE;
        }
      } else return Value.UndefinedValue.INSTANCE;
    }
    return value;
  }

  private static Value filterBool(Value operand, NamedArguments filter, SourceLocation location) {
    if (operand instanceof Value.BooleanValue) return operand;
    throw filterReceiver(filter.name(), operand, location);
  }

  private static String titleCase(String value) {
    var result = new StringBuilder(value.length());
    boolean previousWord = false;
    for (int index = 0; index < value.length(); index++) {
      char character = value.charAt(index);
      boolean word =
          (character >= 'A' && character <= 'Z')
              || (character >= 'a' && character <= 'z')
              || (character >= '0' && character <= '9')
              || character == '_';
      result.append(word && !previousWord ? Character.toUpperCase(character) : character);
      previousWord = word;
    }
    return result.toString();
  }

  private static String capitalize(String value) {
    return value.isEmpty() ? value : Character.toUpperCase(value.charAt(0)) + value.substring(1);
  }

  private static Value filterToString(Value operand, SourceLocation location) {
    if (operand instanceof Value.StringValue string) return string;
    if (operand instanceof Value.ArrayValue
        || operand instanceof Value.TupleValue
        || operand instanceof Value.IntegerValue
        || operand instanceof Value.FloatValue
        || operand instanceof Value.BooleanValue)
      return new Value.StringValue(renderText(operand, location));
    throw filterReceiver("string", operand, location);
  }

  /**
   * Applies an attribute filter to values already validated as {@link Value.ObjectValue} by {@link
   * #filterSelectAttrCall(List, String, Expression.CallExpression, Environment, RenderBudget,
   * SourceLocation, boolean)}.
   */
  private static Value filterSelectAttr(
      List<Value> values, NamedArguments filter, SourceLocation location, boolean select) {
    if (filter.positional().isEmpty())
      throw new TemplateRenderException(
          "`" + filter.name() + "` filter requires at least one argument",
          ErrorCategory.ARITY,
          location);
    var attr = requireFilterString(filter, 0, location);
    String testName =
        filter.positional().size() > 1 ? requireFilterString(filter, 1, location).value() : null;
    Value comparison = filter.positional().size() > 2 ? filter.positional().get(2) : null;
    var result = new ArrayList<Value>();
    for (var item : values) {
      var attrValue = Value.materialize(((Value.ObjectValue) item).values().get(attr.value()));
      boolean matched =
          attrValue != null
              && (testName == null
                  ? truthy(attrValue)
                  : namedTest(testName, attrValue, comparison, location));
      if (matched == select) result.add(item);
    }
    return new Value.ArrayValue(result);
  }

  private static Value.StringValue requireFilterString(
      NamedArguments filter, int index, SourceLocation location) {
    var value = filter.positional().get(index);
    if (value instanceof Value.StringValue string && !string.undefinedBacked()) return string;
    throw new TemplateRenderException(
        "`" + filter.name() + "` arguments must be strings", ErrorCategory.TYPE, location);
  }

  private static boolean namedTest(
      String name, Value value, Value comparison, SourceLocation location) {
    return switch (name) {
      case "equalto", "eq" -> {
        if (comparison == null)
          throw filterType("`" + name + "` test requires a comparison value", location);
        yield JsOperations.strictEquals(value, comparison);
      }
      case "defined" -> !undefinedLike(value);
      case "undefined" -> undefinedLike(value);
      case "none" -> value instanceof Value.NullValue;
      case "true" -> value instanceof Value.BooleanValue booleanValue && booleanValue.value();
      case "false" -> value instanceof Value.BooleanValue booleanValue && !booleanValue.value();
      case "boolean" -> value instanceof Value.BooleanValue;
      case "callable" -> value instanceof Value.CallableValue;
      case "odd" -> integerTest(value, true, location);
      case "even" -> integerTest(value, false, location);
      case "integer" -> value instanceof Value.IntegerValue;
      case "lower" -> lowerTest(value, location);
      case "upper" -> upperTest(value, location);
      case "number" -> JsOperations.numeric(value);
      case "string" -> value instanceof Value.StringValue;
      case "mapping" ->
          value instanceof Value.ObjectValue || value instanceof Value.KeywordArgumentsValue;
      case "iterable" -> value instanceof Value.ArrayValue || value instanceof Value.StringValue;
      case "sequence" ->
          value instanceof Value.ArrayValue
              || value instanceof Value.TupleValue
              || value instanceof Value.ObjectValue
              || value instanceof Value.KeywordArgumentsValue
              || value instanceof Value.StringValue;
      default -> throw filterType("Unknown test: " + name, location);
    };
  }

  private static boolean integerTest(Value value, boolean odd, SourceLocation location) {
    if (!(value instanceof Value.IntegerValue number))
      throw filterType("cannot " + (odd ? "odd" : "even") + " on " + type(value), location);
    return (number.value() % 2 != 0) == odd;
  }

  private static boolean upperTest(Value value, SourceLocation location) {
    if (!(value instanceof Value.StringValue string)) return false;
    if (string.undefinedBacked())
      throw filterType("Cannot read properties of undefined (reading 'toUpperCase')", location);
    return string.value().equals(string.value().toUpperCase(Locale.ROOT));
  }

  private static String joinText(Value value, SourceLocation location) {
    return switch (value) {
      case Value.NullValue ignored -> "";
      case Value.UndefinedValue ignored -> "";
      case Value.DeferredUndefinedValue ignored ->
          throw filterType("Cannot read properties of undefined (reading 'type')", location);
      case Value.StringValue string -> string.undefinedBacked() ? "" : string.value();
      case Value.ArrayValue array ->
          array.values().stream()
              .map(item -> joinText(item, location))
              .collect(java.util.stream.Collectors.joining(","));
      case Value.TupleValue tuple ->
          tuple.values().stream()
              .map(item -> joinText(item, location))
              .collect(java.util.stream.Collectors.joining(","));
      default -> JsOperations.payloadText(value, location);
    };
  }

  private static Value filterNumber(
      Value operand, NamedArguments filter, SourceLocation location, boolean integer) {
    var fallback =
        filter.positional().isEmpty()
            ? filter.keywords().get("default")
            : filter.positional().get(0);
    fallback =
        absentFilterArgument(
            fallback, integer ? new Value.IntegerValue(0) : new Value.FloatValue(0));
    if (operand instanceof Value.IntegerValue value)
      return integer ? value : new Value.FloatValue(value.value());
    if (operand instanceof Value.FloatValue value)
      return integer ? new Value.IntegerValue(Math.floor(value.value())) : value;
    if (operand instanceof Value.BooleanValue value)
      return integer
          ? new Value.IntegerValue(value.value() ? 1 : 0)
          : new Value.FloatValue(value.value() ? 1 : 0);
    if (!(operand instanceof Value.StringValue string))
      throw filterReceiver(filter.name(), operand, location);
    var parsed = integer ? parseInt(string.value()) : parseFloat(string.value());
    return Double.isNaN(parsed)
        ? fallback
        : integer ? new Value.IntegerValue(parsed) : new Value.FloatValue(parsed);
  }

  private static double parseInt(String text) {
    var matcher = java.util.regex.Pattern.compile("^[\\s]*([+-]?\\d+)").matcher(text);
    if (!matcher.find()) return Double.NaN;
    try {
      return Double.parseDouble(matcher.group(1));
    } catch (NumberFormatException ignored) {
      return Double.NaN;
    }
  }

  private static double parseFloat(String text) {
    var trimmed = JsOperations.trimEcmaWhitespace(text);
    if (trimmed.startsWith("Infinity") || trimmed.startsWith("+Infinity"))
      return Double.POSITIVE_INFINITY;
    if (trimmed.startsWith("-Infinity")) return Double.NEGATIVE_INFINITY;
    if (trimmed.startsWith("NaN")) return Double.NaN;
    var matcher =
        java.util.regex.Pattern.compile(
                "^([+-]?(?:(?:\\d+\\.?\\d*)|(?:\\.\\d+))(?:[eE][+-]?\\d+)?)")
            .matcher(trimmed);
    if (!matcher.find()) return Double.NaN;
    try {
      return Double.parseDouble(matcher.group(1));
    } catch (NumberFormatException ignored) {
      return Double.NaN;
    }
  }

  private record EvaluatedArguments(List<Value> positional, Map<String, Value> keywords) {}

  /**
   * Evaluates a call or filter argument list left-to-right, matching upstream's evaluateArguments:
   * keyword arguments accumulate separately, array/tuple spreads expand one level into positional
   * values, and an ordinary positional value after a keyword is rejected — but a spread after a
   * keyword is not, since upstream's spread branch bypasses that check.
   */
  private static EvaluatedArguments evaluateArguments(
      List<Expression> args, Environment env, RenderBudget budget) {
    var positional = new ArrayList<Value>();
    // LinkedHashMap preserves source insertion order so error reporting stays deterministic; the
    // returned map wraps this local unmodifiable rather than copying it via an order-agnostic
    // factory. keywords never escapes this method otherwise, so wrapping it directly is safe.
    var keywords = new LinkedHashMap<String, Value>();
    boolean sawKeyword = false;
    for (var argument : args) {
      if (argument instanceof Expression.KeywordArgumentExpression keyword) {
        sawKeyword = true;
        keywords.put(keyword.key().value(), evaluateExpression(keyword.value(), env, budget));
      } else if (argument instanceof Expression.SpreadExpression spread) {
        var value = evaluateExpression(spread.argument(), env, budget);
        if (value instanceof Value.ArrayValue array) positional.addAll(array.values());
        else if (value instanceof Value.TupleValue tuple) positional.addAll(tuple.values());
        else
          throw new TemplateRenderException(
              "Cannot unpack non-iterable type: " + type(value),
              ErrorCategory.TYPE,
              spread.location());
      } else {
        if (sawKeyword)
          throw new TemplateRenderException(
              "Positional arguments must come before keyword arguments",
              ErrorCategory.SYNTAX,
              argument.location());
        positional.add(evaluateExpression(argument, env, budget));
      }
    }
    return new EvaluatedArguments(
        List.copyOf(positional),
        keywords.isEmpty() ? Map.of() : Collections.unmodifiableMap(keywords));
  }

  private static TemplateRenderException filterReceiver(
      String name, Value value, SourceLocation location) {
    deferredValue(value, location);
    return filterType("Cannot apply filter \"" + name + "\" to type: " + type(value), location);
  }

  private static void deferredValue(Value value, SourceLocation location) {
    deferredValue(value, "type", location);
  }

  private static void deferredValue(Value value, String property, SourceLocation location) {
    if (value instanceof Value.DeferredUndefinedValue)
      throw filterType(
          "Cannot read properties of undefined (reading '" + property + "')", location);
  }

  private static TemplateRenderException unknownBareFilter(
      String name, Value value, SourceLocation location) {
    return switch (value) {
      case Value.ArrayValue ignored -> filterType("Unknown ArrayValue filter: " + name, location);
      case Value.TupleValue ignored -> filterType("Unknown ArrayValue filter: " + name, location);
      case Value.StringValue ignored -> filterType("Unknown StringValue filter: " + name, location);
      case Value.IntegerValue ignored ->
          filterType("Unknown NumericValue filter: " + name, location);
      case Value.FloatValue ignored -> filterType("Unknown NumericValue filter: " + name, location);
      case Value.BooleanValue ignored ->
          filterType("Unknown BooleanValue filter: " + name, location);
      case Value.ObjectValue ignored -> filterType("Unknown ObjectValue filter: " + name, location);
      case Value.NullValue ignored -> filterReceiver(name, value, location);
      case Value.UndefinedValue ignored -> filterReceiver(name, value, location);
      case Value.DeferredUndefinedValue ignored -> filterReceiver(name, value, location);
      case Value.CallableValue ignored -> filterReceiver(name, value, location);
      case Value.KeywordArgumentsValue ignored ->
          filterType("Unknown ObjectValue filter: " + name, location);
    };
  }

  private static TemplateRenderException unknownCallFilter(
      String name, Value value, SourceLocation location) {
    return switch (value) {
      case Value.ArrayValue ignored -> filterType("Unknown ArrayValue filter: " + name, location);
      case Value.TupleValue ignored -> filterType("Unknown ArrayValue filter: " + name, location);
      case Value.StringValue ignored -> filterType("Unknown StringValue filter: " + name, location);
      case Value.ObjectValue ignored -> filterType("Unknown ObjectValue filter: " + name, location);
      case Value.KeywordArgumentsValue ignored ->
          filterType("Unknown ObjectValue filter: " + name, location);
      case Value.IntegerValue ignored -> filterReceiver(name, value, location);
      case Value.FloatValue ignored -> filterReceiver(name, value, location);
      case Value.BooleanValue ignored -> filterReceiver(name, value, location);
      case Value.NullValue ignored -> filterReceiver(name, value, location);
      case Value.UndefinedValue ignored -> filterReceiver(name, value, location);
      case Value.DeferredUndefinedValue ignored -> filterReceiver(name, value, location);
      case Value.CallableValue ignored -> filterReceiver(name, value, location);
    };
  }

  private static TemplateRenderException filterType(String message, SourceLocation location) {
    return new TemplateRenderException(message, ErrorCategory.TYPE, location);
  }

  private static boolean undefinedLike(Value value) {
    return value instanceof Value.UndefinedValue || value instanceof Value.DeferredUndefinedValue;
  }

  private static TemplateRenderException operatorUndefined(
      String operator, SourceLocation location) {
    return new TemplateRenderException(
        "Cannot perform operation " + operator + " on undefined values",
        ErrorCategory.UNDEFINED_OR_ACCESS,
        location);
  }

  private static TemplateRenderException operatorNull(SourceLocation location) {
    return new TemplateRenderException(
        "Cannot perform operation on null values", ErrorCategory.UNDEFINED_OR_ACCESS, location);
  }

  private static TemplateRenderException operatorUnsupportedTypes(
      String operator, Value left, Value right, SourceLocation location) {
    return new TemplateRenderException(
        "Unknown operator \"" + operator + "\" between " + type(left) + " and " + type(right),
        ErrorCategory.TYPE,
        location);
  }

  private static TemplateRenderException operatorUnsupportedUnary(
      String operator, Value operand, SourceLocation location) {
    return new TemplateRenderException(
        "Unknown unary operator \"" + operator + "\" for " + type(operand),
        ErrorCategory.TYPE,
        location);
  }

  private static List<Value> values(List<Expression> items, Environment e, RenderBudget b) {
    var r = new ArrayList<Value>();
    for (var x : items) r.add(evaluateExpression(x, e, b));
    return r;
  }

  private static Value object(Expression.ObjectLiteral x, Environment e, RenderBudget b) {
    var r = new LinkedHashMap<Object, Value>();
    for (var item : x.value()) {
      var k = evaluateExpression(item.key(), e, b);
      if (!(k instanceof Value.StringValue key))
        throw new TemplateRenderException(
            "Object keys must be strings: got " + type(k),
            ErrorCategory.TYPE,
            item.key().location());
      r.put(key.undefinedBacked() ? key : key.value(), evaluateExpression(item.value(), e, b));
    }
    return new Value.ObjectValue(r);
  }

  private static ExecResult evaluateIf(Statement.If n, Environment e, RenderBudget b) {
    return evaluateBlock(
        truthy(evaluateExpression(n.test(), e, b), n.test().location()) ? n.body() : n.alternate(),
        e,
        b);
  }

  static boolean truthy(Value v) {
    return truthy(v, null);
  }

  private static boolean truthy(Value v, SourceLocation location) {
    return switch (v) {
      case Value.NullValue ignored -> false;
      case Value.UndefinedValue ignored -> false;
      case Value.DeferredUndefinedValue ignored -> {
        throw filterType("Cannot read properties of undefined (reading '__bool__')", location);
      }
      case Value.BooleanValue x -> x.value();
      case Value.IntegerValue x -> x.value() != 0 && !Double.isNaN(x.value());
      case Value.FloatValue x -> x.value() != 0 && !Double.isNaN(x.value());
      case Value.StringValue x -> !x.undefinedBacked() && !x.value().isEmpty();
      case Value.ArrayValue x -> !x.values().isEmpty();
      case Value.TupleValue x -> !x.values().isEmpty();
      case Value.ObjectValue x -> !x.values().isEmpty();
      case Value.KeywordArgumentsValue x -> !x.values().isEmpty();
      case Value.CallableValue ignored -> true;
    };
  }

  private static Value member(Expression.MemberExpression n, Environment e, RenderBudget b) {
    var target = evaluateExpression(n.object(), e, b);
    deferredValue(target, "builtins", n.location());
    if (n.computed() && n.property() instanceof Expression.SliceExpression slice)
      return slice(target, slice, e, b, n.location());
    var p =
        !n.computed() && n.property() instanceof Expression.Identifier id
            ? new Value.StringValue(id.value())
            : evaluateExpression(n.property(), e, b);
    if (target instanceof Value.ObjectValue x) {
      if (!(p instanceof Value.StringValue s))
        throw access("Cannot access property with non-string: got " + type(p), n.location());
      var key = objectKey(s);
      Value memberValue = x.values().get(key);
      if (memberValue != null && !(memberValue instanceof Value.DeferredUndefinedValue))
        return Value.materialize(memberValue);
      if (!s.undefinedBacked() && "items".equals(s.value()))
        return objectItemsBuiltin(x.builtins(), x.values());
      if (!s.undefinedBacked()
          && (s.value().equals("get")
              || s.value().equals("keys")
              || s.value().equals("values")
              || s.value().equals("dictsort")))
        return objectBuiltin(x.builtins(), x.values(), s.value());
      return Value.UndefinedValue.INSTANCE;
    }
    if (target instanceof Value.KeywordArgumentsValue x) {
      if (!(p instanceof Value.StringValue s))
        throw access("Cannot access property with non-string: got " + type(p), n.location());
      if (s.undefinedBacked()) return Value.UndefinedValue.INSTANCE;
      Value memberValue = x.values().get(s.value());
      if (memberValue != null && !(memberValue instanceof Value.DeferredUndefinedValue))
        return Value.materialize(memberValue);
      if (s.value().equals("items")) return objectItemsBuiltin(x.builtins(), x.values());
      if (s.value().equals("get")
          || s.value().equals("keys")
          || s.value().equals("values")
          || s.value().equals("dictsort"))
        return objectBuiltin(x.builtins(), x.values(), s.value());
      return Value.UndefinedValue.INSTANCE;
    }
    if (target instanceof Value.ArrayValue x) {
      if (p instanceof Value.StringValue property
          && !property.undefinedBacked()
          && property.value().equals("length")) return new Value.IntegerValue(x.values().size());
      return memberIndex(x.values(), p, n.location());
    }
    if (target instanceof Value.TupleValue x) {
      if (p instanceof Value.StringValue property
          && !property.undefinedBacked()
          && property.value().equals("length")) return new Value.IntegerValue(x.values().size());
      return memberIndex(x.values(), p, n.location());
    }
    if (target instanceof Value.StringValue x) {
      if (x.undefinedBacked())
        throw filterType("Cannot read properties of undefined (reading 'at')", n.location());
      if (p instanceof Value.StringValue property) {
        if (!property.undefinedBacked() && property.value().equals("length"))
          return new Value.IntegerValue(x.value().length());
        if (List.of(
                "startswith",
                "endswith",
                "upper",
                "lower",
                "strip",
                "title",
                "capitalize",
                "rstrip",
                "lstrip",
                "split",
                "replace")
            .contains(property.value())) return stringBuiltin(x, property.value());
        return Value.UndefinedValue.INSTANCE;
      }
      if (!(p instanceof Value.IntegerValue))
        throw access(
            "Cannot access property with non-string/non-number: got " + type(p), n.location());
      var v = index(x.value().length(), p);
      return v < 0
          ? Value.StringValue.undefined()
          : new Value.StringValue(String.valueOf(x.value().charAt(v)));
    }
    if (!(p instanceof Value.StringValue))
      throw access("Cannot access property with non-string: got " + type(p), n.location());
    return Value.UndefinedValue.INSTANCE;
  }

  private static Value stringBoundaryBuiltin(
      String receiver, String name, BiPredicate<String, String> matches) {
    return new Value.CallableValue(
        (arguments, hasKeywords, location, environment) -> {
          if (arguments.isEmpty())
            throw new TemplateRenderException(
                name + "() requires at least one argument", ErrorCategory.ARITY, location);
          var pattern = arguments.getFirst();
          if (pattern instanceof Value.StringValue string)
            return new Value.BooleanValue(matches.test(receiver, string.value()));
          if (pattern instanceof Value.ArrayValue array)
            return stringBoundaryTuple(receiver, name, matches, array.values(), location);
          if (pattern instanceof Value.TupleValue tuple)
            return stringBoundaryTuple(receiver, name, matches, tuple.values(), location);
          throw new TemplateRenderException(
              name + "() argument must be a string or tuple of strings",
              ErrorCategory.TYPE,
              location);
        },
        Value.CallableValue.JAVA_FUNCTION_MARKER);
  }

  private static Value stringBuiltin(Value.StringValue receiver, String name) {
    return receiver.builtins().computeIfAbsent(name, ignored -> stringBuiltinValue(receiver, name));
  }

  private static Value.CallableValue stringBuiltinValue(Value.StringValue receiver, String name) {
    String text = receiver.value();
    if (name.equals("startswith") || name.equals("endswith"))
      return (Value.CallableValue)
          stringBoundaryBuiltin(
              text, name, name.equals("startswith") ? String::startsWith : String::endsWith);
    return new Value.CallableValue(
        (arguments, hasKeywords, location, environment) -> {
          return switch (name) {
            case "upper" -> new Value.StringValue(text.toUpperCase(Locale.ROOT));
            case "lower" -> new Value.StringValue(text.toLowerCase(Locale.ROOT));
            case "strip" -> new Value.StringValue(JsOperations.trimEcmaWhitespace(text));
            case "rstrip" -> new Value.StringValue(JsOperations.trimEndEcmaWhitespace(text));
            case "lstrip" -> new Value.StringValue(JsOperations.trimStartEcmaWhitespace(text));
            case "title" -> new Value.StringValue(titleCase(text));
            case "capitalize" -> new Value.StringValue(capitalize(text));
            case "split" -> stringSplit(text, arguments, location);
            case "replace" -> stringReplace(text, arguments, location);
            default -> throw new AssertionError(name);
          };
        },
        Value.CallableValue.JAVA_FUNCTION_MARKER);
  }

  private static List<Value> positional(List<Value> arguments) {
    if (arguments.isEmpty() || !(arguments.getLast() instanceof Value.KeywordArgumentsValue))
      return arguments;
    return arguments.subList(0, arguments.size() - 1);
  }

  private static Value stringSplit(
      String receiver, List<Value> arguments, SourceLocation location) {
    Value separator =
        arguments.isEmpty()
            ? Value.NullValue.INSTANCE
            : absentFilterArgument(arguments.getFirst(), Value.NullValue.INSTANCE);
    if (!(separator instanceof Value.StringValue) && !(separator instanceof Value.NullValue))
      throw filterType("sep argument must be a string or null", location);
    int max = -1;
    Value maxSplit = arguments.size() > 1 ? absentFilterArgument(arguments.get(1), null) : null;
    if (maxSplit != null) {
      if (!(maxSplit instanceof Value.IntegerValue number))
        throw filterType("maxsplit argument must be a number", location);
      max = (int) number.value();
    }
    String[] parts;
    if (separator instanceof Value.NullValue) {
      String text = JsOperations.trimStartEcmaWhitespace(receiver);
      var result = new ArrayList<String>();
      int index = 0;
      while (index < text.length()) {
        while (index < text.length() && JsOperations.isEcmaWhitespace(text.charAt(index))) index++;
        if (index == text.length()) break;
        int start = index;
        while (index < text.length() && !JsOperations.isEcmaWhitespace(text.charAt(index))) index++;
        if (max != -1 && result.size() >= max) {
          // Upstream appends the current match plus the otherwise-unsplit suffix.
          result.add(text.substring(start));
          break;
        }
        result.add(text.substring(start, index));
      }
      parts = result.toArray(String[]::new);
    } else {
      String text = ((Value.StringValue) separator).value();
      if (text.isEmpty()) throw filterType("empty separator", location);
      var split =
          new ArrayList<>(
              java.util.Arrays.asList(
                  receiver.split(java.util.regex.Pattern.quote(text), max == -1 ? -1 : max + 1)));
      if (max < -1) {
        // Upstream splits without a limit, then rejoins the suffix selected by Array.splice(). A
        // negative splice index is relative to the end, rather than another spelling of unlimited.
        int joinStart = Math.max(0, split.size() + max);
        String suffix = String.join(text, split.subList(joinStart, split.size()));
        split.subList(joinStart, split.size()).clear();
        split.add(suffix);
      }
      parts = split.toArray(String[]::new);
    }
    return new Value.ArrayValue(
        java.util.Arrays.stream(parts).map(Value.StringValue::new).map(x -> (Value) x).toList());
  }

  private static Value stringReplace(
      String receiver, List<Value> arguments, SourceLocation location) {
    if (arguments.size() < 2)
      throw filterType("replace() requires at least two arguments", location);
    if (!(arguments.get(0) instanceof Value.StringValue oldValue)
        || !(arguments.get(1) instanceof Value.StringValue newValue))
      throw filterType("replace() arguments must be strings", location);
    if (oldValue.undefinedBacked())
      throw filterType("Cannot read properties of undefined (reading 'length')", location);
    Value count;
    if (arguments.size() > 2) {
      Value thirdArgument = arguments.get(2);
      deferredValue(thirdArgument, "type", location);
      count =
          thirdArgument instanceof Value.KeywordArgumentsValue
              ? absentFilterArgument(keyword(arguments, "count"), Value.NullValue.INSTANCE)
              : thirdArgument;
    } else count = Value.NullValue.INSTANCE;
    if (count != null
        && !(count instanceof Value.IntegerValue)
        && !(count instanceof Value.NullValue))
      throw filterType("replace() count argument must be a number or null", location);
    return new Value.StringValue(
        replace(
            receiver,
            oldValue.value(),
            newValue.value(),
            count instanceof Value.IntegerValue n ? (int) n.value() : -1));
  }

  private static String replace(String value, String oldValue, String newValue, int count) {
    if (count == 0) return value;
    var result = new StringBuilder();
    int position = 0;
    int replacements = 0;
    while (count < 0 || replacements < count) {
      int index = value.indexOf(oldValue, position);
      if (index < 0) break;
      result.append(value, position, index).append(newValue);
      position = index + oldValue.length();
      replacements++;
      if (oldValue.isEmpty()) {
        if (position >= value.length()) break;
        int next = value.offsetByCodePoints(position, 1);
        result.append(value, position, next);
        position = next;
      }
    }
    return result.append(value.substring(position)).toString();
  }

  private static Value keyword(List<Value> arguments, String key) {
    return !arguments.isEmpty()
            && arguments.getLast() instanceof Value.KeywordArgumentsValue keywords
        ? keywords.values().get(key)
        : null;
  }

  private static Value objectBuiltin(
      Map<String, Value.CallableValue> builtins, Map<?, Value> values, String name) {
    return builtins.computeIfAbsent(name, ignored -> objectBuiltinValue(values, name));
  }

  private static Value.CallableValue objectBuiltinValue(Map<?, Value> values, String name) {
    return new Value.CallableValue(
        (arguments, hasKeywords, location, environment) -> {
          var positional = positional(arguments);
          return switch (name) {
            case "get" -> {
              if (arguments.isEmpty() || !(arguments.getFirst() instanceof Value.StringValue key))
                throw filterType(
                    "Object key must be a string: got "
                        + (arguments.isEmpty() ? "UndefinedValue" : type(arguments.getFirst())),
                    location);
              yield values.getOrDefault(
                  key.value(),
                  arguments.size() > 1
                      ? Value.materialize(arguments.get(1))
                      : Value.NullValue.INSTANCE);
            }
            case "keys" ->
                new Value.ArrayValue(
                    values.keySet().stream().map(key -> (Value) objectKeyValue(key)).toList());
            case "values" -> new Value.ArrayValue(new ArrayList<>(values.values()));
            case "dictsort" -> dictSort(values, positional, arguments, location);
            default -> throw new AssertionError(name);
          };
        },
        Value.CallableValue.JAVA_FUNCTION_MARKER);
  }

  private static Value dictSort(
      Map<?, Value> values,
      List<Value> positional,
      List<Value> arguments,
      SourceLocation location) {
    Value caseValue =
        absentFilterArgument(
            positional.isEmpty() ? keyword(arguments, "case_sensitive") : positional.getFirst(),
            null);
    Value byValue =
        absentFilterArgument(
            positional.size() < 2 ? keyword(arguments, "by") : positional.get(1), null);
    Value reverseValue =
        absentFilterArgument(
            positional.size() < 3 ? keyword(arguments, "reverse") : positional.get(2), null);
    boolean caseSensitive =
        caseValue == null ? false : booleanValue(caseValue, "case_sensitive", location);
    String by = byValue == null ? "key" : stringValue(byValue, "by", location);
    if (!by.equals("key") && !by.equals("value"))
      throw filterType("by must be either 'key' or 'value'", location);
    boolean reverse = reverseValue != null && booleanValue(reverseValue, "reverse", location);
    var entries = new ArrayList<Value>();
    for (var entry : values.entrySet()) {
      entries.add(new Value.ArrayValue(List.of(objectKeyValue(entry.getKey()), entry.getValue())));
    }
    int index = by.equals("key") ? 0 : 1;
    entries.sort(
        (a, b) -> {
          int comparison =
              compareValues(
                  ((Value.ArrayValue) a).values().get(index),
                  ((Value.ArrayValue) b).values().get(index),
                  caseSensitive,
                  location);
          return reverse ? -comparison : comparison;
        });
    return new Value.ArrayValue(entries);
  }

  private static boolean booleanValue(Value value, String name, SourceLocation location) {
    if (value instanceof Value.BooleanValue booleanValue) return booleanValue.value();
    throw filterType(name + " must be a boolean", location);
  }

  private static String stringValue(Value value, String name, SourceLocation location) {
    if (value instanceof Value.StringValue string) return string.value();
    throw filterType(name + " must be a string", location);
  }

  private static int compareValues(
      Value left, Value right, boolean caseSensitive, SourceLocation location) {
    if (numericLike(left) && numericLike(right))
      return Double.compare(JsOperations.toNumber(left), JsOperations.toNumber(right));
    if (left instanceof Value.StringValue a && right instanceof Value.StringValue b) {
      if (!caseSensitive && (a.undefinedBacked() || b.undefinedBacked()))
        throw filterType("Cannot read properties of undefined (reading 'toLowerCase')", location);
      if (caseSensitive && (a.undefinedBacked() || b.undefinedBacked())) return 0;
      String aText = caseSensitive ? a.value() : a.value().toLowerCase(Locale.ROOT);
      String bText = caseSensitive ? b.value() : b.value().toLowerCase(Locale.ROOT);
      return aText.compareTo(bText);
    }
    if (left instanceof Value.BooleanValue a && right instanceof Value.BooleanValue b)
      return Boolean.compare(a.value(), b.value());
    if (left instanceof Value.NullValue && right instanceof Value.NullValue) return 0;
    if (left instanceof Value.UndefinedValue && right instanceof Value.UndefinedValue) return 0;
    throw filterType("Cannot compare " + type(left) + " with " + type(right), location);
  }

  private static boolean numericLike(Value value) {
    return JsOperations.numeric(value) || value instanceof Value.BooleanValue;
  }

  private static Value stringBoundaryTuple(
      String receiver,
      String name,
      BiPredicate<String, String> matches,
      List<Value> patterns,
      SourceLocation location) {
    for (var pattern : patterns) {
      if (!(pattern instanceof Value.StringValue string) || string.undefinedBacked())
        throw new TemplateRenderException(
            name + "() tuple elements must be strings", ErrorCategory.TYPE, location);
      if (matches.test(receiver, string.value())) return new Value.BooleanValue(true);
    }
    return new Value.BooleanValue(false);
  }

  private static Value objectItemsBuiltin(
      Map<String, Value.CallableValue> builtins, Map<?, Value> values) {
    return builtins.computeIfAbsent(
        "items",
        ignored ->
            new Value.CallableValue(
                (arguments, hasKeywords, location, environment) -> itemsOf(values),
                Value.CallableValue.JAVA_FUNCTION_MARKER));
  }

  private static Value.ArrayValue itemsOf(Map<?, Value> values) {
    var pairs = new ArrayList<Value>();
    for (var entry : values.entrySet()) {
      var key = objectKeyValue(entry.getKey());
      pairs.add(new Value.ArrayValue(List.of(key, entry.getValue())));
    }
    return new Value.ArrayValue(pairs);
  }

  private static Value.StringValue objectKeyValue(Object key) {
    return key instanceof Value.StringValue string ? string : new Value.StringValue((String) key);
  }

  private static Value slice(
      Value target,
      Expression.SliceExpression expression,
      Environment environment,
      RenderBudget budget,
      SourceLocation memberLocation) {
    if (!(arrayLike(target) || target instanceof Value.StringValue))
      throw access("Slice object must be an array or string", memberLocation);
    var start = sliceComponent(expression.start(), environment, budget);
    var stop = sliceComponent(expression.stop(), environment, budget);
    var step = sliceComponent(expression.step(), environment, budget);
    Double startValue = sliceBound(start, expression.start(), "start");
    Double stopValue = sliceBound(stop, expression.stop(), "stop");
    Double stepValue = sliceBound(step, expression.step(), "step");
    if (target instanceof Value.StringValue string && string.undefinedBacked())
      throw filterType("undefined is not iterable", memberLocation);
    if (arrayLike(target))
      return new Value.ArrayValue(
          JsSlice.slice(arrayValues(target), startValue, stopValue, stepValue));
    var string = ((Value.StringValue) target).value();
    var codePoints = string.codePoints().boxed().toList();
    var result = new StringBuilder();
    for (var codePoint : JsSlice.slice(codePoints, startValue, stopValue, stepValue))
      result.appendCodePoint(codePoint);
    return new Value.StringValue(result.toString());
  }

  private static Value sliceComponent(
      Expression expression, Environment environment, RenderBudget budget) {
    return expression == null
        ? Value.UndefinedValue.INSTANCE
        : evaluateExpression(expression, environment, budget);
  }

  private static Double sliceBound(Value value, Expression expression, String name) {
    if (value instanceof Value.UndefinedValue) return null;
    if (value instanceof Value.IntegerValue integer) return integer.value();
    throw access("Slice " + name + " must be numeric or undefined", expression.location());
  }

  private static TemplateRenderException access(String message, SourceLocation location) {
    return new TemplateRenderException(message, ErrorCategory.TYPE, location);
  }

  private static Object objectKey(Value.StringValue value) {
    return value.undefinedBacked() ? value : value.value();
  }

  private static Value memberIndex(List<Value> values, Value property, SourceLocation location) {
    if (property instanceof Value.StringValue) return Value.UndefinedValue.INSTANCE;
    if (!(property instanceof Value.IntegerValue))
      throw access(
          "Cannot access property with non-string/non-number: got " + type(property), location);
    return indexed(values, property);
  }

  private static Value indexed(List<Value> values, Value p) {
    int i = index(values.size(), p);
    return i < 0 ? Value.UndefinedValue.INSTANCE : Value.materialize(values.get(i));
  }

  private static int index(int size, Value p) {
    if (!(p instanceof Value.IntegerValue x)) return -1;
    int i = (int) x.value();
    if (i < 0) i += size;
    return i < 0 || i >= size ? -1 : i;
  }

  private static Value call(Expression.CallExpression n, Environment e, RenderBudget b) {
    var evaluated = evaluateArguments(n.args(), e, b);
    var arguments = new ArrayList<>(evaluated.positional());
    if (!evaluated.keywords().isEmpty())
      arguments.add(new Value.KeywordArgumentsValue(evaluated.keywords()));
    var callee = evaluateExpression(n.callee(), e, b);
    if (!(callee instanceof Value.CallableValue f))
      throw new TemplateRenderException(
          "Cannot call something that is not a function: got " + type(callee),
          ErrorCategory.TYPE,
          n.location());
    return f.callable().invoke(arguments, !evaluated.keywords().isEmpty(), n.location(), e);
  }

  private static String type(Value v) {
    if (v instanceof Value.DeferredUndefinedValue) return "UndefinedValue";
    return v instanceof Value.CallableValue ? "FunctionValue" : v.getClass().getSimpleName();
  }

  private static Value callableBodyResult(ExecResult result, SourceLocation location) {
    if (result instanceof ExecResult.Normal normal) return new Value.StringValue(normal.output());
    throw new LoopControl(result, location);
  }

  private static ExecResult evaluateMacro(Statement.Macro n, Environment e, RenderBudget b) {
    e.setVariable(
        n.name().value(),
        new Value.CallableValue(
            (arguments, hasKeywords, l, scope) -> {
              // scope is the call-site environment (Step 2), not the macro's own defining
              // environment `e`
              // captured in this closure — this is what makes upstream's late-binding default
              // arguments
              // and call-block scope visibility work.
              var macroScope = new Environment(scope);
              // enterMacro must guard the binding loop below, not just the body: a default
              // argument expression can itself call this macro (e.g. `{% macro m(a=m()) %}`),
              // and evaluateExpression(kwarg.value(), ...) recurses back into this same lambda
              // before evaluateBlock(n.body(), ...) is ever reached. Entering here — before
              // binding — closes that gap.
              //
              // enterMacro is called BEFORE the try, not inside it: enterMacro is
              // check-before-increment, so on the limit-exceeded path it never touches
              // macroDepth — there is nothing for the finally below to undo. Calling it inside
              // the try would make that finally run exitMacro() on a call that never
              // incremented, decrementing macroDepth without a matching increment.
              b.enterMacro(l);
              try {
                var positional = new ArrayList<>(arguments);
                Value.KeywordArgumentsValue kwargs = null;
                if (!positional.isEmpty()
                    && positional.get(positional.size() - 1)
                        instanceof Value.KeywordArgumentsValue k) {
                  kwargs = k;
                  positional.remove(positional.size() - 1);
                }
                for (int i = 0; i < n.args().size(); i++) {
                  var nodeArg = n.args().get(i);
                  Value passed = i < positional.size() ? positional.get(i) : null;
                  if (nodeArg instanceof Expression.Identifier id) {
                    if (passed == null || passed instanceof Value.DeferredUndefinedValue)
                      throw new TemplateRenderException(
                          "Missing positional argument: " + id.value(), ErrorCategory.ARITY, l);
                    macroScope.setVariable(id.value(), passed);
                  } else if (nodeArg instanceof Expression.KeywordArgumentExpression kwarg) {
                    Value fromKwargs =
                        kwargs == null ? null : kwargs.values().get(kwarg.key().value());
                    Value value =
                        passed != null && !(passed instanceof Value.DeferredUndefinedValue)
                            ? passed
                            : fromKwargs != null
                                    && !(fromKwargs instanceof Value.DeferredUndefinedValue)
                                ? fromKwargs
                                : evaluateExpression(kwarg.value(), macroScope, b);
                    macroScope.setVariable(kwarg.key().value(), value);
                  } else {
                    throw new TemplateRenderException(
                        "Unknown argument type: " + nodeArg.getClass().getSimpleName(),
                        ErrorCategory.SYNTAX,
                        nodeArg.location());
                  }
                }
                return callableBodyResult(evaluateBlock(n.body(), macroScope, b), l);
              } finally {
                b.exitMacro();
              }
            },
            Value.CallableValue.JAVA_FUNCTION_MARKER));
    return new ExecResult.Normal("");
  }

  private static ExecResult evaluateCallStatement(
      Statement.CallStatement n, Environment e, RenderBudget b) {
    var caller =
        new Value.CallableValue(
            (callerArgs, hasKeywords, l, callerScope) -> {
              // callerScope is wherever caller() gets called from inside the macro body — this is
              // what lets the call-block body see macro-local state set before caller() runs.
              var callBlockEnv = new Environment(callerScope);
              if (n.callerArgs() != null) {
                for (int i = 0; i < n.callerArgs().size(); i++) {
                  var param = n.callerArgs().get(i);
                  if (!(param instanceof Expression.Identifier id))
                    throw new TemplateRenderException(
                        "Caller parameter must be an identifier, got "
                            + param.getClass().getSimpleName(),
                        ErrorCategory.SYNTAX,
                        param.location());
                  Value value =
                      i < callerArgs.size() ? callerArgs.get(i) : Value.UndefinedValue.INSTANCE;
                  callBlockEnv.setVariable(id.value(), value);
                }
              }
              // enterMacro is called BEFORE the try, not inside it — see the matching comment in
              // evaluateMacro above: it is check-before-increment, so a limit-exceeded throw
              // never touches macroDepth, and calling it inside the try would make the finally
              // below decrement a call that never incremented.
              b.enterMacro(l);
              try {
                return callableBodyResult(evaluateBlock(n.body(), callBlockEnv, b), l);
              } finally {
                b.exitMacro();
              }
            },
            Value.CallableValue.JAVA_FUNCTION_MARKER);

    var evaluated = evaluateArguments(n.call().args(), e, b);
    var arguments = new ArrayList<>(evaluated.positional());
    // Unlike call(), the keyword-arguments bag is pushed unconditionally, even when empty,
    // matching upstream's own evaluateCallStatement/evaluateCallExpression asymmetry (needed for
    // range()/namespace()/macro callees to match the oracle). This means every non-host callee
    // invoked via {% call %} sees a trailing empty-map argument; HostFunctions.invoke strips it
    // before it reaches user-registered host functions, since they have no such upstream contract.
    arguments.add(new Value.KeywordArgumentsValue(evaluated.keywords()));
    var callee = evaluateExpression(n.call().callee(), e, b);
    if (!(callee instanceof Value.CallableValue f))
      throw new TemplateRenderException(
          "Cannot call something that is not a function: got " + type(callee),
          ErrorCategory.TYPE,
          n.call().location());
    var newEnv = new Environment(e);
    newEnv.setVariable("caller", caller);
    var result =
        f.callable()
            .invoke(arguments, !evaluated.keywords().isEmpty(), n.call().location(), newEnv);
    var text =
        result instanceof Value.NullValue || result instanceof Value.UndefinedValue
            ? ""
            : renderText(result, n.location());
    // The call-block body's own text was already charged once inside evaluateBlock above (each
    // inner Expression statement charges as it renders); this charges the callee's returned text
    // again. maxOutputLength is therefore a bound on cumulative rendered characters, not on final
    // output size, for any construct that re-renders already-charged text — evaluateSet's
    // {% set x %}...{% endset %} block-capture already has this same property (charged once
    // inside the block, again wherever `x` is later interpolated). Not a regression introduced
    // here; consistent with pre-existing behavior.
    b.chargeOutput(text.length(), n.location());
    return new ExecResult.Normal(text);
  }

  private static ExecResult evaluateFilterStatement(
      Statement.FilterStatement n, Environment e, RenderBudget b) {
    var rendered = evaluateBlock(n.body(), e, b);
    // Propagating a non-Normal result outward (rather than erroring, unlike the macro/call-block
    // paths above) is deliberate: FilterStatement never crosses the Value.CallableValue.Callable
    // boundary, so this composes correctly exactly as evaluateSet's block-capture already does —
    // confirmed against the oracle that a bare break here bleeds into the caller's enclosing loop.
    if (!(rendered instanceof ExecResult.Normal normal)) return rendered;
    var filtered =
        applyFilter(
            new Value.StringValue(normal.output()), n.filter(), e, b, n.filter().location());
    var text =
        filtered instanceof Value.NullValue || filtered instanceof Value.UndefinedValue
            ? ""
            : renderText(filtered, n.location());
    // Re-charges the body's already-charged text after filtering; see the matching comment in
    // evaluateCallStatement above.
    b.chargeOutput(text.length(), n.location());
    return new ExecResult.Normal(text);
  }

  private static ExecResult evaluateSet(Statement.SetStatement n, Environment e, RenderBudget b) {
    Value rhs;
    if (n.value() != null) rhs = evaluateExpression(n.value(), e, b);
    else {
      var r = evaluateBlock(n.body(), e, b);
      if (!(r instanceof ExecResult.Normal normal)) return r;
      rhs = new Value.StringValue(normal.output());
    }
    if (n.assignee() instanceof Expression.Identifier x) e.setVariable(x.value(), rhs);
    else if (n.assignee() instanceof Expression.TupleLiteral x) {
      if (!(rhs instanceof Value.ArrayValue || rhs instanceof Value.TupleValue))
        throw new TemplateRenderException(
            "Cannot unpack non-iterable type in set: " + type(rhs),
            ErrorCategory.TYPE,
            n.location());
      List<Value> vals =
          rhs instanceof Value.ArrayValue a ? a.values() : ((Value.TupleValue) rhs).values();
      if (vals.size() != x.value().size())
        throw new TemplateRenderException(
            "Too " + (vals.size() < x.value().size() ? "few" : "many") + " items to unpack in set",
            ErrorCategory.VALUE,
            n.location());
      for (int i = 0; i < vals.size(); i++) {
        if (!(x.value().get(i) instanceof Expression.Identifier id))
          throw new TemplateRenderException(
              "Cannot unpack to non-identifier in set: "
                  + x.value().get(i).getClass().getSimpleName(),
              ErrorCategory.TYPE,
              n.location());
        e.setVariable(id.value(), vals.get(i));
      }
    } else if (n.assignee() instanceof Expression.MemberExpression x) {
      var target = evaluateExpression(x.object(), e, b);
      if (!(target instanceof Value.ObjectValue || target instanceof Value.KeywordArgumentsValue))
        throw new TemplateRenderException(
            "Cannot assign to member of non-object", ErrorCategory.TYPE, n.location());
      if (!(x.property() instanceof Expression.Identifier key))
        throw new TemplateRenderException(
            "Cannot assign to member with non-identifier property",
            ErrorCategory.TYPE,
            n.location());
      if (target instanceof Value.ObjectValue obj) obj.values().put(key.value(), rhs);
      else ((Value.KeywordArgumentsValue) target).values().put(key.value(), rhs);
    } else
      throw new TemplateRenderException(
          "Invalid LHS inside assignment expression: " + astJson(n.assignee()),
          ErrorCategory.SYNTAX,
          n.location());
    return new ExecResult.Normal("");
  }

  private static String astJson(Expression expression) {
    if (expression instanceof Expression.Identifier value)
      return "{\"type\":\"Identifier\",\"value\":" + JsFormat.quote(value.value()) + "}";
    if (expression instanceof Expression.IntegerLiteral value)
      return "{\"type\":\"IntegerLiteral\",\"value\":" + JsFormat.jsonString(value.value()) + "}";
    if (expression instanceof Expression.FloatLiteral value)
      return "{\"type\":\"FloatLiteral\",\"value\":" + JsFormat.jsonString(value.value()) + "}";
    if (expression instanceof Expression.StringLiteral value)
      return "{\"type\":\"StringLiteral\",\"value\":" + JsFormat.quote(value.value()) + "}";
    if (expression instanceof Expression.ArrayLiteral value)
      return "{\"type\":\"ArrayLiteral\",\"value\":" + astJsonList(value.value()) + "}";
    if (expression instanceof Expression.TupleLiteral value)
      return "{\"type\":\"TupleLiteral\",\"value\":" + astJsonList(value.value()) + "}";
    return "{\"type\":" + JsFormat.quote(expression.getClass().getSimpleName()) + "}";
  }

  private static String astJsonList(List<Expression> values) {
    return values.stream()
        .map(Interpreter::astJson)
        .collect(java.util.stream.Collectors.joining(",", "[", "]"));
  }

  private static ExecResult evaluateFor(Statement.For n, Environment e, RenderBudget b) {
    Expression test = n.iterable() instanceof Expression.SelectExpression s ? s.test() : null;
    Expression iterableExpression =
        n.iterable() instanceof Expression.SelectExpression s ? s.lhs() : n.iterable();
    var scope = new Environment(e);
    var iterable = evaluateExpression(iterableExpression, scope, b);
    List<Value> items;
    if (iterable instanceof Value.ArrayValue a) items = a.values();
    else if (iterable instanceof Value.TupleValue a) items = a.values();
    else if (iterable instanceof Value.ObjectValue o) {
      items = new ArrayList<>();
      for (var k : o.values().keySet())
        items.add(
            k instanceof Value.StringValue string ? string : new Value.StringValue((String) k));
    } else if (iterable instanceof Value.KeywordArgumentsValue o) {
      items = new ArrayList<>();
      for (var k : o.values().keySet()) items.add(new Value.StringValue(k));
    } else
      throw new TemplateRenderException(
          "Expected iterable or object type in for loop: got " + type(iterable),
          ErrorCategory.TYPE,
          n.location());
    var filtered = new ArrayList<Value>();
    for (var item : items) {
      b.chargeLoopIteration(n.location());
      var filterScope = new Environment(e);
      bind(n.loopVariable(), item, filterScope, n.location());
      if (test == null || truthy(evaluateExpression(test, filterScope, b), test.location()))
        filtered.add(item);
    }
    items = filtered;
    var out = new StringBuilder();
    boolean none = true;
    for (int i = 0; i < items.size(); i++) {
      var loop = new LinkedHashMap<String, Value>();
      loop.put("index", new Value.IntegerValue(i + 1));
      loop.put("index0", new Value.IntegerValue(i));
      loop.put("revindex", new Value.IntegerValue(items.size() - i));
      loop.put("revindex0", new Value.IntegerValue(items.size() - i - 1));
      loop.put("first", new Value.BooleanValue(i == 0));
      loop.put("last", new Value.BooleanValue(i == items.size() - 1));
      loop.put("length", new Value.IntegerValue(items.size()));
      loop.put("previtem", i > 0 ? items.get(i - 1) : Value.UndefinedValue.INSTANCE);
      loop.put("nextitem", i + 1 < items.size() ? items.get(i + 1) : Value.UndefinedValue.INSTANCE);
      scope.setVariable("loop", new Value.ObjectValue(loop));
      bind(n.loopVariable(), items.get(i), scope, n.location());
      ExecResult r;
      try {
        r = evaluateBlock(n.body(), scope, b);
      } catch (LoopControl control) {
        r = control.result();
      }
      if (r instanceof ExecResult.Break) break;
      if (r instanceof ExecResult.Continue) continue;
      none = false;
      out.append(((ExecResult.Normal) r).output());
    }
    return none ? evaluateBlock(n.defaultBlock(), scope, b) : new ExecResult.Normal(out.toString());
  }

  private static void bind(Expression target, Value item, Environment e, SourceLocation l) {
    if (target instanceof Expression.Identifier x) {
      e.setVariable(x.value(), item);
      return;
    }
    if (!(target instanceof Expression.TupleLiteral tuple))
      throw new TemplateRenderException(
          "Invalid loop variable(s): " + target.getClass().getSimpleName(), ErrorCategory.TYPE, l);
    if (!(item instanceof Value.ArrayValue a))
      throw new TemplateRenderException(
          "Cannot unpack non-iterable type: " + type(item), ErrorCategory.TYPE, l);
    if (a.values().size() != tuple.value().size())
      throw new TemplateRenderException(
          "Too " + (a.values().size() < tuple.value().size() ? "few" : "many") + " items to unpack",
          ErrorCategory.VALUE,
          l);
    for (int i = 0; i < a.values().size(); i++) {
      if (!(tuple.value().get(i) instanceof Expression.Identifier id))
        throw new TemplateRenderException(
            "Cannot unpack non-identifier type: " + tuple.value().get(i).getClass().getSimpleName(),
            ErrorCategory.TYPE,
            l);
      e.setVariable(id.value(), a.values().get(i));
    }
  }

  static String renderText(Value v, SourceLocation l) {
    return switch (v) {
      case Value.BooleanValue x -> Boolean.toString(x.value());
      case Value.IntegerValue x -> JsFormat.plainString(x.value());
      case Value.FloatValue x -> renderFloat(x.value());
      case Value.StringValue x -> x.value();
      case Value.ArrayValue ignored -> renderJson(v, l);
      case Value.ObjectValue ignored -> renderJson(v, l);
      case Value.TupleValue ignored -> renderJson(v, l);
      case Value.KeywordArgumentsValue ignored -> renderJson(v, l);
      case Value.CallableValue x -> x.renderedText();
      case Value.NullValue ignored -> throw new AssertionError("unreachable: " + v);
      case Value.UndefinedValue ignored -> throw new AssertionError("unreachable: " + v);
      case Value.DeferredUndefinedValue ignored ->
          throw filterType("Cannot read properties of undefined (reading 'type')", l);
    };
  }

  private static String renderFloat(double v) {
    return JsFormat.floatString(v);
  }

  static String renderJson(Value v, SourceLocation l) {
    return JsFormat.runtimeJson(v, l);
  }

  private static Value range(List<Value> a, boolean k, SourceLocation l, RenderBudget budget) {
    Value current = argument(a, 0);
    Value stop = argument(a, 1);
    Value step = argument(a, 2);
    deferredValue(current, "value", l);
    deferredValue(stop, "value", l);
    deferredValue(step, "value", l);
    if (step instanceof Value.UndefinedValue) step = new Value.IntegerValue(1);
    if (stop instanceof Value.UndefinedValue) {
      stop = current;
      current = new Value.IntegerValue(0);
    }
    if (step instanceof Value.IntegerValue integerStep && integerStep.value() == 0
        || step instanceof Value.FloatValue floatStep && floatStep.value() == 0)
      throw new TemplateRenderException("range() step must not be zero", ErrorCategory.VALUE, l);
    boolean ascending = JsOperations.toNumber(step) > 0;
    var r = new ArrayList<Value>();
    while (ascending
        ? JsOperations.toNumber(current) < JsOperations.toNumber(stop)
        : JsOperations.toNumber(current) > JsOperations.toNumber(stop)) {
      budget.chargeRangeElement(l);
      r.add(rangeElement(current));
      current = JsOperations.add(current, step, l);
    }
    return new Value.ArrayValue(r);
  }

  private static Value rangeElement(Value value) {
    if (value instanceof Value.FloatValue number)
      return number.value() == Math.rint(number.value())
          ? new Value.IntegerValue(number.value())
          : value;
    return value;
  }

  private static Value argument(List<Value> arguments, int index) {
    if (index >= arguments.size()) return Value.UndefinedValue.INSTANCE;
    var value = arguments.get(index);
    return value instanceof Value.NullValue
            || value instanceof Value.StringValue string && string.undefinedBacked()
        ? Value.UndefinedValue.INSTANCE
        : value;
  }

  private static Value raise(List<Value> a, boolean k, SourceLocation l) {
    throw new TemplateRenderException(
        exceptionText(argument(a, 0), l), ErrorCategory.EXPLICIT_RAISE, l);
  }

  private static String exceptionText(Value value, SourceLocation location) {
    return switch (value) {
      case Value.NullValue ignored -> "";
      case Value.UndefinedValue ignored -> "";
      case Value.DeferredUndefinedValue ignored -> "";
      case Value.ArrayValue array ->
          array.values().stream()
              .map(Interpreter::nestedExceptionText)
              .reduce((left, right) -> left + "," + right)
              .orElse("");
      case Value.TupleValue tuple ->
          tuple.values().stream()
              .map(Interpreter::nestedExceptionText)
              .reduce((left, right) -> left + "," + right)
              .orElse("");
      case Value.ObjectValue ignored -> "[object Map]";
      case Value.KeywordArgumentsValue ignored -> "[object Map]";
      case Value.CallableValue x -> x.renderedText();
      case Value.BooleanValue x -> Boolean.toString(x.value());
      case Value.IntegerValue x -> JsFormat.plainString(x.value());
      case Value.FloatValue x -> JsFormat.plainString(x.value());
      case Value.StringValue x -> x.value();
    };
  }

  private static String nestedExceptionText(Value value) {
    return switch (value) {
      case Value.NullValue ignored -> "undefined";
      case Value.UndefinedValue ignored -> "undefined";
      case Value.DeferredUndefinedValue ignored -> "undefined";
      case Value.ArrayValue array ->
          array.values().stream()
              .map(Interpreter::nestedExceptionText)
              .reduce((left, right) -> left + ", " + right)
              .map(text -> "[" + text + "]")
              .orElse("[]");
      case Value.TupleValue tuple ->
          tuple.values().stream()
              .map(Interpreter::nestedExceptionText)
              .reduce((left, right) -> left + ", " + right)
              .map(text -> "[" + text + "]")
              .orElse("[]");
      case Value.ObjectValue ignored -> "[object Map]";
      case Value.KeywordArgumentsValue ignored -> "[object Map]";
      case Value.CallableValue x -> x.renderedText();
      case Value.BooleanValue x -> Boolean.toString(x.value());
      case Value.IntegerValue x -> JsFormat.plainString(x.value());
      case Value.FloatValue x -> JsFormat.plainString(x.value());
      case Value.StringValue x -> x.value();
    };
  }

  private static Value strftime(List<Value> a, boolean k, SourceLocation l, RenderOptions o) {
    // A missing format means "no positional format argument was supplied," which is not the same
    // test as
    // `a.isEmpty()`: two call shapes reach here with a non-empty `a` yet no positional value.
    // `{% call strftime_now() %}...{% endcall %}` has evaluateCallStatement append a
    // KeywordArgumentsValue bag as the last element of `a` even when there are no keywords,
    // landing at index 0 only because this call has no positionals. `strftime_now(fmt='%Y')` has
    // call()'s conditional append push the same shape at index 0 for that same no-positionals
    // reason -- its bag exists at all because its keywords are non-empty. Guard on the shape, not
    // just the count, before calling argument(a, 0) at all.
    //
    // This guard cannot distinguish "no positional value was supplied" from
    // "a positional value was supplied whose runtime type happens to be KeywordArgumentsValue" --
    // e.g. `strftime_now(namespace(a=1))`, where `namespace` returns its keyword bag verbatim
    // (Environment.namespace). Both shapes arrive here as `a=[KeywordArgumentsValue], k=false`;
    // the Callable protocol carries no positional-arity signal that could tell them apart. This
    // guard resolves that ambiguity as a missing-format TYPE error. That matches the upstream
    // TypeError family for each of these call shapes; preserving positional count through the
    // Callable signature would only improve the local message, not its observable category.
    if (a.isEmpty() || a.get(0) instanceof Value.KeywordArgumentsValue)
      throw new TemplateRenderException(
          "strftime_now() expected one string argument", ErrorCategory.TYPE, l);
    var format = argument(a, 0);
    // Upstream's convertToRuntimeValues unwraps every argument shape to its raw .value before
    // calling strftime_now, so an absent/undefined-backed argument and an explicit `none` both
    // reach the same unguarded format.replace(...) call there and only differ in which
    // "Cannot read properties of ... (reading 'replace')" message surfaces -- never a single
    // shared message. hfjinja normalizes all of these failures into the TYPE category, matching
    // the upstream TypeError family while retaining stable local messages.
    if (!(format instanceof Value.StringValue f))
      throw new TemplateRenderException(
          "strftime_now() format must be a string", ErrorCategory.TYPE, l);
    if (o.clock().isEmpty() || o.zoneId().isEmpty())
      throw new TemplateRenderException(
          "strftime_now requires clock and zone", ErrorCategory.VALUE, l);
    var z = ZonedDateTime.now(o.clock().get()).withZoneSameInstant(o.zoneId().get());
    return new Value.StringValue(PosixStrftime.format(z, f.value()));
  }
}
