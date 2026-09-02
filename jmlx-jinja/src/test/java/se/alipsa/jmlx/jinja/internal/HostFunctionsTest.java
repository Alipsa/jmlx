package se.alipsa.jmlx.jinja.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static se.alipsa.jmlx.jinja.internal.Value.ArrayValue;
import static se.alipsa.jmlx.jinja.internal.Value.FloatValue;
import static se.alipsa.jmlx.jinja.internal.Value.IntegerValue;
import static se.alipsa.jmlx.jinja.internal.Value.KeywordArgumentsValue;
import static se.alipsa.jmlx.jinja.internal.Value.NullValue;
import static se.alipsa.jmlx.jinja.internal.Value.ObjectValue;
import static se.alipsa.jmlx.jinja.internal.Value.StringValue;
import static se.alipsa.jmlx.jinja.internal.Value.TupleValue;
import static se.alipsa.jmlx.jinja.internal.Value.UndefinedValue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.HostFunction;
import se.alipsa.jmlx.jinja.RenderOptions;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateRenderException;

class HostFunctionsTest {
  private static final SourceLocation CALL_LOCATION = new SourceLocation(7, 1, 8);

  @Test
  void dispatchesOnlyRegisteredFunctionsWithDefensiveInertArguments() {
    var options =
        RenderOptions.builder()
            .hostFunction(
                "inspect",
                arguments -> {
                  assertThrows(UnsupportedOperationException.class, () -> arguments.add("later"));
                  Map<?, ?> nested = assertInstanceOf(Map.class, arguments.get(0));
                  assertEquals(Arrays.asList("first", null), nested.get("items"));
                  assertUnmodifiableMap(nested);
                  List<?> items = assertInstanceOf(List.class, nested.get("items"));
                  assertUnmodifiableList(items);
                  return Map.of("argumentCount", arguments.size());
                })
            .build();

    var result =
        HostFunctions.invoke(
            "inspect",
            function(options, "inspect"),
            List.of(
                new ObjectValue(
                    Map.of(
                        "items",
                        new ArrayValue(List.of(new StringValue("first"), NullValue.INSTANCE))))),
            false,
            CALL_LOCATION);

    assertEquals(new ObjectValue(Map.of("argumentCount", new IntegerValue(1d))), result);
  }

  @Test
  void rejectsKeywordsAndUndefinedArgumentsAtTheCallLocation() {
    var options = RenderOptions.builder().hostFunction("known", arguments -> null).build();

    var keywordError =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "known", function(options, "known"), List.of(), true, CALL_LOCATION));
    assertEquals(
        "Host function 'known' does not accept keyword arguments", keywordError.getMessage());

    var called = new AtomicBoolean();
    var undefinedOptions =
        RenderOptions.builder()
            .hostFunction(
                "known",
                arguments -> {
                  called.set(true);
                  return null;
                })
            .build();
    var undefinedError =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "known",
                    function(undefinedOptions, "known"),
                    List.of(
                        new ObjectValue(
                            Map.of("items", new ArrayValue(List.of(UndefinedValue.INSTANCE))))),
                    false,
                    CALL_LOCATION));
    assertEquals(
        "Host function 'known' cannot receive undefined value at argument 0.items[0]",
        undefinedError.getMessage());
    assertEquals(null, undefinedError.getCause());
    assertEquals(false, called.get());
  }

  @Test
  void rejectsUndefinedBackedStringArguments() {
    var options = RenderOptions.builder().hostFunction("known", arguments -> null).build();

    var error =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "known",
                    function(options, "known"),
                    List.of(StringValue.undefined()),
                    false,
                    CALL_LOCATION));

    assertEquals(
        "Host function 'known' cannot receive undefined value at argument 0", error.getMessage());
  }

  @Test
  void wrapsHostExceptionsAndInvalidReturnValuesAtTheCallLocation() {
    var functionFailure = new IllegalStateException("broken");
    var options =
        RenderOptions.builder()
            .hostFunction(
                "throws",
                arguments -> {
                  throw functionFailure;
                })
            .hostFunction("badReturn", arguments -> 'x')
            .hostFunction("wideReturn", arguments -> 1L << 60)
            .build();

    var thrownError =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "throws", function(options, "throws"), List.of(), false, CALL_LOCATION));
    assertEquals("Host function 'throws' failed", thrownError.getMessage());
    assertSame(functionFailure, thrownError.getCause());

    var returnError =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "badReturn", function(options, "badReturn"), List.of(), false, CALL_LOCATION));
    assertEquals(
        "Host function 'badReturn' returned an unsupported value", returnError.getMessage());
    var conversionCause = assertInstanceOf(TemplateRenderException.class, returnError.getCause());
    assertEquals(ErrorCategory.HOST_CONVERSION, conversionCause.category());

    var wideReturn =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "wideReturn",
                    function(options, "wideReturn"),
                    List.of(),
                    false,
                    CALL_LOCATION));
    var wideConversionCause =
        assertInstanceOf(TemplateRenderException.class, wideReturn.getCause());
    assertEquals(
        "Integer is outside the JavaScript safe-integer range: 1152921504606846976",
        wideConversionCause.getMessage());
  }

  @Test
  void preservesIntegerArgumentsAndWrapsUnexpectedConversionFailures() {
    var malformedNumber =
        new Number() {
          @Override
          public int intValue() {
            return 0;
          }

          @Override
          public long longValue() {
            return 0;
          }

          @Override
          public float floatValue() {
            return 0;
          }

          @Override
          public double doubleValue() {
            return 0;
          }

          @Override
          public String toString() {
            return null;
          }
        };
    var options =
        RenderOptions.builder()
            .hostFunction(
                "identity",
                arguments -> {
                  assertEquals(3L, arguments.get(0));
                  return arguments.get(0);
                })
            .hostFunction("malformed", arguments -> malformedNumber)
            .build();

    assertEquals(
        new IntegerValue(3d),
        HostFunctions.invoke(
            "identity",
            function(options, "identity"),
            List.of(new IntegerValue(3d)),
            false,
            CALL_LOCATION));
    var malformedError =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "malformed", function(options, "malformed"), List.of(), false, CALL_LOCATION));
    var conversionCause =
        assertInstanceOf(TemplateRenderException.class, malformedError.getCause());
    assertEquals(ErrorCategory.HOST_CONVERSION, conversionCause.category());
  }

  @Test
  void preservesAliasedAndFloatValuesAcrossHostFunctionRoundTrips() {
    var shared = new ArrayValue(List.of(new StringValue("value")));
    var options =
        RenderOptions.builder()
            .hostFunction(
                "sharing",
                arguments -> {
                  assertSame(arguments.get(0), arguments.get(1));
                  return null;
                })
            .hostFunction("identity", arguments -> arguments.get(0))
            .hostFunction("float", arguments -> HostFunction.floatResult(2d))
            .build();

    HostFunctions.invoke(
        "sharing", function(options, "sharing"), List.of(shared, shared), false, CALL_LOCATION);
    assertEquals(
        new FloatValue(2d),
        HostFunctions.invoke(
            "identity",
            function(options, "identity"),
            List.of(new FloatValue(2d)),
            false,
            CALL_LOCATION));
    assertEquals(
        new FloatValue(2d),
        HostFunctions.invoke("float", function(options, "float"), List.of(), false, CALL_LOCATION));
  }

  @Test
  void exposesComputedNumbersToHostFunctions() {
    var options =
        RenderOptions.builder()
            .hostFunction("type", arguments -> arguments.get(0).getClass().getSimpleName())
            .build();
    assertEquals(
        new StringValue("Long"),
        HostFunctions.invoke(
            "type",
            function(options, "type"),
            List.of(new IntegerValue(1L << 60)),
            false,
            CALL_LOCATION));
    assertEquals(
        new StringValue("Double"),
        HostFunctions.invoke(
            "type",
            function(options, "type"),
            List.of(new FloatValue(Double.NaN)),
            false,
            CALL_LOCATION));
  }

  @Test
  void preservesWideIntegerArgumentsAcrossAnIdentityHostFunction() {
    var options =
        RenderOptions.builder().hostFunction("identity", arguments -> arguments.get(0)).build();

    for (double value : new double[] {1L << 60, 9_007_199_254_740_994d, 1e19}) {
      assertEquals(
          new IntegerValue(value),
          HostFunctions.invoke(
              "identity",
              function(options, "identity"),
              List.of(new IntegerValue(value)),
              false,
              CALL_LOCATION));
    }
  }

  @Test
  void preservesComputedNumericResultsThatNeedRuntimeTypeMarkers() {
    var options =
        RenderOptions.builder()
            .hostFunction(
                "wide", arguments -> HostFunction.integerResult(((Long) arguments.get(0)) + 0L))
            .hostFunction("integerNan", arguments -> arguments.get(0))
            .hostFunction("negativeFloat", arguments -> HostFunction.floatResult(-0d))
            .hostFunction("negativeInteger", arguments -> HostFunction.integerResult(-0d))
            .build();

    assertEquals(
        new IntegerValue(1L << 60),
        HostFunctions.invoke(
            "wide",
            function(options, "wide"),
            List.of(new IntegerValue(1L << 60)),
            false,
            CALL_LOCATION));
    assertEquals(
        new IntegerValue(Double.NaN),
        HostFunctions.invoke(
            "integerNan",
            function(options, "integerNan"),
            List.of(new IntegerValue(Double.NaN)),
            false,
            CALL_LOCATION));
    assertEquals(
        new IntegerValue(-0d),
        HostFunctions.invoke(
            "integerNan",
            function(options, "integerNan"),
            List.of(new IntegerValue(-0d)),
            false,
            CALL_LOCATION));
    assertEquals(
        new FloatValue(-0d),
        HostFunctions.invoke(
            "negativeFloat", function(options, "negativeFloat"), List.of(), false, CALL_LOCATION));
    assertEquals(
        new IntegerValue(-0d),
        HostFunctions.invoke(
            "negativeInteger",
            function(options, "negativeInteger"),
            List.of(),
            false,
            CALL_LOCATION));
  }

  @Test
  void rejectsNonFiniteAndNonIntegralResultMarkers() {
    var options =
        RenderOptions.builder()
            .hostFunction("nanFloat", arguments -> HostFunction.floatResult(Double.NaN))
            .hostFunction(
                "infiniteFloat", arguments -> HostFunction.floatResult(Double.POSITIVE_INFINITY))
            .hostFunction("fractionalInteger", arguments -> HostFunction.integerResult(2.5))
            .build();

    var nanCause =
        assertInstanceOf(
            TemplateRenderException.class,
            assertHostFailure(
                    () ->
                        HostFunctions.invoke(
                            "nanFloat",
                            function(options, "nanFloat"),
                            List.of(),
                            false,
                            CALL_LOCATION))
                .getCause());
    assertEquals(ErrorCategory.HOST_CONVERSION, nanCause.category());
    assertEquals("Float result must be finite: NaN", nanCause.getMessage());

    var infiniteCause =
        assertInstanceOf(
            TemplateRenderException.class,
            assertHostFailure(
                    () ->
                        HostFunctions.invoke(
                            "infiniteFloat",
                            function(options, "infiniteFloat"),
                            List.of(),
                            false,
                            CALL_LOCATION))
                .getCause());
    assertEquals("Float result must be finite: Infinity", infiniteCause.getMessage());

    var fractionalCause =
        assertInstanceOf(
            TemplateRenderException.class,
            assertHostFailure(
                    () ->
                        HostFunctions.invoke(
                            "fractionalInteger",
                            function(options, "fractionalInteger"),
                            List.of(),
                            false,
                            CALL_LOCATION))
                .getCause());
    assertEquals("Integer result must be finite and integral: 2.5", fractionalCause.getMessage());
  }

  @Test
  void preservesResultMarkersNestedInsideReturnedContainers() {
    var options =
        RenderOptions.builder()
            .hostFunction("nestedFloat", arguments -> List.of(HostFunction.floatResult(2d)))
            .hostFunction(
                "nestedInteger", arguments -> Map.of("value", HostFunction.integerResult(1L << 60)))
            .build();

    assertEquals(
        new ArrayValue(List.of(new FloatValue(2d))),
        HostFunctions.invoke(
            "nestedFloat", function(options, "nestedFloat"), List.of(), false, CALL_LOCATION));
    assertEquals(
        new ObjectValue(Map.of("value", new IntegerValue(1L << 60))),
        HostFunctions.invoke(
            "nestedInteger", function(options, "nestedInteger"), List.of(), false, CALL_LOCATION));
  }

  @Test
  void preservesIdentityForAReturnedSubContainer() {
    var innerArray = new ArrayValue(List.of(new StringValue("value")));
    var outerObject = new ObjectValue(Map.of("items", innerArray));
    var options =
        RenderOptions.builder()
            .hostFunction("innerItems", arguments -> ((Map<?, ?>) arguments.get(0)).get("items"))
            .build();

    var result =
        HostFunctions.invoke(
            "innerItems",
            function(options, "innerItems"),
            List.of(outerObject),
            false,
            CALL_LOCATION);
    assertSame(innerArray, result);
  }

  @Test
  void rejectsACyclicReturnedStructure() {
    var options =
        RenderOptions.builder()
            .hostFunction(
                "cyclic",
                arguments -> {
                  var cycle = new ArrayList<>();
                  cycle.add(cycle);
                  return cycle;
                })
            .build();

    var error =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "cyclic", function(options, "cyclic"), List.of(), false, CALL_LOCATION));
    assertEquals("Host function 'cyclic' returned an unsupported value", error.getMessage());
    var cause = assertInstanceOf(TemplateRenderException.class, error.getCause());
    assertEquals(ErrorCategory.HOST_CONVERSION, cause.category());
    assertEquals("Host value graph contains a cycle", cause.getMessage());
  }

  @Test
  void rejectsACyclicRuntimeArgumentBeforeCallingTheHostFunction() {
    var cyclic = new ObjectValue(new LinkedHashMap<>());
    cyclic.values().put("self", cyclic);
    var options = RenderOptions.builder().hostFunction("known", arguments -> null).build();

    var error =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "known", function(options, "known"), List.of(cyclic), false, CALL_LOCATION));

    assertEquals("Host function 'known' cannot receive argument 0", error.getMessage());
    assertInstanceOf(IllegalStateException.class, error.getCause());
    assertEquals("Runtime value graph contains a cycle", error.getCause().getMessage());
  }

  @Test
  void acceptsKeywordArgumentObjectsAsPositionalHostArguments() {
    var options =
        RenderOptions.builder()
            .hostFunction("read", arguments -> ((Map<?, ?>) arguments.get(0)).get("a"))
            .build();
    var result =
        HostFunctions.invoke(
            "read",
            function(options, "read"),
            List.of(new KeywordArgumentsValue(Map.of("a", new IntegerValue(1)))),
            false,
            CALL_LOCATION);
    assertEquals(new IntegerValue(1), result);
  }

  @Test
  void convertsEchoedTupleArgumentsIntoArrays() {
    var options =
        RenderOptions.builder().hostFunction("identity", arguments -> arguments.get(0)).build();
    var result =
        HostFunctions.invoke(
            "identity",
            function(options, "identity"),
            List.of(new TupleValue(List.of(new IntegerValue(1)))),
            false,
            CALL_LOCATION);
    assertEquals(new ArrayValue(List.of(new IntegerValue(1))), result);
  }

  @Test
  void wrapsUnexpectedArgumentConversionFailuresAtTheCallLocation() {
    var options = RenderOptions.builder().hostFunction("known", arguments -> null).build();
    var error =
        assertHostFailure(
            () ->
                HostFunctions.invoke(
                    "known",
                    function(options, "known"),
                    new ArrayList<>(Arrays.asList((Value) null)),
                    false,
                    CALL_LOCATION));

    assertEquals("Host function 'known' cannot receive argument 0", error.getMessage());
    assertInstanceOf(NullPointerException.class, error.getCause());
  }

  private static HostFunction function(RenderOptions options, String name) {
    return options.hostFunctions().get(name);
  }

  private static TemplateRenderException assertHostFailure(
      org.junit.jupiter.api.function.Executable action) {
    var error = assertThrows(TemplateRenderException.class, action);
    assertEquals(ErrorCategory.HOST_FUNCTION, error.category());
    assertEquals(CALL_LOCATION, error.location().orElseThrow());
    return error;
  }

  @SuppressWarnings({"rawtypes", "unchecked"})
  private static void assertUnmodifiableMap(Map<?, ?> map) {
    assertThrows(UnsupportedOperationException.class, () -> ((Map) map).put("other", 1d));
  }

  @SuppressWarnings({"rawtypes", "unchecked"})
  private static void assertUnmodifiableList(List<?> list) {
    assertThrows(UnsupportedOperationException.class, () -> ((List) list).add("later"));
  }
}
