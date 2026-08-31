package se.alipsa.jmlx.jinja.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static se.alipsa.jmlx.jinja.internal.Value.ArrayValue;
import static se.alipsa.jmlx.jinja.internal.Value.FloatValue;
import static se.alipsa.jmlx.jinja.internal.Value.IntegerValue;
import static se.alipsa.jmlx.jinja.internal.Value.NullValue;
import static se.alipsa.jmlx.jinja.internal.Value.ObjectValue;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.HostFunction;
import se.alipsa.jmlx.jinja.TemplateRenderException;

class ValuesTest {
  @Test
  void acceptsJsonStyleNumbersWhenTheirShortestJsFormMatches() {
    assertEquals(new FloatValue(0.7d), Values.fromHost(new BigDecimal("0.7")));
    assertEquals(new IntegerValue(42d), Values.fromHost(new BigDecimal("42.0")));
  }

  @Test
  void rejectsDecimalNarrowingAndUnsafeIntegers() {
    assertConversionFailure(new BigDecimal("0.1234567890123456789"));
    assertConversionFailure(new BigDecimal("9007199254740992"));
  }

  @Test
  void reportsUnsafeIntegerRangeBeforeRepresentability() {
    // An integral value above 2^53 is both unsafe and, once rounded to a double, no longer
    // matches its exact decimal text. The safe-integer diagnostic must win: it is the more
    // specific and more useful message, and
    // HostFunctionsTest.wrapsHostExceptionsAndInvalidReturnValuesAtTheCallLocation pins the same
    // ordering for a computed Long return.
    var error = assertConversionFailure(new BigInteger("9007199254740993"));
    assertEquals(
        "Integer is outside the JavaScript safe-integer range: 9007199254740993",
        error.getMessage());
  }

  @Test
  void rejectsHugeExponentsWithoutExpandingThem() {
    assertConversionFailure(new BigDecimal("1E+100000000"));
  }

  @Test
  void acceptsSubnormalNumbersUsingTheJsShortestForm() {
    assertEquals(new FloatValue(Double.MIN_VALUE), Values.fromHost(new BigDecimal("5E-324")));
    assertEquals(new FloatValue(Double.MIN_VALUE), Values.fromHost(Double.MIN_VALUE));
  }

  @Test
  void convertsFloatThroughItsCanonicalDecimalRepresentation() {
    assertEquals(new FloatValue(0.1d), Values.fromHost(0.1f));
    assertEquals(new FloatValue(1.1d), Values.fromHost(1.1f));
  }

  @Test
  void rejectsNonFiniteNumbersWithAFiniteDiagnostic() {
    var nan = assertConversionFailure(Double.NaN);
    assertEquals("Number must be finite: NaN", nan.getMessage());
    var infinity = assertConversionFailure(Float.POSITIVE_INFINITY);
    assertEquals("Number must be finite: Infinity", infinity.getMessage());
  }

  @Test
  void wrapsMalformedNumberTextAsAConversionFailure() {
    Number malformed =
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

    assertConversionFailure(malformed);
  }

  @Test
  void rejectsFloatResultOutsideHostFunctionReturns() {
    assertConversionFailure(HostFunction.floatResult(2d));
  }

  @Test
  void rejectsIntegerResultOutsideHostFunctionReturns() {
    assertConversionFailure(HostFunction.integerResult(1L << 60));
  }

  @Test
  void supportsCommonNumberSubclassesAndRejectsUnsupportedHostValues() {
    assertEquals(new IntegerValue(7d), Values.fromHost(new AtomicInteger(7)));
    assertEquals(new IntegerValue(42d), Values.fromHost(new BigInteger("42")));
    var error = assertConversionFailure('x');
    assertEquals("Unsupported host value type: java.lang.Character", error.getMessage());
  }

  @Test
  void normalizesNegativeZero() {
    assertEquals(new IntegerValue(0d), Values.fromHost(-0.0d));
    assertEquals(0L, Double.doubleToRawLongBits(((IntegerValue) Values.fromHost(-0.0d)).value()));
    assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(new FloatValue(-0.0d).value()));
    assertEquals(Double.NEGATIVE_INFINITY, 1d / new FloatValue(-0.0d).value());
  }

  @Test
  void pinsFloatBoundaryConversions() {
    assertEquals(
        new IntegerValue(Double.parseDouble(Float.toString(Float.MAX_VALUE))),
        Values.fromHost(Float.MAX_VALUE));
    assertEquals(new FloatValue(1.4e-45), Values.fromHost(Float.MIN_VALUE));
    assertEquals(new IntegerValue(0d), Values.fromHost(-0.0f));
  }

  @Test
  void rejectsNullCollectionValuesWithAUsefulMessage() {
    var arrayError = assertThrows(NullPointerException.class, () -> new ArrayValue(null));
    assertEquals("values", arrayError.getMessage());
    var objectError = assertThrows(NullPointerException.class, () -> new ObjectValue(null));
    assertEquals("values", objectError.getMessage());
  }

  @Test
  void distinguishesNullAndCopiesNestedValues() {
    var value =
        assertInstanceOf(
            ObjectValue.class, Values.fromHost(Map.of("items", java.util.Arrays.asList(1, null))));
    var array = assertInstanceOf(ArrayValue.class, value.values().get("items"));
    assertEquals(NullValue.INSTANCE, array.values().get(1));
  }

  @Test
  void rejectsCyclesAndNonStringMapKeys() {
    var cycle = new java.util.ArrayList<>();
    cycle.add(cycle);
    assertConversionFailure(cycle);
    assertConversionFailure(Map.of(1, "no"));
  }

  @Test
  void reusesConvertedDagNodesInsteadOfRewalkingThem() {
    var shared = List.of("value");
    var converted = assertInstanceOf(ArrayValue.class, Values.fromHost(List.of(shared, shared)));
    assertEquals(converted.values().get(0), converted.values().get(1));
    org.junit.jupiter.api.Assertions.assertSame(
        converted.values().get(0), converted.values().get(1));
  }

  @Test
  void convertsDeepSharedDagWithoutExponentialWork() {
    Object graph = List.of("leaf");
    for (int depth = 0; depth < 40; depth++) {
      graph = List.of(graph, graph);
    }
    assertInstanceOf(ArrayValue.class, Values.fromHost(graph));
  }

  @Test
  void convertsDeepSharedMutableDagWithoutExpandingItIntoATree() {
    Map<String, Object> graph = Map.of("value", 0);
    for (int depth = 0; depth < 26; depth++) {
      graph = Map.of("a", graph, "b", graph);
    }
    var root = graph;
    var error =
        assertThrows(TemplateRenderException.class, () -> Values.fromHost(Map.of("root", root)));
    assertEquals(ErrorCategory.HOST_CONVERSION, error.category());
    assertEquals("Host value graph is too large after mutable copy isolation", error.getMessage());
  }

  private static TemplateRenderException assertConversionFailure(Object input) {
    var error = assertThrows(TemplateRenderException.class, () -> Values.fromHost(input));
    assertEquals(ErrorCategory.HOST_CONVERSION, error.category());
    return error;
  }
}
