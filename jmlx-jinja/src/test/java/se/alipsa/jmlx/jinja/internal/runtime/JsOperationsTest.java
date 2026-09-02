package se.alipsa.jmlx.jinja.internal.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.internal.Value;

class JsOperationsTest {
  @Test
  void coercesEcmaNumbersAndKeepsJavaScriptSpecialValues() {
    assertEquals(3, JsOperations.toNumber(new Value.StringValue("\u00a03\u00a0")));
    assertEquals(3, JsOperations.toNumber(new Value.StringValue("0x3")));
    assertTrue(Double.isNaN(JsOperations.toNumber(new Value.StringValue("1d"))));
    var negativeZero =
        JsOperations.arithmetic("*", new Value.FloatValue(0), new Value.FloatValue(-1));
    var floatValue = assertInstanceOf(Value.FloatValue.class, negativeZero);
    assertEquals(Double.doubleToRawLongBits(-0d), Double.doubleToRawLongBits(floatValue.value()));
    var infinity = JsOperations.arithmetic("/", new Value.FloatValue(1), new Value.FloatValue(-0d));
    assertEquals(
        Double.NEGATIVE_INFINITY, assertInstanceOf(Value.FloatValue.class, infinity).value());
  }

  @Test
  void implementsLooseEqualityRawTruthinessAndMembership() {
    assertTrue(JsOperations.looseEquals(Value.NullValue.INSTANCE, Value.UndefinedValue.INSTANCE));
    assertTrue(
        JsOperations.looseEquals(Value.StringValue.undefined(), Value.UndefinedValue.INSTANCE));
    assertFalse(
        JsOperations.looseEquals(
            Value.StringValue.undefined(), new Value.StringValue("undefined")));
    assertTrue(JsOperations.looseEquals(new Value.StringValue("1"), new Value.BooleanValue(true)));
    assertTrue(JsOperations.looseEquals(new Value.IntegerValue(1), new Value.BooleanValue(true)));
    assertTrue(JsOperations.looseEquals(new Value.IntegerValue(0), new Value.BooleanValue(false)));
    assertTrue(JsOperations.looseEquals(new Value.BooleanValue(true), new Value.FloatValue(1)));
    assertTrue(JsOperations.looseEquals(new Value.BooleanValue(false), new Value.FloatValue(0)));
    assertFalse(JsOperations.looseEquals(new Value.StringValue("x"), new Value.IntegerValue(1)));
    assertTrue(JsOperations.rawTruthy(new Value.ArrayValue(java.util.List.of())));
    assertTrue(JsOperations.rawTruthy(new Value.ObjectValue(java.util.Map.of())));
    assertFalse(
        JsOperations.contains(
            new Value.StringValue("x"), new Value.ArrayValue(java.util.List.of())));
  }
}
