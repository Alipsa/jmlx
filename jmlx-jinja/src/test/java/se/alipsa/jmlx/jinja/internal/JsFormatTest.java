package se.alipsa.jmlx.jinja.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class JsFormatTest {
  @Test
  void formatsFiniteAndNonFiniteNumbers() {
    assertEquals("1.5", JsFormat.jsonString(1.5d));
    assertEquals("null", JsFormat.jsonString(Double.NaN));
    assertEquals("Infinity", JsFormat.plainString(Double.POSITIVE_INFINITY));
    assertEquals("-Infinity", JsFormat.plainString(Double.NEGATIVE_INFINITY));
    assertEquals("NaN", JsFormat.plainString(Double.NaN));
  }

  @Test
  void quotesControlsAndUnpairedSurrogates() {
    assertEquals("\"a\\nb\\\"c\\\\\"", JsFormat.quote("a\nb\"c\\"));
    assertEquals("\"x\\ud800y\"", JsFormat.quote("x\uD800y"));
  }
}
