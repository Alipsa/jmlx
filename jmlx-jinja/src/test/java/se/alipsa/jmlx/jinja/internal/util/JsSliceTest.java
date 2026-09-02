package se.alipsa.jmlx.jinja.internal.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class JsSliceTest {
  @Test
  void matchesPinnedPositiveAndNegativeDirectionNormalization() {
    var values = List.of(0, 1, 2, 3, 4);
    assertEquals(List.of(1, 3), JsSlice.slice(values, 1d, 4d, 2d));
    assertEquals(List.of(1, 2, 3), JsSlice.slice(values, -4d, -1d, null));
    assertEquals(List.of(4, 3, 2, 1, 0), JsSlice.slice(values, null, null, -1d));
    assertEquals(List.of(), JsSlice.slice(values, 5d, 1d, null));
    assertEquals(List.of(), JsSlice.slice(values, null, null, 0d));
  }

  @Test
  void clampsLargeBoundsBeforeNarrowing() {
    var values = List.of(0, 1, 2);
    assertEquals(List.of(), JsSlice.slice(values, 3_000_000_000d, null, null));
    assertEquals(values, JsSlice.slice(values, -3_000_000_000d, null, null));
  }
}
