package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;

/** Decoder checks run even without a staged native runtime. */
class OracleFixtureReaderTest {
  @Test
  void decodesNestedFiniteAndNonfiniteValuesAndRejectsUnknownSentinels() {
    ObjectMapper mapper = new ObjectMapper();
    assertArrayEquals(
        new float[] {1, -2.5f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY},
        OracleFixtureReader.floats(
            mapper.readTree("[[1,-2.5],[\"NaN\",\"Infinity\",\"-Infinity\"]]")));
    assertThrows(
        IllegalArgumentException.class,
        () -> OracleFixtureReader.floats(mapper.readTree("[\"unknown\"]")));
    assertThrows(
        IllegalArgumentException.class,
        () -> OracleFixtureReader.floats(mapper.readTree("[null]")));
  }
}
