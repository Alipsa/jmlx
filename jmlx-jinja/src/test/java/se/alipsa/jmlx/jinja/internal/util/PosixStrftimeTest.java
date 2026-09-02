package se.alipsa.jmlx.jinja.internal.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import org.junit.jupiter.api.Test;

class PosixStrftimeTest {
  @Test
  void formatsEachPinnedDirectiveWithZeroPadding() {
    var single = ZonedDateTime.of(2026, 1, 5, 3, 7, 0, 0, ZoneOffset.UTC);
    assertEquals("2026", PosixStrftime.format(single, "%Y"));
    assertEquals("01", PosixStrftime.format(single, "%m"));
    assertEquals("05", PosixStrftime.format(single, "%d"));
    assertEquals("Jan", PosixStrftime.format(single, "%b"));
    assertEquals("January", PosixStrftime.format(single, "%B"));
    assertEquals("03", PosixStrftime.format(single, "%H"));
    assertEquals("07", PosixStrftime.format(single, "%M"));
    assertEquals("%", PosixStrftime.format(single, "%%"));

    var doubleDigits = ZonedDateTime.of(2026, 8, 21, 9, 5, 0, 0, ZoneOffset.UTC);
    assertEquals("Aug August", PosixStrftime.format(doubleDigits, "%b %B"));
    assertEquals("2026-08-21 09:05", PosixStrftime.format(doubleDigits, "%Y-%m-%d %H:%M"));
  }

  @Test
  void matchesPinnedPercentEdgeVector() {
    var dateTime = ZonedDateTime.of(2026, 8, 21, 9, 5, 0, 0, ZoneOffset.UTC);
    assertEquals("2026|%Y|%2026|%q%|%", PosixStrftime.format(dateTime, "%Y|%%Y|%%%Y|%q%%|%"));
  }

  @Test
  void passesThroughUnknownDirectivesAndNonAsciiLiterally() {
    var dateTime = ZonedDateTime.of(2026, 8, 21, 9, 5, 0, 0, ZoneOffset.UTC);
    assertEquals("%x", PosixStrftime.format(dateTime, "%x"));
    assertEquals("abc%", PosixStrftime.format(dateTime, "abc%"));
    assertEquals("café 日本語 😀", PosixStrftime.format(dateTime, "café 日本語 😀"));
  }
}
