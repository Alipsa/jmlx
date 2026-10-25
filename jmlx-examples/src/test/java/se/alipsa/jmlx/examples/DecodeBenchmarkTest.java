package se.alipsa.jmlx.examples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

/** Regression coverage for paired benchmark statistics. */
class DecodeBenchmarkTest {
  @Test
  void ratiosPreserveIterationPairsBeforeSorting() {
    // Independently sorting these durations would produce a median ratio of 1.
    var stats =
        DecodeBenchmark.seamOverOld(
            List.of(200L, 100L, 600L, 400L, 1000L, 300L, 1400L),
            List.of(100L, 200L, 300L, 400L, 500L, 600L, 700L));
    assertEquals(0.5, stats.get("p10"));
    assertEquals(2.0, stats.get("median"));
    assertEquals(2.0, stats.get("p90"));
  }

  @Test
  void invalidPairsAreRejected() {
    assertThrows(
        IllegalArgumentException.class, () -> DecodeBenchmark.seamOverOld(List.of(), List.of()));
    assertThrows(
        IllegalArgumentException.class,
        () -> DecodeBenchmark.seamOverOld(List.of(1L), List.of(1L, 2L)));
    assertThrows(
        IllegalArgumentException.class,
        () -> DecodeBenchmark.seamOverOld(List.of(1L), List.of(0L)));
  }
}
