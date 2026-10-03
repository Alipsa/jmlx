package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;

class Tf32BenchmarkTest {
  @Test
  void comparisonHonorsAlternatingModeOrderAndUsesPairedRatios() {
    ObjectMapper mapper = new ObjectMapper();
    var result =
        Tf32Benchmark.comparison(
            List.of(
                mapper.readTree("{\"tf32\":\"1\",\"prefill_ns\":[10,20]}"),
                mapper.readTree("{\"tf32\":\"0\",\"prefill_ns\":[20,40]}"),
                mapper.readTree("{\"tf32\":\"0\",\"prefill_ns\":[300,600]}"),
                mapper.readTree("{\"tf32\":\"1\",\"prefill_ns\":[100,200]}")),
            "prefill_ns",
            2);
    assertEquals(60.0, result.get("enabled_median_ns"));
    assertEquals(170.0, result.get("disabled_median_ns"));
    assertEquals(List.of(2.0, 3.0), result.get("paired_ratios"));
    assertEquals(2.5, result.get("disabled_over_enabled_median_ratio"));
  }

  @Test
  void rejectsEmptyMeasurementsAndInvalidWorkloads() {
    assertThrows(IllegalArgumentException.class, () -> Tf32Benchmark.median(List.of()));
    assertThrows(
        IllegalArgumentException.class,
        () -> Tf32Benchmark.Options.parse(new String[] {"model", "out", "0"}));
    assertThrows(
        IllegalArgumentException.class,
        () -> Tf32Benchmark.Options.parse(new String[] {"model", "out", "1", "1", "1", "-1"}));
    assertEquals(
        0,
        Tf32Benchmark.Options.parse(new String[] {"model", "out", "1", "1", "1", "0"}).warmups());
  }
}
