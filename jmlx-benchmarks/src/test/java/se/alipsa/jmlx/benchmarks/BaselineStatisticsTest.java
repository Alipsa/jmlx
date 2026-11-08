package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Hand-computed type-7 (numpy "R") quantile values for the summary builder. The expected numbers
 * follow from h = (n - 1) q with linear interpolation between the floor and ceil samples.
 */
class BaselineStatisticsTest {

  @Test
  void fourSamples() {
    BaselineStatistics.Quantiles q = BaselineStatistics.quantiles(List.of(1.0, 2.0, 3.0, 4.0));
    // h(p10) = 0.3 -> 1 + 0.3 * (2 - 1); h(median) = 1.5; h(p90) = 2.7.
    assertEquals(1.3, q.p10(), 1e-12);
    assertEquals(2.5, q.median(), 1e-12);
    assertEquals(3.7, q.p90(), 1e-12);
    assertEquals(4, q.n());
  }

  @Test
  void twentySamples() {
    List<Double> values = new ArrayList<>();
    for (int i = 1; i <= 20; i++) {
      values.add((double) i);
    }
    BaselineStatistics.Quantiles q = BaselineStatistics.quantiles(values);
    // h(p10) = 1.9 -> 2 + 0.9; h(median) = 9.5; h(p90) = 17.1 -> 18 + 0.1.
    assertEquals(2.9, q.p10(), 1e-12);
    assertEquals(10.5, q.median(), 1e-12);
    assertEquals(18.1, q.p90(), 1e-12);
    assertEquals(20, q.n());
  }

  @Test
  void exactPositionHit() {
    BaselineStatistics.Quantiles q = BaselineStatistics.quantiles(List.of(10.0, 20.0, 30.0));
    // h(median) = 1 exactly -> the sample itself, no interpolation.
    assertEquals(20.0, q.median(), 1e-12);
    assertEquals(12.0, q.p10(), 1e-12);
    assertEquals(28.0, q.p90(), 1e-12);
    assertEquals(3, q.n());
  }

  @Test
  void singleSample() {
    BaselineStatistics.Quantiles q = BaselineStatistics.quantiles(List.of(7.5));
    assertEquals(7.5, q.p10(), 1e-12);
    assertEquals(7.5, q.median(), 1e-12);
    assertEquals(7.5, q.p90(), 1e-12);
    assertEquals(1, q.n());
  }

  @Test
  void emptySamples() {
    BaselineStatistics.Quantiles q = BaselineStatistics.quantiles(List.of());
    assertTrue(Double.isNaN(q.p10()));
    assertTrue(Double.isNaN(q.median()));
    assertTrue(Double.isNaN(q.p90()));
    assertEquals(0, q.n());
  }

  @Test
  void inputOrderIsNotMutated() {
    List<Double> values = new ArrayList<>(List.of(4.0, 1.0, 3.0, 2.0));
    List<Double> snapshot = new ArrayList<>(values);
    BaselineStatistics.quantiles(values);
    assertEquals(snapshot, values);
  }
}
