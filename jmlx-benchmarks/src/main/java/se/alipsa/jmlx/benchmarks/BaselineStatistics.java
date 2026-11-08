package se.alipsa.jmlx.benchmarks;

import java.util.List;

/**
 * Type-7 (linear interpolation, numpy "R" convention) quantiles over the frozen phase 8.2 baseline
 * samples (req/plans/phase8-2-reference-findings.md section 10: "linear interpolation (numpy
 * R/type-7) over the per-process warm medians"). Pure Java, headline reports only.
 */
final class BaselineStatistics {
  private BaselineStatistics() {}

  /** One sample's quantile summary; empty samples carry NaNs. */
  record Quantiles(double p10, double median, double p90, int n) {}

  /** Type-7 quantiles of the values (mutates a copy; the input is left untouched). */
  static Quantiles quantiles(List<Double> values) {
    if (values.isEmpty()) {
      return new Quantiles(Double.NaN, Double.NaN, Double.NaN, 0);
    }
    List<Double> sorted = values.stream().sorted().toList();
    return new Quantiles(
        quantile(sorted, 0.10), quantile(sorted, 0.50), quantile(sorted, 0.90), sorted.size());
  }

  /**
   * Linear interpolation: h = (n - 1) q, the value at floor(h) plus the fractional part of (value
   * at ceil(h) minus value at floor(h)). Matches numpy's default percentile.
   */
  static double quantile(List<Double> sorted, double q) {
    if (sorted.size() == 1) {
      return sorted.get(0);
    }
    double h = (sorted.size() - 1) * q;
    int lower = (int) Math.floor(h);
    int upper = (int) Math.ceil(h);
    if (lower == upper) {
      return sorted.get(lower);
    }
    double fraction = h - lower;
    return sorted.get(lower) + fraction * (sorted.get(upper) - sorted.get(lower));
  }
}
