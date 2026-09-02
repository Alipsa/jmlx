package se.alipsa.jmlx.jinja.internal.util;

import java.util.ArrayList;
import java.util.List;

/** Python-style sequence slicing with the pinned runtime's JavaScript number semantics. */
public final class JsSlice {
  private JsSlice() {}

  /**
   * Slices {@code values} according to the pinned upstream {@code utils.ts::slice} rules.
   *
   * @param <T> element type
   * @param values input sequence
   * @param start start bound, or null when omitted
   * @param stop exclusive stop bound, or null when omitted
   * @param step step bound, or null when omitted
   * @return a new sliced sequence
   */
  public static <T> List<T> slice(List<T> values, Double start, Double stop, Double step) {
    double actualStep = step == null ? 1 : step;
    double direction = Math.signum(actualStep);
    int length = values.size();
    double actualStart;
    double actualStop;
    if (direction >= 0) {
      actualStart = positiveBound(start == null ? 0 : start, length);
      actualStop = positiveBound(stop == null ? length : stop, length);
    } else {
      actualStart = negativeStartBound(start == null ? length - 1 : start, length);
      actualStop = negativeStopBound(stop == null ? -1 : stop, length);
    }
    var result = new ArrayList<T>();
    for (double index = actualStart;
        direction * index < direction * actualStop;
        index += actualStep) {
      result.add(values.get((int) index));
    }
    return result;
  }

  private static double positiveBound(double bound, int length) {
    return bound < 0 ? Math.max(length + bound, 0) : Math.min(bound, length);
  }

  private static double negativeStartBound(double bound, int length) {
    return bound < 0 ? Math.max(length + bound, -1) : Math.min(bound, length - 1);
  }

  private static double negativeStopBound(double bound, int length) {
    return bound < -1 ? Math.max(length + bound, -1) : Math.min(bound, length - 1);
  }
}
