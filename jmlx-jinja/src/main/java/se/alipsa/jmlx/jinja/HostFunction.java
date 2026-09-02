package se.alipsa.jmlx.jinja;

import java.util.List;

/**
 * An explicitly registered callable available to a template render.
 *
 * <p>Arguments are immutable, inert Java values: booleans, strings, {@link Long} integers that fit
 * a {@code long}, {@link Double} floats and larger integer doubles, {@code null}, and recursively
 * immutable lists or maps. A template {@code undefined} value cannot cross this boundary; calls
 * containing it fail with {@code HOST_FUNCTION}. Return values must satisfy the same closed
 * host-value boundary as render context values. Returning an exact scalar argument object unchanged
 * preserves its runtime int/float tag. Computed {@link Double} or {@link Long} results convert by
 * value: use {@link FloatResult} to retain an integral float, or {@link IntegerResult} for an
 * integral result outside the JavaScript safe-integer range.
 *
 * <p>Exception to the integer contract: an integral {@code -0} argument is delivered as {@link
 * Double}, not {@link Long}. The standard {@code Long} boxing path is subject to JVM caching, and a
 * cached box cannot be tied to one specific {@code -0} argument without risking that an unrelated
 * {@code +0} elsewhere silently aliases to it.
 */
@FunctionalInterface
public interface HostFunction {
  /**
   * Invokes this function with the template-supplied arguments.
   *
   * @param arguments the call's arguments, honoring the closed host-value boundary described above
   * @return the result, honoring the same host-value boundary
   */
  Object apply(List<Object> arguments);

  /**
   * Marks a host-function result as a template float, including integral values such as {@code
   * 2.0}.
   *
   * @param value the float value to return to the template
   * @return the marker to return from {@link #apply(List)}
   */
  static FloatResult floatResult(double value) {
    return new FloatResult(value);
  }

  /**
   * Marks a computed integral host-function result, including values outside the safe-integer
   * range.
   *
   * @param value the integral value to return to the template
   * @return the marker to return from {@link #apply(List)}
   */
  static IntegerResult integerResult(double value) {
    return new IntegerResult(value);
  }

  /**
   * Explicit float result marker for {@link #floatResult(double)}.
   *
   * @param value the float value to return to the template
   */
  record FloatResult(double value) {}

  /**
   * Explicit integer result marker for {@link #integerResult(double)}.
   *
   * @param value the integral value to return to the template
   */
  record IntegerResult(double value) {}
}
