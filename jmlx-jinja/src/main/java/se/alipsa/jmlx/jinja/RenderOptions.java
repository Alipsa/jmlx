package se.alipsa.jmlx.jinja;

import java.time.Clock;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Immutable render-time options with optional clock/zone settings and explicitly named host
 * functions.
 */
@SuppressWarnings("doclint:missing")
public final class RenderOptions {
  private static final Set<String> BUILTIN_GLOBALS =
      Set.of(
          "false",
          "true",
          "none",
          "raise_exception",
          "range",
          "strftime_now",
          "True",
          "False",
          "None",
          "namespace");

  /** Render options with no clock/zone override and no host functions. */
  public static final RenderOptions DEFAULT =
      new RenderOptions(null, null, Map.of(), 10_000_000, 1_000_000, 10_000_000, 500);

  private final Clock clock;
  private final ZoneId zoneId;
  private final Map<String, HostFunction> hostFunctions;
  private final int maxSteps;
  private final int maxLoopIterations;
  private final int maxOutputLength;
  private final int maxMacroDepth;

  private RenderOptions(
      Clock clock,
      ZoneId zoneId,
      Map<String, HostFunction> hostFunctions,
      int maxSteps,
      int maxLoopIterations,
      int maxOutputLength,
      int maxMacroDepth) {
    this.clock = clock;
    this.zoneId = zoneId;
    this.hostFunctions = Collections.unmodifiableMap(new LinkedHashMap<>(hostFunctions));
    this.maxSteps = maxSteps;
    this.maxLoopIterations = maxLoopIterations;
    this.maxOutputLength = maxOutputLength;
    this.maxMacroDepth = maxMacroDepth;
  }

  /**
   * Starts construction of immutable render options.
   *
   * @return a new builder with default options
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Returns the optional caller-supplied clock.
   *
   * @return the clock, or empty; {@code strftime_now} requires an explicit clock at first use
   */
  public Optional<Clock> clock() {
    return Optional.ofNullable(clock);
  }

  /**
   * Returns the optional caller-supplied time zone.
   *
   * @return the time zone, or empty; {@code strftime_now} requires an explicit zone at first use
   */
  public Optional<ZoneId> zoneId() {
    return Optional.ofNullable(zoneId);
  }

  /**
   * Returns the immutable functions that are callable by name from a template.
   *
   * @return the host functions, keyed by their template-visible name
   */
  public Map<String, HostFunction> hostFunctions() {
    return hostFunctions;
  }

  /** Returns the maximum number of render steps allowed. */
  public int maxSteps() {
    return maxSteps;
  }

  /** Returns the maximum number of loop iterations allowed. */
  public int maxLoopIterations() {
    return maxLoopIterations;
  }

  /**
   * Returns the maximum cumulative rendered character count allowed, not a bound on final output
   * size: a construct that re-renders already-charged text — for example {@code {% set %}...{%
   * endset %}}, {@code {% filter %}...{% endfilter %}}, or {@code {% call %}...{% endcall %}} —
   * charges that text again each time it is re-emitted, so the visible output can be smaller than
   * this limit by a factor of nesting depth.
   */
  public int maxOutputLength() {
    return maxOutputLength;
  }

  /**
   * Returns the maximum macro/call-block invocation depth allowed. This bounds invocation count,
   * not total interpreter recursion depth: a macro body that itself nests control-flow constructs
   * (nested loops, conditionals, or call blocks) consumes multiple interpreter stack frames per
   * invocation, so this limit does not by itself guarantee the native JVM stack cannot be exhausted
   * first for a sufficiently deep-nested body. Rendering still fails with {@link
   * ErrorCategory#RESOURCE_LIMIT} rather than an escaped {@link StackOverflowError} in that case —
   * the interpreter catches it at the top of {@code render()} as a backstop — but that backstop
   * does not bound how much work is done before failing the way this limit does for straight
   * recursion.
   */
  public int maxMacroDepth() {
    return maxMacroDepth;
  }

  /** Builder for {@link RenderOptions}. */
  public static final class Builder {
    private Clock clock;
    private ZoneId zoneId;
    private final List<HostFunctionRegistration> hostFunctions = new ArrayList<>();
    private int maxSteps = 10_000_000;
    private int maxLoopIterations = 1_000_000;
    private int maxOutputLength = 10_000_000;
    private int maxMacroDepth = 500;

    private Builder() {}

    /**
     * Overrides the clock used for time-dependent host functions.
     *
     * @param clock the clock to use
     * @return this builder
     */
    public Builder clock(Clock clock) {
      this.clock = Objects.requireNonNull(clock, "clock");
      return this;
    }

    /**
     * Overrides the time zone used for time-dependent host functions.
     *
     * @param zoneId the time zone to use
     * @return this builder
     */
    public Builder zoneId(ZoneId zoneId) {
      this.zoneId = Objects.requireNonNull(zoneId, "zoneId");
      return this;
    }

    public Builder maxSteps(int value) {
      maxSteps = positive(value, "maxSteps");
      return this;
    }

    public Builder maxLoopIterations(int value) {
      maxLoopIterations = positive(value, "maxLoopIterations");
      return this;
    }

    /**
     * Overrides the maximum cumulative rendered character count allowed before rendering fails with
     * {@link ErrorCategory#RESOURCE_LIMIT}. See {@link RenderOptions#maxOutputLength()} for why
     * this is a bound on cumulative charges, not on final output size.
     *
     * @param value the new limit; must be positive
     * @return this builder
     */
    public Builder maxOutputLength(int value) {
      maxOutputLength = positive(value, "maxOutputLength");
      return this;
    }

    /**
     * Overrides the maximum macro/call-block invocation depth allowed before rendering fails with
     * {@link ErrorCategory#RESOURCE_LIMIT}.
     *
     * @param value the new limit; must be positive
     * @return this builder
     */
    public Builder maxMacroDepth(int value) {
      maxMacroDepth = positive(value, "maxMacroDepth");
      return this;
    }

    /**
     * Registers a function under a template-visible name.
     *
     * @param name the template-visible name; must be a valid template identifier
     * @param function the function to register
     * @return this builder
     */
    public Builder hostFunction(String name, HostFunction function) {
      if (name == null || !name.matches("[A-Za-z_][A-Za-z0-9_]*")) {
        throw new IllegalArgumentException("Host function name must be a template identifier");
      }
      hostFunctions.add(
          new HostFunctionRegistration(name, Objects.requireNonNull(function, "function")));
      return this;
    }

    /**
     * Validates registrations and creates immutable render options.
     *
     * @return the immutable render options
     */
    public RenderOptions build() {
      var functions = new LinkedHashMap<String, HostFunction>();
      for (var registration : hostFunctions) {
        if (BUILTIN_GLOBALS.contains(registration.name)) {
          throw new IllegalArgumentException(
              "Host function name collides with built-in global: " + registration.name);
        }
        if (functions.putIfAbsent(registration.name, registration.function) != null) {
          throw new IllegalArgumentException("Duplicate host function name: " + registration.name);
        }
      }
      return new RenderOptions(
          clock, zoneId, functions, maxSteps, maxLoopIterations, maxOutputLength, maxMacroDepth);
    }
  }

  private static int positive(int value, String name) {
    if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
    return value;
  }

  private record HostFunctionRegistration(String name, HostFunction function) {}
}
