package se.alipsa.jmlx.nn;

import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.SequencedMap;
import java.util.Set;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Base class for neural-network layers: a tree of named parameters ({@link MLXArray}s) and named submodules, built up
 * by a subclass's constructor via {@link #param(String, MLXArray)} and {@link #child(String, Module)}. See
 * req/phase4-plan.md §5-6 for the design this class implements.
 *
 * <p>
 * {@link #param(String, MLXArray)} registers a value and returns it, but subclasses must NOT cache that return value
 * into a field for later use in {@code forward()} -- they must call {@link #param(String)} again, fresh, every time
 * they need the current value. This is not a style preference: {@link #rebind} (used by a not-yet-built autograd
 * feature, M2's {@code ModuleGrad}, to swap in traced primals around a loss call) updates only this class's internal
 * {@code params} map. A layer that cached the array returned by the setter would keep reading the original value
 * forever, silently disconnecting it from any later rebind -- "correct shapes, no exception" is exactly the failure
 * mode this contract exists to avoid. A cached scalar field (e.g. a {@code float eps}) has no such hazard, since it is
 * never an {@link MLXArray} and is never rebound.
 */
public abstract class Module {

  private final MLXScope scope;
  private final LinkedHashMap<String, MLXArray> params = new LinkedHashMap<>();
  private final LinkedHashMap<String, Module> children = new LinkedHashMap<>();
  private boolean frozen;

  /** Creates a module whose parameters and submodules will be allocated into {@code scope}. */
  protected Module(MLXScope scope) {
    this.scope = scope;
  }

  /**
   * Registers {@code value} as this module's own parameter named {@code name} and returns it. Subclasses may use the
   * returned value only to validate it at construction time (e.g. read its shape) -- never to cache it for
   * {@code forward()}; see this class's javadoc for why.
   *
   * @throws IllegalStateException if this module is frozen, or if {@code name} is already registered
   * @throws NullPointerException if {@code value} is {@code null}
   */
  protected final MLXArray param(String name, MLXArray value) {
    Objects.requireNonNull(value, "param \"" + name + "\": value must not be null");
    if (frozen) {
      throw new IllegalStateException("Module is frozen: cannot register parameter \"" + name + "\"");
    }
    if (params.containsKey(name)) {
      throw new IllegalStateException("parameter \"" + name + "\" is already registered");
    }
    params.put(name, value);
    return value;
  }

  /**
   * Looks up the current value bound to this module's own parameter named {@code name}. Call this fresh from
   * {@code forward()} rather than caching the value returned by {@link #param(String, MLXArray)} -- see this class's
   * javadoc.
   *
   * @throws IllegalStateException if {@code name} is not a registered parameter of this module
   */
  protected final MLXArray param(String name) {
    if (!params.containsKey(name)) {
      throw new IllegalStateException("unknown parameter \"" + name + "\"");
    }
    return params.get(name);
  }

  /**
   * Registers {@code module} as this module's own submodule named {@code name} and returns it.
   *
   * @throws IllegalStateException if this module is frozen, or if {@code name} is already registered
   */
  protected final <M extends Module> M child(String name, M module) {
    if (frozen) {
      throw new IllegalStateException("Module is frozen: cannot register child \"" + name + "\"");
    }
    if (children.containsKey(name)) {
      throw new IllegalStateException("child \"" + name + "\" is already registered");
    }
    children.put(name, module);
    return module;
  }

  /**
   * The {@link MLXScope} passed to this module's constructor -- WEIGHTS live here. {@code forward()} implementations
   * must never allocate directly into this scope: a creation op called with {@code scope()}, or a single-operand op on
   * a parameter (e.g. {@code astype(W)}, {@code transpose(W)}), allocates its result here and leaks it once per forward
   * call, since this scope lives for the model's lifetime rather than closing per step. Target the activation's own
   * scope instead (e.g. {@code x.scope()}), or use an op's explicit-target overload when the operand is a parameter
   * rather than an activation.
   */
  protected final MLXScope scope() {
    return scope;
  }

  /**
   * Every parameter in this module's subtree, keyed by dotted path from this module (own parameters first, in insertion
   * order, then each child's own parameters recursively, in child-insertion order). The returned map is an immutable
   * snapshot: later registrations or rebinds are not reflected in a previously returned map.
   */
  public final SequencedMap<String, MLXArray> parameters() {
    SequencedMap<String, MLXArray> out = new LinkedHashMap<>();
    collectParameters("", out);
    return Collections.unmodifiableSequencedMap(out);
  }

  private void collectParameters(String prefix, SequencedMap<String, MLXArray> out) {
    for (Map.Entry<String, MLXArray> entry : params.entrySet()) {
      out.put(prefix + entry.getKey(), entry.getValue());
    }
    for (Map.Entry<String, Module> entry : children.entrySet()) {
      entry.getValue().collectParameters(prefix + entry.getKey() + ".", out);
    }
  }

  /**
   * Writes each entry's value into the parameter at its dotted path, then calls {@link #onParametersUpdated()} on
   * exactly the modules whose own {@code params} map was written to (depth-first: this module first, then
   * {@code children} in insertion order). Every write completes before any notification runs -- spec §5:
   * "all-writes-then-all-notifies, not interleaved".
   *
   * @throws IllegalArgumentException if any path does not resolve to a registered parameter
   * @throws NullPointerException if any value is {@code null}
   */
  public final void update(Map<String, MLXArray> byPath) {
    Set<Module> touched = new HashSet<>();
    for (Map.Entry<String, MLXArray> entry : byPath.entrySet()) {
      touched.add(resolveAndWrite(entry.getKey(), entry.getKey(), entry.getValue()));
    }
    notifyDepthFirst(touched);
  }

  private void notifyDepthFirst(Set<Module> touched) {
    if (touched.contains(this)) {
      onParametersUpdated();
    }
    for (Module child : children.values()) {
      child.notifyDepthFirst(touched);
    }
  }

  /**
   * Freezes this module and, cascading, every module already registered as a descendant of it: after this call,
   * {@link #param(String, MLXArray)} and {@link #child(String, Module)} throw on this module and on every module
   * reachable through {@code children} at the time of the call. The cascade matters because a nested submodule
   * registered before its ancestor's {@code freeze()} call would otherwise still accept new registrations after the
   * ancestor believes the whole tree is frozen.
   */
  public final void freeze() {
    frozen = true;
    for (Module child : children.values()) {
      child.freeze();
    }
  }

  /**
   * Writes each entry's value into the parameter at its dotted path, exactly like {@link #update}, but never calls
   * {@link #onParametersUpdated()} -- spec §6: "rebind must NOT fire {@code onParametersUpdated()}". Legal even after
   * {@link #freeze()}, since {@link #resolveAndWrite} never checks {@code frozen}.
   *
   * @throws IllegalArgumentException if any path does not resolve to a registered parameter
   * @throws NullPointerException if any value is {@code null}
   */
  public final void rebind(SequencedMap<String, MLXArray> values) {
    for (Map.Entry<String, MLXArray> entry : values.entrySet()) {
      resolveAndWrite(entry.getKey(), entry.getKey(), entry.getValue());
    }
  }

  /**
   * Called after {@link #update} writes one or more of this module's own parameters. Empty by default; never called
   * from {@link #rebind}. Do NOT use this to cache a derived {@link MLXArray} view (e.g. a transposed weight) in a
   * field: a not-yet-built autograd feature ({@code ModuleGrad}) will call {@link #rebind} to swap in traced primals
   * around a loss call and restore them afterward without notifying this hook, so a cached view here would either leak
   * (recomputed once per step in a scope that never closes, if the hook fired on restore) or dangle (if the recompute
   * were suppressed). Read the current value fresh via {@link #param(String)} instead -- see this class's javadoc and
   * req/phase4-plan.md §2 for the full analysis.
   */
  protected void onParametersUpdated() {}

  // fullPath is threaded through unchanged so an exception thrown after
  // descending into a child still names the original dotted path the
  // caller passed in, not just the trailing segment local to this module --
  // update/rebind's contract requires the exception to name the full path.
  private Module resolveAndWrite(String fullPath, String path, MLXArray value) {
    Objects.requireNonNull(value, "parameter path \"" + fullPath + "\": value must not be null");
    int dot = path.indexOf('.');
    if (dot < 0) {
      if (!params.containsKey(path)) {
        throw new IllegalArgumentException("unknown parameter path \"" + fullPath + "\"");
      }
      params.put(path, value);
      return this;
    }
    String childName = path.substring(0, dot);
    Module target = children.get(childName);
    if (target == null) {
      throw new IllegalArgumentException("unknown parameter path \"" + fullPath + "\"");
    }
    return target.resolveAndWrite(fullPath, path.substring(dot + 1), value);
  }
}
