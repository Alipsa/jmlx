package se.alipsa.jmlx.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXGrad;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Module-aware autograd: wraps {@link MLXGrad.Fn} and rebinds {@code tree}'s parameters to the
 * traced primals around each {@code loss} call, so the differentiated graph actually runs over the
 * model's own ops rather than over disconnected primal arrays (req/phase4-plan.md §6). Lives here,
 * not in {@code se.alipsa.jmlx.core}, so that package never has to import {@link Module} -- see the
 * class javadoc on {@link MLXGrad}.
 *
 * <p>Differentiates with respect to every entry of {@code tree.parameters()} whose current value
 * has a floating dtype ({@link se.alipsa.jmlx.core.DType#isInexact()}) -- {@link Module} has no
 * non-trainable/buffer concept of its own, but {@code isInexact()} is enough to exclude a {@code
 * UINT32}-packed parameter automatically: in practice, {@link QuantizedLinear}'s {@code weight},
 * which has no native gradient at all (see {@link QuantizedLinear}'s own javadoc for the confirmed
 * native error). A non-floating parameter never appears in {@link #paramPaths}, never appears in
 * {@link #apply}'s traced primal vector, and never appears as a key in a {@link Result#grads()} map
 * -- confirmed empirically that excluding its index from {@code argnums} avoids the native failure
 * entirely, rather than merely working around it after the fact: a tree containing a {@code
 * QuantizedLinear} now trains its {@code scales}/{@code biases}/{@code bias} through {@link #apply}
 * exactly like any other parameter, whether or not {@code loss} actually reaches that layer's
 * {@code forward}, with only the packed {@code weight} itself excluded. An earlier version of this
 * class differentiated with respect to every entry unconditionally, which both failed outright
 * whenever {@code loss} reached a {@code QuantizedLinear}'s {@code forward} and, when it didn't,
 * returned a nonsensical {@code UINT32}-typed "gradient" for the packed weight that a training loop
 * applying {@code weight - lr * grad} across every entry of {@link Result#grads()} would have
 * silently corrupted it with -- excluding the parameter by dtype closes both failure modes at once,
 * rather than requiring every caller to filter {@link Result#grads()} themselves.
 */
public final class ModuleGrad implements AutoCloseable {

  private final Thread owner = Thread.currentThread();
  private final Module tree;
  private final BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss;
  private final List<String> paramPaths;
  private final MLXGrad.Fn fn;
  private boolean closed;

  private ModuleGrad(Module tree, BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss) {
    this.tree = tree;
    this.loss = loss;
    // Freeze only after every other fallible step, last: freeze() is irreversible and cascades to
    // every descendant, so a tree rejected anywhere in this constructor -- for having no
    // parameters, or because MLXGrad.valueAndGrad itself throws -- must be left mutable, letting
    // the caller fix it up and retry. Body.apply is not invoked during construction, so nothing
    // above this line can observe tree in a not-yet-frozen state and rely on that by accident.
    List<String> paths = inexactParamPaths(tree.parameters());
    if (paths.isEmpty()) {
      throw new IllegalStateException("ModuleGrad: tree has no parameters to differentiate");
    }
    this.paramPaths = List.copyOf(paths);
    int[] argnums = IntStream.range(0, paramPaths.size()).toArray();
    this.fn = MLXGrad.valueAndGrad(new Body(tree, loss, paths), argnums);
    tree.freeze();
  }

  /**
   * The dotted paths of {@code parameters} whose current value has a floating dtype ({@link
   * se.alipsa.jmlx.core.DType#isInexact()}), in {@code parameters}' own iteration order -- a
   * non-floating parameter (in practice, only {@link QuantizedLinear}'s {@code UINT32}-packed
   * {@code weight}) is excluded from differentiation entirely, never appearing in {@link
   * #paramPaths}, in {@link #apply}'s traced primal vector, or in a {@link Result#grads()} map:
   * confirmed empirically that excluding such a parameter's index from {@code argnums} avoids
   * {@code QuantizedMatmul}'s native {@code "[QuantizedMatmul::vjp] no gradient wrt the quantized
   * weights"} failure entirely (the failure fires only when that parameter's own gradient is
   * actually requested), rather than merely hiding a nonsensical {@code UINT32}-typed "gradient"
   * the caller would otherwise have to know to filter out of {@link Result#grads()} itself before
   * using it in a weight update -- see this class's own javadoc for why a caller doing so
   * unconditionally would otherwise silently corrupt a packed weight.
   */
  private static List<String> inexactParamPaths(SequencedMap<String, MLXArray> parameters) {
    List<String> paths = new ArrayList<>();
    for (Map.Entry<String, MLXArray> entry : parameters.entrySet()) {
      if (entry.getValue().dtype().isInexact()) {
        paths.add(entry.getKey());
      }
    }
    return paths;
  }

  /**
   * Freezes {@code tree} and captures the dotted paths of its floating-dtype parameters (see {@link
   * #inexactParamPaths}) -- the ORDER only, not the values (req/phase4-plan.md §6: re-reading
   * values every {@link #apply} is what keeps grads current after an {@code update}). {@code loss}
   * receives {@code (params, inputs)} and must return a rank-0 loss as element 0 -- see {@link
   * MLXGrad.Fn#apply} for what happens if it does not.
   *
   * <p>{@code loss}'s {@code params} array is indexed by this instance's {@link #paramPaths()}, in
   * that exact order: {@code params[i]} is the current value of {@code paramPaths().get(i)}. This
   * is <strong>not necessarily {@code tree.parameters()}'s own index order</strong> -- {@link
   * #paramPaths} only ever contains the floating-dtype subset (see {@link #inexactParamPaths}), so
   * a tree containing any non-floating parameter (in practice, a {@link QuantizedLinear}'s packed
   * {@code weight}) has an index correspondence in {@code params} that skips over it entirely:
   * {@code params[0]} is whichever floating parameter is first in {@code tree.parameters()}'s own
   * order, not necessarily {@code tree.parameters()}'s own element 0. A {@code loss} written to
   * assume positional correspondence with {@code tree.parameters()} itself, rather than with {@link
   * #paramPaths()}, silently reads the wrong parameter at every following index once the tree has
   * any excluded one -- call {@link #paramPaths()} rather than recomputing this filter.
   *
   * <p>{@code loss} must not strongly reference the returned {@code ModuleGrad} -- directly, or
   * transitively through an enclosing object the caller bound it to (e.g. a training loop's {@code
   * this::loss}, where the loop itself owns the {@code ModuleGrad} in a field). {@code Body} holds
   * {@code loss} for as long as the wrapped {@link MLXGrad.Fn}'s upcall stub is alive (see {@link
   * MLXGrad#valueAndGrad}'s own javadoc for why that's unavoidable), so such a reference closes the
   * identical cycle {@code valueAndGrad} warns against one level out -- through {@code loss}
   * instead of through {@code Body} itself -- and pins the {@code Fn}, this {@code Module} tree,
   * and its {@link MLXScope} (the model's GPU weights) for the process lifetime. Not fixable inside
   * this class: {@code loss} must be retained for the {@code Fn}'s lifetime for the upcall to be
   * able to call it at all, so avoiding the cycle is on the caller.
   */
  public static ModuleGrad of(Module tree, BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss) {
    return new ModuleGrad(tree, loss);
  }

  /**
   * The dotted paths this instance differentiates with respect to, in the exact order {@code
   * loss}'s {@code params} array corresponds to -- {@code paramPaths().get(i)} names {@code
   * params[i]} on every {@link #apply} call. Fixed at construction (see {@link #of}'s own javadoc
   * for why this can differ from {@code tree.parameters()}'s own index order) and never changes
   * afterward, regardless of any later {@code update}/{@code rebind} on {@code tree}.
   */
  public List<String> paramPaths() {
    return paramPaths;
  }

  /**
   * The closure body handed to {@link MLXGrad#valueAndGrad}. A standalone class -- never a bound
   * {@code ModuleGrad::body} method reference -- so the upcall target captures only {@code tree},
   * {@code loss} and {@code paramPaths}, never the enclosing {@link ModuleGrad}: a bound reference
   * back to {@code ModuleGrad} would close the exact reference cycle {@code MLXGrad.Fn}'s {@code
   * Holder}/{@code Upcall} split exists to prevent, this time through {@code ModuleGrad.fn} --
   * pinning the {@code Fn}, this {@code Module} tree, and its {@link MLXScope} (the model's GPU
   * weights) for the process lifetime whenever a caller drops a {@code ModuleGrad} without {@code
   * close()}, exactly the case {@code Fn}'s {@link java.lang.ref.Cleaner} backstop exists for.
   */
  private static final class Body implements Function<MLXArray[], MLXArray[]> {
    private final Module tree;
    private final BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss;
    private final List<String> paramPaths;

    Body(
        Module tree, BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss, List<String> paramPaths) {
      this.tree = tree;
      this.loss = loss;
      this.paramPaths = paramPaths;
    }

    /**
     * Splits the traced primal vector into params (indices {@code [0, paramPaths.size())},
     * differentiated) and inputs (the rest, passed through unchanged), rebinds {@code tree} onto
     * the traced params for the duration of {@code loss}, and restores the pre-call bindings in a
     * {@code finally} -- including when {@code loss} throws, so a later call outside the traced
     * region never reads into a scope that has since closed.
     */
    @Override
    public MLXArray[] apply(MLXArray[] tracedPrimals) {
      int paramCount = paramPaths.size();
      MLXArray[] tracedParams = Arrays.copyOfRange(tracedPrimals, 0, paramCount);
      MLXArray[] inputs = Arrays.copyOfRange(tracedPrimals, paramCount, tracedPrimals.length);
      SequencedMap<String, MLXArray> saved = tree.parameters();
      SequencedMap<String, MLXArray> traced = new LinkedHashMap<>();
      for (int i = 0; i < paramCount; i++) {
        traced.put(paramPaths.get(i), tracedParams[i]);
      }
      tree.rebind(traced);
      try {
        return loss.apply(tracedParams, inputs);
      } finally {
        tree.rebind(saved);
      }
    }
  }

  /**
   * Per-iteration: {@code target} is this step's scope (grads and the returned loss value land
   * there), {@code inputs} is this batch. Re-reads {@code tree.parameters()}'s current VALUES on
   * every call (req/phase4-plan.md §6: snapshotting them once in {@link #of} would differentiate
   * against stale weights after the first {@code update}).
   *
   * <p>Key-set drift is impossible: {@link #of} freezes {@code tree} before returning, {@code
   * param}/{@code child} both throw once frozen, and {@code update}/{@code rebind} only ever write
   * to an already-existing key (an unknown path throws rather than being inserted) -- so {@code
   * tree.parameters().keySet()} provably cannot differ from the paths captured in {@link #of} on
   * any call reachable through this class's own public surface. DTYPE drift is a distinct claim
   * this javadoc previously conflated with the key-set one: {@link #paramPaths} is filtered by
   * dtype ONCE, at construction (see {@link #inexactParamPaths}), and neither {@code update} nor
   * {@code rebind} enforces that a replacement keeps the value's dtype unchanged -- confirmed
   * reachable, not merely hypothetical: {@code Module.update}/{@code rebind} run no dtype check of
   * their own, so a plain {@link Linear}'s FLOAT32 {@code weight}, already included in {@link
   * #paramPaths} at construction, can be replaced afterward with a same-shape {@code INT32} (or
   * {@code UINT32}) array, and this method would otherwise have fed it straight into the traced
   * primal vector, failing deep inside {@code fn.apply} with whatever opaque native error the
   * untraceable op produces for that dtype, rather than a message naming {@code ModuleGrad} or the
   * offending path. This method now checks every {@link #paramPaths} entry's current dtype on every
   * call instead of silently trusting it (see the {@code isInexact()} check below) -- but only in
   * this one direction: the REVERSE case, a parameter excluded from {@link #paramPaths} at
   * construction for being non-floating and later replaced with a floating value, is NOT caught
   * here and remains a silent miss by design, since {@link #paramPaths} is captured once and never
   * grows -- such a parameter is simply never differentiated through this instance regardless of
   * its current dtype; construct a new {@code ModuleGrad} after such a replacement if it should
   * become trainable.
   *
   * <p>Each call builds three fresh parameter maps ({@code current} here, plus {@code Body}'s
   * {@code saved}/{@code traced}) and runs two {@code rebind} calls, each re-splitting every dotted
   * path -- immaterial for the model sizes this slice targets, but worth revisiting if a later
   * phase iterates this over substantially larger trees.
   */
  public Result apply(MLXScope target, MLXArray[] inputs) {
    ensureOpen();
    SequencedMap<String, MLXArray> current = tree.parameters();
    MLXArray[] primals = new MLXArray[paramPaths.size() + inputs.length];
    int i = 0;
    for (String path : paramPaths) {
      MLXArray value = current.get(path);
      if (!value.dtype().isInexact()) {
        throw new IllegalStateException(
            "ModuleGrad: parameter \""
                + path
                + "\" had a floating dtype when this ModuleGrad was constructed but is now "
                + value.dtype()
                + " -- a non-floating value cannot be differentiated; construct a new ModuleGrad"
                + " if this replacement is intentional");
      }
      primals[i++] = value;
    }
    for (MLXArray input : inputs) {
      primals[i++] = input;
    }
    MLXGrad.Result r = fn.apply(target, primals);
    SequencedMap<String, MLXArray> grads = new LinkedHashMap<>();
    for (int p = 0; p < paramPaths.size(); p++) {
      grads.put(paramPaths.get(p), r.grads().get(p));
    }
    return new Result(r.values().get(0), Collections.unmodifiableSequencedMap(grads));
  }

  /**
   * {@code value} is the rank-0 loss for this call; {@code grads} is keyed by dotted parameter
   * path.
   */
  public record Result(MLXArray value, SequencedMap<String, MLXArray> grads) {}

  @Override
  public void close() {
    checkThread();
    if (closed) {
      return;
    }
    closed = true;
    fn.close();
  }

  private void ensureOpen() {
    checkThread();
    if (closed) {
      throw new IllegalStateException("ModuleGrad is closed");
    }
  }

  private void checkThread() {
    Thread current = Thread.currentThread();
    if (current != owner) {
      throw new IllegalStateException(
          "ModuleGrad is confined to " + owner + " but was accessed from " + current);
    }
  }
}
