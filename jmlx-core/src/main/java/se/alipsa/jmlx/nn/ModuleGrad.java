package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.SequencedMap;
import java.util.function.BiFunction;
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
 */
public final class ModuleGrad implements AutoCloseable {

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
    // the caller fix it up and retry. this::body is not invoked during construction, so nothing
    // above this line can observe tree in a not-yet-frozen state and rely on that by accident.
    List<String> paths = List.copyOf(tree.parameters().keySet());
    if (paths.isEmpty()) {
      throw new IllegalStateException("ModuleGrad: tree has no parameters to differentiate");
    }
    this.paramPaths = paths;
    int[] argnums = IntStream.range(0, paramPaths.size()).toArray();
    this.fn = MLXGrad.valueAndGrad(this::body, argnums);
    tree.freeze();
  }

  /**
   * Freezes {@code tree} and captures {@code tree.parameters().keySet()} -- the ORDER only, not the
   * values (req/phase4-plan.md §6: re-reading values every {@link #apply} is what keeps grads
   * current after an {@code update}). {@code loss} receives {@code (params, inputs)} and must
   * return a rank-0 loss as element 0 -- see {@link MLXGrad.Fn#apply} for what happens if it does
   * not.
   */
  public static ModuleGrad of(Module tree, BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss) {
    return new ModuleGrad(tree, loss);
  }

  /**
   * The closure body handed to {@link MLXGrad#valueAndGrad}: splits the traced primal vector into
   * params (indices {@code [0, paramPaths.size())}, differentiated) and inputs (the rest, passed
   * through unchanged), rebinds {@code tree} onto the traced params for the duration of {@code
   * loss}, and restores the pre-call bindings in a {@code finally} -- including when {@code loss}
   * throws, so a later call outside the traced region never reads into a scope that has since
   * closed.
   */
  private MLXArray[] body(MLXArray[] tracedPrimals) {
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

  /**
   * Per-iteration: {@code target} is this step's scope (grads and the returned loss value land
   * there), {@code inputs} is this batch. Re-reads {@code tree.parameters()}'s current VALUES on
   * every call (req/phase4-plan.md §6: snapshotting them once in {@link #of} would differentiate
   * against stale weights after the first {@code update}).
   *
   * <p>No key-set drift check: {@link #of} freezes {@code tree} before returning, {@code
   * param}/{@code child} both throw once frozen, and {@code update}/{@code rebind} only ever write
   * to an already-existing key (an unknown path throws rather than being inserted) -- so {@code
   * tree.parameters().keySet()} provably cannot differ from the paths captured in {@link #of} on
   * any call reachable through this class's own public surface.
   */
  public Result apply(MLXScope target, MLXArray[] inputs) {
    ensureOpen();
    SequencedMap<String, MLXArray> current = tree.parameters();
    MLXArray[] primals = new MLXArray[paramPaths.size() + inputs.length];
    int i = 0;
    for (String path : paramPaths) {
      primals[i++] = current.get(path);
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
    if (closed) {
      return;
    }
    closed = true;
    fn.close();
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("ModuleGrad is closed");
    }
  }
}
