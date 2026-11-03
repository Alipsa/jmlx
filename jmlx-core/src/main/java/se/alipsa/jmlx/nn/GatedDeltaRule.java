package se.alipsa.jmlx.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** FLOAT32 delta-rule prototypes; callers explicitly select recurrence or chunked execution. */
final class GatedDeltaRule {
  private GatedDeltaRule() {}

  /** Evaluated output [B,T,Hv,Dv] and final state [B,Hv,Dk,Dv], owned by the input scope. */
  record Result(MLXArray output, MLXArray state) {}

  /** Uses one independently evaluated token update per child scope. Queries are already scaled. */
  static Result recurrent(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial) {
    return run(q, k, v, g, beta, initial, 1);
  }

  /** Uses at most 64 real positions per independently evaluated child scope. */
  static Result chunked(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial) {
    return run(q, k, v, g, beta, initial, 64);
  }

  private static Result run(
      MLXArray q,
      MLXArray k,
      MLXArray v,
      MLXArray g,
      MLXArray beta,
      MLXArray initial,
      int chunkSize) {
    validate(q, k, v, g, beta, initial);
    MLXScope target = q.scope();
    int length = q.shape()[1];
    int repeat = v.shape()[2] / q.shape()[2];
    List<MLXArray> parts = new ArrayList<>();
    MLXArray state = initial;
    boolean ownsState = false;
    try {
      for (int start = 0; start < length; start += chunkSize) {
        int end = Math.min(length, start + chunkSize);
        try (MLXScope chunk = target.newChild()) {
          MLXArray qc = heads(window(inScope(q, chunk), 1, start, end), repeat);
          MLXArray kc = heads(window(inScope(k, chunk), 1, start, end), repeat);
          MLXArray vc = heads(window(inScope(v, chunk), 1, start, end), 1);
          MLXArray gc =
              MLXShape.transpose(window(inScope(g, chunk), 1, start, end), new int[] {0, 2, 1});
          MLXArray bc =
              MLXShape.transpose(window(inScope(beta, chunk), 1, start, end), new int[] {0, 2, 1});
          MLXArray sc = inScope(state, chunk);
          Result result =
              chunkSize == 1 ? token(qc, kc, vc, gc, bc, sc) : chunk(qc, kc, vc, gc, bc, sc);
          MLX.eval(result.output(), result.state());
          MLXArray next = MLX.hoist(result.state(), target);
          parts.add(MLX.hoist(result.output(), target));
          if (ownsState) {
            state.close();
          }
          state = next;
          ownsState = true;
        }
      }
      MLXArray joined = MLXShape.concatenate(parts.toArray(MLXArray[]::new), 2);
      MLXArray output = MLXShape.transpose(joined, new int[] {0, 2, 1, 3});
      MLX.eval(output, state);
      joined.close();
      return new Result(output, state);
    } catch (RuntimeException | Error error) {
      if (ownsState) {
        state.close();
      }
      throw error;
    } finally {
      parts.forEach(MLXArray::close);
    }
  }

  private static Result token(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray state) {
    MLXArray decayed = MLXOps.multiply(state, MLXShape.expandDims(MLXOps.exp(g), -1));
    MLXArray predicted = MLXOps.matmul(k, decayed);
    MLXArray delta = MLXOps.multiply(MLXOps.subtract(v, predicted), MLXShape.expandDims(beta, -1));
    MLXArray next = MLXOps.add(decayed, MLXOps.matmul(swapLast(k), delta));
    return new Result(MLXOps.matmul(q, next), next);
  }

  private static Result chunk(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray state) {
    int count = q.shape()[2];
    int valueWidth = v.shape()[3];
    MLXArray cumulative = MLXOps.cumulativeSumAxis(g, 2, false, true);
    MLXArray diff =
        MLXOps.subtract(MLXShape.expandDims(cumulative, -1), MLXShape.expandDims(cumulative, -2));
    MLXScope scope = q.scope();
    MLXArray lower =
        MLX.astype(
            MLXShape.tril(MLX.ones(scope, new int[] {count, count}, DType.FLOAT32), 0), DType.BOOL);
    MLXArray forbidden = MLX.full(scope, new int[] {1}, Float.NEGATIVE_INFINITY, DType.FLOAT32);
    // Mask before exp: forbidden upper-triangle decay differences can be arbitrarily positive.
    MLXArray pairwise = MLXOps.exp(MLXOps.where(lower, diff, forbidden));
    MLXArray decay = MLXShape.expandDims(MLXOps.exp(cumulative), -1);
    MLXArray weightedKeys = MLXOps.multiply(k, MLXShape.expandDims(beta, -1));
    MLXArray weightedValues = MLXOps.multiply(v, MLXShape.expandDims(beta, -1));
    MLXArray system = MLXOps.multiply(MLXOps.matmul(weightedKeys, swapLast(k)), pairwise);
    MLXArray rhs =
        MLXShape.concatenate(
            new MLXArray[] {weightedValues, MLXOps.multiply(weightedKeys, decay)}, -1);
    MLXArray solved = solveUnitLower(system, rhs);
    MLXArray values = window(solved, 3, 0, valueWidth);
    MLXArray decayedKeys = window(solved, 3, valueWidth, solved.shape()[3]);
    MLXArray updates = MLXOps.subtract(values, MLXOps.matmul(decayedKeys, state));
    MLXArray intra = MLXOps.multiply(MLXOps.matmul(q, swapLast(k)), pairwise);
    MLXArray output =
        MLXOps.add(MLXOps.matmul(MLXOps.multiply(q, decay), state), MLXOps.matmul(intra, updates));
    MLXArray last = window(cumulative, 2, count - 1, count);
    MLXArray tailDecay = MLXShape.expandDims(MLXOps.exp(MLXOps.subtract(last, cumulative)), -1);
    MLXArray next =
        MLXOps.add(
            MLXOps.multiply(state, MLXShape.expandDims(MLXOps.exp(last), -1)),
            MLXOps.matmul(swapLast(MLXOps.multiply(k, tailDecay)), updates));
    return new Result(output, next);
  }

  private static MLXArray solveUnitLower(MLXArray system, MLXArray rhs) {
    List<MLXArray> rows = new ArrayList<>();
    for (int i = 0; i < rhs.shape()[2]; i++) {
      MLXArray row = window(rhs, 2, i, i + 1);
      if (i > 0) {
        MLXArray coefficients = window(window(system, 2, i, i + 1), 3, 0, i);
        MLXArray previous = MLXShape.concatenate(rows.toArray(MLXArray[]::new), 2);
        row = MLXOps.subtract(row, MLXOps.matmul(coefficients, previous));
      }
      rows.add(row);
    }
    return MLXShape.concatenate(rows.toArray(MLXArray[]::new), 2);
  }

  private static MLXArray heads(MLXArray a, int repeat) {
    MLXArray repeated = repeat == 1 ? a : MLXShape.repeatAxis(a, repeat, 2);
    return MLXShape.transpose(repeated, new int[] {0, 2, 1, 3});
  }

  private static MLXArray swapLast(MLXArray a) {
    return MLXShape.transpose(a, new int[] {0, 1, 3, 2});
  }

  private static MLXArray inScope(MLXArray a, MLXScope target) {
    return MLXShape.reshape(MLXShape.expandDims(a, target, 0), a.shape());
  }

  private static MLXArray window(MLXArray a, int axis, int start, int stop) {
    int[] starts = new int[a.ndim()];
    int[] stops = a.shape();
    starts[axis] = start;
    stops[axis] = stop;
    return MLXShape.slice(a, starts, stops);
  }

  private static void validate(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial) {
    if (q.ndim() != 4
        || !Arrays.equals(q.shape(), k.shape())
        || v.ndim() != 4
        || g.ndim() != 3
        || !Arrays.equals(g.shape(), beta.shape())
        || initial.ndim() != 4) {
      throw new IllegalArgumentException("delta rule: invalid tensor ranks or paired shapes");
    }
    int[] qs = q.shape();
    int[] vs = v.shape();
    int[] gs = g.shape();
    for (int dimension : qs) {
      if (dimension < 1) {
        throw new IllegalArgumentException("delta rule: dimensions must be positive");
      }
    }
    if (vs[0] != qs[0]
        || vs[1] != qs[1]
        || vs[2] < 1
        || vs[3] < 1
        || vs[2] % qs[2] != 0
        || !Arrays.equals(gs, new int[] {qs[0], qs[1], vs[2]})
        || !Arrays.equals(initial.shape(), new int[] {qs[0], vs[2], qs[3], vs[3]})) {
      throw new IllegalArgumentException(
          "delta rule: incompatible batch, sequence, head or state dimensions");
    }
    for (MLXArray a : new MLXArray[] {q, k, v, g, beta, initial}) {
      if (a.dtype() != DType.FLOAT32 || !a.scope().isAncestorOf(q.scope())) {
        throw new IllegalArgumentException(
            "delta rule: FLOAT32 inputs must share the query scope or an ancestor");
      }
    }
  }
}
