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

/** FLOAT32 delta rule with batched intra-chunk preprocessing and request-owned state. */
final class GatedDeltaRule {
  private static final int CHUNK_SIZE = 64;
  private static final int BATCH_CHUNKS = 16;
  private static final int EVALUATION_CHUNKS = 8;

  private GatedDeltaRule() {}

  /** Output [B,T,Hv,Dv] and final state [B,Hv,Dk,Dv], owned by the input scope. */
  record Result(MLXArray output, MLXArray state) {}

  private record Layout(
      int batch, int length, int keyHeads, int valueHeads, int keyDim, int valueDim) {}

  private record Prepared(
      MLXArray query,
      MLXArray output,
      MLXArray keys,
      MLXArray values,
      MLXArray tailKeys,
      MLXArray decay) {
    MLXArray[] arrays() {
      return new MLXArray[] {query, output, keys, values, tailKeys, decay};
    }

    Prepared retain(MLXScope target) {
      return new Prepared(
          MLX.hoist(query, target),
          MLX.hoist(output, target),
          MLX.hoist(keys, target),
          MLX.hoist(values, target),
          MLX.hoist(tailKeys, target),
          MLX.hoist(decay, target));
    }
  }

  /** Single-token recurrence: no evaluation, concatenation or temporary child scope. */
  static Result step(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial) {
    Layout shape = validate(q, k, v, g, beta, initial);
    if (shape.length != 1) {
      throw new IllegalArgumentException("delta step requires exactly one position");
    }
    return step(q, k, v, g, beta, initial, shape);
  }

  private static Result step(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial, Layout s) {
    k = inScope(k, q.scope());
    v = inScope(v, q.scope());
    g = inScope(g, q.scope());
    beta = inScope(beta, q.scope());
    int repeat = s.valueHeads / s.keyHeads;
    MLXArray query = repeat == 1 ? q : MLXShape.repeatAxis(q, repeat, 2);
    MLXArray key = repeat == 1 ? k : MLXShape.repeatAxis(k, repeat, 2);
    query = MLXShape.reshape(query, new int[] {s.batch, s.valueHeads, 1, s.keyDim});
    key = MLXShape.reshape(key, new int[] {s.batch, s.valueHeads, 1, s.keyDim});
    MLXArray value = MLXShape.reshape(v, new int[] {s.batch, s.valueHeads, 1, s.valueDim});
    MLXArray decay = MLXShape.reshape(MLXOps.exp(g), new int[] {s.batch, s.valueHeads, 1, 1});
    MLXArray rate = MLXShape.reshape(beta, new int[] {s.batch, s.valueHeads, 1, 1});
    MLXArray decayed = MLXOps.multiply(initial, decay);
    MLXArray delta = MLXOps.multiply(MLXOps.subtract(value, MLXOps.matmul(key, decayed)), rate);
    MLXArray next = MLXOps.add(decayed, MLXOps.matmul(swap4(key), delta));
    MLXArray output =
        MLXShape.reshape(
            MLXOps.matmul(query, next), new int[] {s.batch, 1, s.valueHeads, s.valueDim});
    return new Result(output, next);
  }

  /** Token-by-token reference; single-token callers use the lazy decode path. */
  static Result recurrent(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial) {
    Layout s = validate(q, k, v, g, beta, initial);
    if (s.length == 1) {
      return step(q, k, v, g, beta, initial, s);
    }
    MLXScope target = q.scope();
    MLXArray state = initial;
    boolean owned = false;
    List<MLXArray> parts = new ArrayList<>();
    try {
      for (int t = 0; t < s.length; t++) {
        try (MLXScope token = target.newChild()) {
          Layout one = new Layout(s.batch, 1, s.keyHeads, s.valueHeads, s.keyDim, s.valueDim);
          Result result =
              step(
                  window(inScope(q, token), 1, t, t + 1),
                  window(inScope(k, token), 1, t, t + 1),
                  window(inScope(v, token), 1, t, t + 1),
                  window(inScope(g, token), 1, t, t + 1),
                  window(inScope(beta, token), 1, t, t + 1),
                  inScope(state, token),
                  one);
          MLX.eval(result.output, result.state);
          MLXArray next = MLX.hoist(result.state, target);
          parts.add(MLX.hoist(result.output, target));
          if (owned) {
            state.close();
          }
          state = next;
          owned = true;
        }
      }
      return new Result(MLXShape.concatenate(parts.toArray(MLXArray[]::new), 1), state);
    } catch (RuntimeException | Error error) {
      if (owned) {
        state.close();
      }
      throw error;
    } finally {
      parts.forEach(MLXArray::close);
    }
  }

  /** Batched 64-position chunks, bounded preprocessing and eight-chunk state evaluation windows. */
  static Result chunked(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial) {
    return chunked(q, k, v, g, beta, initial, () -> {});
  }

  static Result chunked(
      MLXArray q,
      MLXArray k,
      MLXArray v,
      MLXArray g,
      MLXArray beta,
      MLXArray initial,
      Runnable checkpoint) {
    Layout s = validate(q, k, v, g, beta, initial);
    if (s.length == 1) {
      return step(q, k, v, g, beta, initial, s);
    }
    MLXScope target = q.scope();
    MLXArray state = initial;
    boolean owned = false;
    List<MLXArray> parts = new ArrayList<>();
    try {
      for (int start = 0; start < s.length; start += BATCH_CHUNKS * CHUNK_SIZE) {
        checkpoint.run();
        int length = Math.min(s.length - start, BATCH_CHUNKS * CHUNK_SIZE);
        int count = Math.min(CHUNK_SIZE, length);
        int chunks = (length + count - 1) / count;
        try (MLXScope group = target.newChild()) {
          Prepared prepared;
          try (MLXScope preprocessing = group.newChild()) {
            prepared = prepare(q, k, v, g, beta, s, start, length, count, chunks, preprocessing);
            MLX.eval(prepared.arrays());
            prepared = prepared.retain(group);
          }
          for (int first = 0; first < chunks; first += EVALUATION_CHUNKS) {
            int last = Math.min(chunks, first + EVALUATION_CHUNKS);
            try (MLXScope scan = group.newChild()) {
              MLXArray carried = inScope(state, scan);
              List<MLXArray> outputs = new ArrayList<>();
              for (int i = first; i < last; i++) {
                checkpoint.run();
                MLXArray query = part(inScope(prepared.query, scan), i);
                MLXArray baseOutput = part(inScope(prepared.output, scan), i);
                MLXArray keys = part(inScope(prepared.keys, scan), i);
                MLXArray values = part(inScope(prepared.values, scan), i);
                MLXArray tailKeys = part(inScope(prepared.tailKeys, scan), i);
                MLXArray decay = part(inScope(prepared.decay, scan), i);
                outputs.add(MLXOps.add(baseOutput, MLXOps.matmul(query, carried)));
                MLXArray updates = MLXOps.subtract(values, MLXOps.matmul(keys, carried));
                carried =
                    MLXOps.add(MLXOps.multiply(carried, decay), MLXOps.matmul(tailKeys, updates));
              }
              MLXArray joined =
                  outputs.size() == 1
                      ? outputs.getFirst()
                      : MLXShape.concatenate(outputs.toArray(MLXArray[]::new), 2);
              int real = Math.min(length - first * count, (last - first) * count);
              if (real != (last - first) * count) {
                joined = window(joined, 2, 0, real);
              }
              MLXArray output = MLXShape.transpose(joined, new int[] {0, 2, 1, 3});
              MLX.eval(output, carried);
              MLXArray next = MLX.hoist(carried, target);
              parts.add(MLX.hoist(output, target));
              if (owned) {
                state.close();
              }
              state = next;
              owned = true;
            }
          }
        }
      }
      MLXArray output =
          parts.size() == 1
              ? parts.removeFirst()
              : MLXShape.concatenate(parts.toArray(MLXArray[]::new), 1);
      return new Result(output, state);
    } catch (RuntimeException | Error error) {
      if (owned) {
        state.close();
      }
      throw error;
    } finally {
      parts.forEach(MLXArray::close);
    }
  }

  private static Prepared prepare(
      MLXArray q,
      MLXArray k,
      MLXArray v,
      MLXArray g,
      MLXArray beta,
      Layout s,
      int start,
      int length,
      int count,
      int chunks,
      MLXScope scope) {
    int padded = chunks * count;
    int repeat = s.valueHeads / s.keyHeads;
    final MLXArray query =
        chunks(q, s, start, length, padded, count, chunks, s.keyDim, repeat, scope);
    MLXArray key = chunks(k, s, start, length, padded, count, chunks, s.keyDim, repeat, scope);
    final MLXArray value = chunks(v, s, start, length, padded, count, chunks, s.valueDim, 1, scope);
    MLXArray logDecay = scalars(g, s, start, length, padded, count, chunks, scope);
    MLXArray rate = scalars(beta, s, start, length, padded, count, chunks, scope);
    MLXArray cumulative = MLXOps.cumulativeSumAxis(logDecay, 3, false, true);
    MLXArray diff =
        MLXOps.subtract(MLXShape.expandDims(cumulative, -1), MLXShape.expandDims(cumulative, -2));
    MLXArray lower =
        MLX.astype(
            MLXShape.tril(MLX.ones(scope, new int[] {count, count}, DType.FLOAT32), 0), DType.BOOL);
    MLXArray forbidden = MLX.full(scope, new int[] {1}, Float.NEGATIVE_INFINITY, DType.FLOAT32);
    MLXArray pairwise = MLXOps.exp(MLXOps.where(lower, diff, forbidden));
    diff.close();
    MLXArray decay = MLXShape.expandDims(MLXOps.exp(cumulative), -1);
    MLXArray weightedKey = MLXOps.multiply(key, MLXShape.expandDims(rate, -1));
    MLXArray keyProducts = MLXOps.matmul(weightedKey, swap5(key));
    MLXArray system = MLXOps.multiply(keyProducts, pairwise);
    keyProducts.close();
    int[] matrixShape = {s.batch, s.valueHeads, chunks, count, count};
    final MLXArray inverse = invertUnitLower(system, matrixShape, scope);
    system.close();
    MLXArray weightedValue = MLXOps.multiply(value, MLXShape.expandDims(rate, -1));
    MLXArray decayedKey = MLXOps.multiply(weightedKey, decay);
    final MLXArray rhs = MLXShape.concatenate(new MLXArray[] {weightedValue, decayedKey}, -1);
    weightedValue.close();
    decayedKey.close();
    weightedKey.close();
    final MLXArray solved = MLXOps.matmul(inverse, rhs);
    inverse.close();
    rhs.close();
    final MLXArray values = window(solved, 4, 0, s.valueDim);
    final MLXArray keys = window(solved, 4, s.valueDim, s.valueDim + s.keyDim);
    solved.close();
    MLXArray queryProducts = MLXOps.matmul(query, swap5(key));
    MLXArray intra = MLXOps.multiply(queryProducts, pairwise);
    queryProducts.close();
    pairwise.close();
    MLXArray decayedQuery = MLXOps.multiply(query, decay);
    MLXArray withinKeys = MLXOps.matmul(intra, keys);
    final MLXArray effectiveQuery = MLXOps.subtract(decayedQuery, withinKeys);
    decayedQuery.close();
    withinKeys.close();
    MLXArray baseOutput = MLXOps.matmul(intra, values);
    intra.close();
    MLXArray last = window(cumulative, 3, count - 1, count);
    MLXArray tail = MLXOps.exp(MLXOps.subtract(last, cumulative));
    MLXArray tailKeys = swap5(MLXOps.multiply(key, MLXShape.expandDims(tail, -1)));
    return new Prepared(
        effectiveQuery,
        baseOutput,
        keys,
        values,
        tailKeys,
        MLXShape.expandDims(MLXOps.exp(last), -1));
  }

  private static MLXArray invertUnitLower(MLXArray system, int[] shape, MLXScope scope) {
    int count = shape[3];
    float[] identity = new float[count * count];
    for (int i = 0; i < count; i++) {
      identity[i * count + i] = 1;
    }
    MLXArray eye =
        MLXShape.broadcastTo(MLX.array(scope, identity, new int[] {count, count}), shape);
    List<MLXArray> rows = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      MLXArray row = window(eye, shape, 3, i, i + 1);
      if (i > 0) {
        MLXArray systemRow = window(system, shape, 3, i, i + 1);
        MLXArray coefficients = window(systemRow, 4, 0, i);
        systemRow.close();
        MLXArray previous = MLXShape.concatenate(rows.toArray(MLXArray[]::new), 3);
        MLXArray product = MLXOps.matmul(coefficients, previous);
        final MLXArray solved = MLXOps.subtract(row, product);
        // Graph inputs are owned by value. Retiring wrappers here lets evaluation recycle
        // previous concatenation buffers instead of pinning every row's intermediate matrix.
        row.close();
        coefficients.close();
        previous.close();
        product.close();
        row = solved;
      }
      rows.add(row);
    }
    final MLXArray inverse = MLXShape.concatenate(rows.toArray(MLXArray[]::new), 3);
    rows.forEach(MLXArray::close);
    return inverse;
  }

  private static MLXArray chunks(
      MLXArray input,
      Layout s,
      int start,
      int length,
      int padded,
      int count,
      int chunks,
      int width,
      int repeat,
      MLXScope scope) {
    MLXArray a = pad(window(inScope(input, scope), 1, start, start + length), padded - length);
    if (repeat != 1) {
      a = MLXShape.repeatAxis(a, repeat, 2);
    }
    return MLXShape.transpose(
        MLXShape.reshape(a, new int[] {s.batch, chunks, count, s.valueHeads, width}),
        new int[] {0, 3, 1, 2, 4});
  }

  private static MLXArray scalars(
      MLXArray input,
      Layout s,
      int start,
      int length,
      int padded,
      int count,
      int chunks,
      MLXScope scope) {
    MLXArray a = pad(window(inScope(input, scope), 1, start, start + length), padded - length);
    return MLXShape.transpose(
        MLXShape.reshape(a, new int[] {s.batch, chunks, count, s.valueHeads}),
        new int[] {0, 3, 1, 2});
  }

  private static MLXArray pad(MLXArray a, int extra) {
    return extra == 0
        ? a
        : MLXShape.pad(
            a,
            new int[] {1},
            new int[] {0},
            new int[] {extra},
            MLX.full(a.scope(), new int[] {1}, 0f, DType.FLOAT32),
            MLXShape.PadMode.CONSTANT);
  }

  private static MLXArray part(MLXArray a, int index) {
    int[] shape = a.shape();
    return MLXShape.reshape(
        window(a, shape, 2, index, index + 1), new int[] {shape[0], shape[1], shape[3], shape[4]});
  }

  private static MLXArray swap4(MLXArray a) {
    return MLXShape.transpose(a, new int[] {0, 1, 3, 2});
  }

  private static MLXArray swap5(MLXArray a) {
    return MLXShape.transpose(a, new int[] {0, 1, 2, 4, 3});
  }

  private static MLXArray inScope(MLXArray a, MLXScope target) {
    return a.scope() == target ? a : MLXShape.reshape(MLXShape.expandDims(a, target, 0), a.shape());
  }

  private static MLXArray window(MLXArray a, int axis, int start, int stop) {
    return window(a, a.shape(), axis, start, stop);
  }

  private static MLXArray window(MLXArray a, int[] shape, int axis, int start, int stop) {
    int[] starts = new int[shape.length];
    int[] stops = shape.clone();
    starts[axis] = start;
    stops[axis] = stop;
    return MLXShape.slice(a, starts, stops);
  }

  private static Layout validate(
      MLXArray q, MLXArray k, MLXArray v, MLXArray g, MLXArray beta, MLXArray initial) {
    int[] qs = q.shape();
    int[] vs = v.shape();
    int[] gs = g.shape();
    if (qs.length != 4
        || !Arrays.equals(qs, k.shape())
        || vs.length != 4
        || gs.length != 3
        || !Arrays.equals(gs, beta.shape())
        || initial.shape().length != 4) {
      throw new IllegalArgumentException("delta rule: invalid tensor ranks or paired shapes");
    }
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
    return new Layout(qs[0], qs[1], qs[2], vs[2], qs[3], vs[3]);
  }
}
