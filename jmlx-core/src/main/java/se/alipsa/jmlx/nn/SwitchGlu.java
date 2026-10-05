package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Stacked gated experts, {@code down(act(gate(x)) * up(x))} per expert, evaluated only for the
 * experts each token was routed to, using {@link MLXOps#gatherMatmul}. Weights are the Hugging Face
 * per-expert matrices stacked on a new leading axis: gate/up {@code [E, F, H]}, down {@code [E, H,
 * F]}. See req/plans/phase6-3-performance.md.
 */
public final class SwitchGlu extends Module {

  /**
   * Index count ({@code B * T * K}) at or above which slots are sorted by expert before the
   * gathered matmuls, so each expert's weights are read contiguously (mlx-lm's {@code SwitchGLU}
   * uses the same value). Below it the sort costs more than it saves.
   */
  static final int SORT_THRESHOLD = 64;

  private final UnaryModule activationLayer;
  private final int experts;
  private final int hidden;

  /**
   * Registers the stacked projections as {@code gateWeight}, {@code upWeight}, {@code downWeight}.
   */
  public SwitchGlu(
      MLXScope scope,
      MLXArray gateWeight,
      MLXArray upWeight,
      MLXArray downWeight,
      Activation activation) {
    super(scope);
    int[] gate = rank3(gateWeight, "gateWeight");
    int[] up = rank3(upWeight, "upWeight");
    int[] down = rank3(downWeight, "downWeight");
    if (upWeight.dtype() != gateWeight.dtype() || downWeight.dtype() != gateWeight.dtype()) {
      throw new IllegalArgumentException(
          "gate/up/down weights must share one dtype, got "
              + gateWeight.dtype()
              + "/"
              + upWeight.dtype()
              + "/"
              + downWeight.dtype());
    }
    if (!Arrays.equals(gate, up)) {
      throw new IllegalArgumentException(
          "upWeight must have gateWeight's shape "
              + Arrays.toString(gate)
              + ", got "
              + Arrays.toString(up));
    }
    if (down[0] != gate[0] || down[1] != gate[2] || down[2] != gate[1]) {
      throw new IllegalArgumentException(
          "downWeight must have shape [E, H, F] = ["
              + gate[0]
              + ", "
              + gate[2]
              + ", "
              + gate[1]
              + "], got "
              + Arrays.toString(down));
    }
    param("gateWeight", gateWeight);
    param("upWeight", upWeight);
    param("downWeight", downWeight);
    experts = gate[0];
    hidden = gate[2];
    activationLayer = Objects.requireNonNull(activation, "activation").layer(scope);
  }

  /** Number of stacked experts, {@code E}. */
  public int experts() {
    return experts;
  }

  /**
   * Returns the {@code [B, T, K, H]} outputs of the expert chosen for each of a token's {@code K}
   * slots, before routing weights are applied. {@code indices} must be INT32 {@code [B, T, K]} with
   * every value in {@code [0, experts())}: the native gather does not bounds-check, so an
   * out-of-range index returns unspecified values instead of failing. {@code indices} must live in
   * {@code x}'s scope or a descendant of it; otherwise an {@link IllegalArgumentException} is
   * thrown, since every array derived from them is allocated into their scope and indices kept in
   * an ancestor (for example the model scope) would leak those arrays until it closes.
   */
  public MLXArray forward(MLXArray x, MLXArray indices) {
    Objects.requireNonNull(indices, "indices");
    return forward(x, indices, indices.size() >= SORT_THRESHOLD);
  }

  MLXArray forward(MLXArray x, MLXArray indices, boolean sort) {
    Objects.requireNonNull(x, "x");
    Objects.requireNonNull(indices, "indices");
    int[] xs = x.shape();
    int[] is = indices.shape();
    if (xs.length != 3 || xs[2] != hidden) {
      throw new IllegalArgumentException(
          "x must have shape [batch, tokens, " + hidden + "], got " + Arrays.toString(xs));
    }
    if (is.length != 3 || is[0] != xs[0] || is[1] != xs[1] || indices.dtype() != DType.INT32) {
      throw new IllegalArgumentException(
          "indices must be INT32 [batch, tokens, topK] matching x, got "
              + indices.dtype()
              + " "
              + Arrays.toString(is));
    }
    MLXScope s = x.scope();
    if (!s.isAncestorOf(indices.scope())) {
      throw new IllegalArgumentException(
          "indices must live in x's scope or a descendant of it, not an ancestor or unrelated"
              + " scope");
    }
    int b = xs[0];
    int t = xs[1];
    int k = is[2];
    // gather_mm has no VJP with respect to its indices; routing is not differentiable anyway.
    MLXArray idx = MLXOps.stopGradient(indices);
    // Weight views go into the caller's scope, never the model scope (req/phase4-plan.md §2).
    MLXArray gateT = MLXShape.swapaxes(param("gateWeight"), s, -1, -2);
    MLXArray upT = MLXShape.swapaxes(param("upWeight"), s, -1, -2);
    MLXArray downT = MLXShape.swapaxes(param("downWeight"), s, -1, -2);
    if (!sort) {
      MLXArray rows = MLXShape.reshape(x, new int[] {b, t, 1, 1, hidden});
      MLXArray h = glu(rows, gateT, upT, idx, false);
      MLXArray y = MLXOps.gatherMatmul(h, downT, null, idx, false);
      return MLXShape.reshape(y, new int[] {b, t, k, hidden});
    }
    int n = b * t;
    int m = n * k;
    MLXArray flat = MLXShape.reshape(idx, new int[] {m});
    MLXArray order = MLXOps.argsortAxis(flat, 0);
    MLXArray inverse = MLXOps.argsortAxis(order, 0);
    MLXArray tokenOfSlot =
        MLXShape.reshape(
            MLXShape.broadcastTo(
                MLXShape.reshape(MLX.arange(s, 0, n, 1, DType.INT32), new int[] {n, 1}),
                new int[] {n, k}),
            new int[] {m});
    MLXArray rows =
        MLXShape.takeAxis(
            MLXShape.reshape(x, new int[] {n, 1, hidden}),
            MLXShape.takeAxis(tokenOfSlot, order, 0),
            0);
    MLXArray sortedIdx = MLXShape.takeAxis(flat, order, 0);
    MLXArray h = glu(rows, gateT, upT, sortedIdx, true);
    MLXArray y = MLXOps.gatherMatmul(h, downT, null, sortedIdx, true);
    return MLXShape.reshape(MLXShape.takeAxis(y, inverse, 0), new int[] {b, t, k, hidden});
  }

  private MLXArray glu(
      MLXArray rows, MLXArray gateT, MLXArray upT, MLXArray idx, boolean sortedIndices) {
    MLXArray gate = MLXOps.gatherMatmul(rows, gateT, null, idx, sortedIndices);
    MLXArray up = MLXOps.gatherMatmul(rows, upT, null, idx, sortedIndices);
    return MLXOps.multiply(activationLayer.forward(gate), up);
  }

  private static int[] rank3(MLXArray weight, String name) {
    Objects.requireNonNull(weight, name);
    if (weight.ndim() != 3) {
      throw new IllegalArgumentException(
          name + " must be rank 3, got shape " + Arrays.toString(weight.shape()));
    }
    return weight.shape();
  }
}
