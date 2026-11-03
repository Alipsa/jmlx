package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Pre-normalized residual block with either recurrent DeltaNet or gated full attention. */
public final class HybridDecoderBlock extends Module {
  private final RMSNorm inputNorm;
  private final RMSNorm postAttentionNorm;
  private final UnaryLayer mlp;
  private final GatedDeltaNet linearAttention;
  private final GatedDecoderAttention fullAttention;

  /** Creates a linear-attention block with its own request-owned recurrent cache. */
  public HybridDecoderBlock(
      MLXScope scope,
      RMSNorm inputNorm,
      GatedDeltaNet attention,
      RMSNorm postAttentionNorm,
      UnaryLayer mlp) {
    this(scope, inputNorm, Objects.requireNonNull(attention), null, postAttentionNorm, mlp);
  }

  /** Creates a full-attention block with its own full K/V cache. */
  public HybridDecoderBlock(
      MLXScope scope,
      RMSNorm inputNorm,
      GatedDecoderAttention attention,
      RMSNorm postAttentionNorm,
      UnaryLayer mlp) {
    this(scope, inputNorm, null, Objects.requireNonNull(attention), postAttentionNorm, mlp);
  }

  private HybridDecoderBlock(
      MLXScope scope,
      RMSNorm inputNorm,
      GatedDeltaNet linear,
      GatedDecoderAttention full,
      RMSNorm postAttentionNorm,
      UnaryLayer mlp) {
    super(scope);
    this.inputNorm = child("inputNorm", Objects.requireNonNull(inputNorm));
    this.postAttentionNorm = child("postAttentionNorm", Objects.requireNonNull(postAttentionNorm));
    this.mlp = child("mlp", Objects.requireNonNull(mlp));
    linearAttention = linear == null ? null : child("attention", linear);
    fullAttention = full == null ? null : child("attention", full);
  }

  /** Whether this layer owns recurrent rather than full-attention state. */
  public boolean isLinear() {
    return linearAttention != null;
  }

  /** Builds the linear block; evaluation/commit occurs at the request's execution boundary. */
  public MLXArray forward(MLXArray x, RecurrentCache cache, GatedDeltaNet.PrefillMethod method) {
    return forward(x, cache, method, () -> {});
  }

  /** Builds a linear block with cooperative prefill checks delegated to its DeltaNet scan. */
  public MLXArray forward(
      MLXArray x, RecurrentCache cache, GatedDeltaNet.PrefillMethod method, Runnable checkpoint) {
    if (linearAttention == null) {
      throw new IllegalStateException("full-attention block requires a K/V cache");
    }
    try {
      return finish(x, linearAttention.forward(inputNorm.forward(x), cache, method, checkpoint));
    } catch (RuntimeException | Error error) {
      // A DeltaNet failure poisons on its own; a post-attention failure must too, or the cache
      // would retain this step's provisional state at an advanced position and look reusable.
      cache.poison();
      throw error;
    }
  }

  /** Builds the full-attention block with the step's shared multi-axis rotation. */
  public MLXArray forward(MLXArray x, KVCache cache, MLXArray mask, MultiAxisRotation rotation) {
    if (fullAttention == null) {
      throw new IllegalStateException("linear-attention block requires a recurrent cache");
    }
    try {
      return finish(x, fullAttention.forward(inputNorm.forward(x), cache, mask, rotation));
    } catch (RuntimeException | Error error) {
      // Without this, a post-attention failure leaves the K/V cache holding this step's
      // lazy append, apparently usable for the next step with wrong history.
      cache.poison();
      throw error;
    }
  }

  private MLXArray finish(MLXArray input, MLXArray attended) {
    MLXArray residual = MLXOps.add(input, attended);
    return MLXOps.add(residual, mlp.forward(postAttentionNorm.forward(residual)));
  }
}
