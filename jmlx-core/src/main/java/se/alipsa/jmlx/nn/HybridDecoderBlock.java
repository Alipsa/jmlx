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
    if (linearAttention == null) {
      throw new IllegalStateException("full-attention block requires a K/V cache");
    }
    return finish(x, linearAttention.forward(inputNorm.forward(x), cache, method));
  }

  /** Builds the full-attention block with the step's shared multi-axis rotation. */
  public MLXArray forward(MLXArray x, KVCache cache, MLXArray mask, MultiAxisRotation rotation) {
    if (fullAttention == null) {
      throw new IllegalStateException("linear-attention block requires a recurrent cache");
    }
    return finish(x, fullAttention.forward(inputNorm.forward(x), cache, mask, rotation));
  }

  private MLXArray finish(MLXArray input, MLXArray attended) {
    MLXArray residual = MLXOps.add(input, attended);
    return MLXOps.add(residual, mlp.forward(postAttentionNorm.forward(residual)));
  }
}
