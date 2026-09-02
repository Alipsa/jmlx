package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * A pre-normalized causal decoder block: attention and SwiGLU MLP, each followed by a residual
 * addition. Its parameter names are deliberately architecture-neutral; model loaders translate the
 * Hugging Face checkpoint names when constructing the block.
 */
public final class DecoderBlock extends Module {

  private final RMSNorm inputNorm;
  private final GroupedQueryAttention attention;
  private final RMSNorm postAttentionNorm;
  private final SwiGLU mlp;

  /** Creates a decoder block from its already-constructed checkpoint-compatible components. */
  public DecoderBlock(
      MLXScope scope,
      RMSNorm inputNorm,
      GroupedQueryAttention attention,
      RMSNorm postAttentionNorm,
      SwiGLU mlp) {
    super(scope);
    this.inputNorm =
        child(
            "inputNorm",
            Objects.requireNonNull(inputNorm, "DecoderBlock: inputNorm must not be null"));
    this.attention =
        child(
            "attention",
            Objects.requireNonNull(attention, "DecoderBlock: attention must not be null"));
    this.postAttentionNorm =
        child(
            "postAttentionNorm",
            Objects.requireNonNull(
                postAttentionNorm, "DecoderBlock: postAttentionNorm must not be null"));
    this.mlp = child("mlp", Objects.requireNonNull(mlp, "DecoderBlock: mlp must not be null"));
  }

  /** Applies the block to {@code x}, using {@code cache} for this block's attention state. */
  public MLXArray forward(MLXArray x, KVCache cache) {
    Objects.requireNonNull(x, "DecoderBlock.forward: x must not be null");
    MLXArray afterAttention = MLXOps.add(x, attention.forward(inputNorm.forward(x), cache));
    return MLXOps.add(afterAttention, mlp.forward(postAttentionNorm.forward(afterAttention)));
  }
}
