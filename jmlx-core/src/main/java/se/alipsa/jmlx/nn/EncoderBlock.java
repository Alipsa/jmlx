package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;

/** Bidirectional residual attention/FFN block with explicit pre/post-normalization selection. */
public final class EncoderBlock extends Module {
  private final BidirectionalAttention attention;
  private final UnaryLayer attentionNorm;
  private final UnaryLayer feedForward;
  private final UnaryLayer outputNorm;
  private final boolean preNorm;

  /** Registers all children; pre-norm is T5 ordering, post-norm is BERT ordering. */
  public EncoderBlock(
      MLXScope scope,
      BidirectionalAttention attention,
      UnaryLayer attentionNorm,
      UnaryLayer feedForward,
      UnaryLayer outputNorm,
      boolean preNorm) {
    super(scope);
    this.attention = child("attention", attention);
    this.attentionNorm = child("attentionNorm", attentionNorm);
    this.feedForward = child("feedForward", feedForward);
    this.outputNorm = child("outputNorm", outputNorm);
    this.preNorm = preNorm;
  }

  /** Applies key padding and optional additive bias without discarding padded query rows. */
  public MLXArray forward(MLXArray input, MLXArray mask, MLXArray bias) {
    MLXArray residual =
        MLXOps.add(
            input,
            attention.forward(preNorm ? attentionNorm.forward(input) : input, mask, bias, null));
    if (!preNorm) {
      residual = attentionNorm.forward(residual);
    }
    MLXArray output =
        MLXOps.add(
            residual, feedForward.forward(preNorm ? outputNorm.forward(residual) : residual));
    return preNorm ? output : outputNorm.forward(output);
  }
}
