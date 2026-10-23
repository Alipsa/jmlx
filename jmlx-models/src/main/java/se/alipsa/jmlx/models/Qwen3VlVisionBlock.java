package se.alipsa.jmlx.models;

import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.LayerNorm;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.Sequential;
import se.alipsa.jmlx.nn.TwoAxisRotaryAttention;
import se.alipsa.jmlx.nn.TwoAxisRotation;

/**
 * One pre-norm vision transformer block of {@code Qwen3VLVisionBlock} (pinned 4.57.6 source,
 * findings §11.1): the reference's two sequential statements {@code h = h + attn(norm1(h))} then
 * {@code h = h + mlp(norm2(h))} — the second statement's {@code norm2} and residual both see the
 * block's <em>updated</em> residual (the attention output already added), not the block input. The
 * attention sublayer runs on the block's two-axis rotation for this image only (the tower feeds
 * each image's rows as a standalone sequence, so the rows and mask match the reference's
 * variable-length per-image attention chunks exactly; any difference from the reference is kernel
 * precision, not structure).
 *
 * <p>The submodules are supplied by the tower and registered here as the children {@code norm1},
 * {@code norm2}, {@code attention} and {@code mlp}, so the tower's dotted parameter paths mirror
 * the checkpoint's {@code blocks.{i}.norm{1,2}.*}, {@code blocks.{i}.attn.{qkv,proj}.*} and {@code
 * blocks.{i}.mlp.linear_fc{1,2}.*} trees.
 */
public final class Qwen3VlVisionBlock extends Module {

  private final LayerNorm norm1;
  private final LayerNorm norm2;
  private final TwoAxisRotaryAttention attention;
  private final Sequential mlp;

  /**
   * Registers the tower-built submodules as this block's children.
   *
   * @throws NullPointerException if a submodule is {@code null}
   */
  public Qwen3VlVisionBlock(
      MLXScope scope,
      LayerNorm norm1,
      LayerNorm norm2,
      TwoAxisRotaryAttention attention,
      Sequential mlp) {
    super(scope);
    this.norm1 = child("norm1", norm1);
    this.norm2 = child("norm2", norm2);
    this.attention = child("attention", attention);
    this.mlp = child("mlp", mlp);
  }

  /**
   * Runs one pre-norm block over the image's {@code [N, hidden]} rows.
   *
   * @param hidden the block input, {@code [N, hidden]}
   * @param rotation the two-axis rotation prepared for exactly these {@code N} rows
   * @return the block output, {@code [N, hidden]}
   * @throws NullPointerException if an argument is {@code null}
   * @throws IllegalArgumentException if a submodule rejects the shape or rotation
   */
  public MLXArray forward(MLXArray hidden, TwoAxisRotation rotation) {
    MLXArray afterAttention =
        MLXOps.add(hidden, attention.forward(norm1.forward(hidden), rotation));
    return MLXOps.add(afterAttention, mlp.forward(norm2.forward(afterAttention)));
  }
}
