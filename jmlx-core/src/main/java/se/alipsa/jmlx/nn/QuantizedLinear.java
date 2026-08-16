package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * A quantized fully-connected layer: {@code y = quantizedMatmul(x, weight, scales, biases,
 * transpose=true) + bias}, {@code mode} fixed to {@code "affine"} (the only mode with a documented
 * rationale in this codebase -- req/phase4-plan.md, Research findings). {@code weight} is the
 * {@code UINT32}-packed form of a checkpoint-layout {@code [out, in]} weight -- see {@link
 * MLXQuant#quantize} to build one from a float weight, or load one directly from a quantized
 * checkpoint (Phase 5). Unlike {@link Linear}, {@code forward} needs no explicit-target overload
 * for its weight-bearing op: {@link MLXQuant#quantizedMatmul}'s javadoc explains why.
 *
 * <p><strong>Incompatible with {@link ModuleGrad}</strong> -- not merely untested together: {@code
 * QuantizedMatmul}'s native backward pass has no gradient with respect to a quantized weight at all
 * (confirmed empirically: {@code "[QuantizedMatmul::vjp] no gradient wrt the quantized weights."}),
 * so {@code ModuleGrad.of(quantizedLinear, loss).apply(...)} always throws {@code MLXException} on
 * its first {@code apply} call -- after the constructor has already frozen {@code tree}
 * irreversibly, since {@link Module} has no non-trainable/buffer concept to exclude {@code weight}
 * from differentiation. This is an inherent native limitation, not a gap this layer's own code
 * could close: quantization-aware training is out of scope for this layer
 * (req/plans/phase4-m4-plan.md, "Deliberately not covered").
 */
public final class QuantizedLinear extends Module implements UnaryModule {

  private static final String MODE = "affine";

  private final int groupSize;
  private final int bits;
  private final boolean hasBias;

  /**
   * {@code weight} is {@code [out, packedIn]} {@code UINT32} (packed columns = {@code in * bits /
   * 32} -- only for {@code bits} in {@code {2, 4, 8}}; {@code bits} in {@code {3, 5, 6}} pack
   * unevenly and this constructor cannot check them, see below); {@code scales}/{@code biases} are
   * each {@code [out, in / groupSize]}; {@code bias}, if non-null, is {@code [out]}. {@code
   * groupSize}/{@code bits} are each restricted to the exact legal sets this native version
   * supports ({@code {32, 64, 128}} / {@code {2, 3, 4, 5, 6, 8}}, confirmed against the shipped
   * binary, Findings section) rather than merely {@code > 0} -- an out-of-set value would otherwise
   * clear every other check unchanged and fail as an opaque native error at first {@link #forward}
   * call, the same failure mode {@code weight.dtype() != UINT32} exists to prevent for a plain
   * float weight. This hardcodes a native-version-specific set into the layer, the same trade-off
   * Global Constraint 4 already accepts for {@code mode} ("a genuinely new upstream mode would need
   * this validation set updated too") -- accepted here for the same reason: legal range, not a
   * default value (distinct from Global Constraint 5's "absent means let native pick a default"
   * argument). Native validates the packing/group-size relationship on the first {@link #forward}
   * call; this constructor validates only the shape relationships (and now the legal-set
   * membership) it can check without unpacking {@code weight} -- the same division of labor {@link
   * Linear}'s own constructor draws. Non-null: {@code weight}, {@code scales}, {@code biases};
   * {@code bias} is the only nullable parameter. Unlike every {@code MLXQuant} method (Global
   * Constraint 5), a {@code null} {@code scales}/{@code biases} here fails as a bare {@code
   * NullPointerException} out of {@code .ndim()} with no named message -- the same gap {@link
   * Linear}'s own constructor already has for its {@code weight} parameter, so this is
   * precedent-consistent rather than a new one, not a guard this constructor is expected to add.
   */
  public QuantizedLinear(
      MLXScope scope,
      MLXArray weight,
      MLXArray scales,
      MLXArray biases,
      MLXArray bias,
      int groupSize,
      int bits) {
    super(scope);
    if (weight.dtype() != DType.UINT32) {
      throw new IllegalArgumentException(
          "QuantizedLinear: weight must be UINT32-packed, got " + weight.dtype());
    }
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException(
          "QuantizedLinear: weight must be rank 2 [out, packedIn], got shape "
              + Arrays.toString(weight.shape()));
    }
    if (scales.ndim() != 2 || !Arrays.equals(scales.shape(), biases.shape())) {
      throw new IllegalArgumentException(
          "QuantizedLinear: scales and biases must have the same rank-2 shape, got "
              + Arrays.toString(scales.shape())
              + " and "
              + Arrays.toString(biases.shape()));
    }
    if (scales.shape()[0] != weight.shape()[0]) {
      throw new IllegalArgumentException(
          "QuantizedLinear: scales/biases first dimension must match weight's out dimension ("
              + weight.shape()[0]
              + "), got "
              + scales.shape()[0]);
    }
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException(
          "QuantizedLinear: bias must be rank 1 with length weight.shape()[0]="
              + weight.shape()[0]
              + ", got shape "
              + Arrays.toString(bias.shape()));
    }
    if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 && bits != 8) {
      throw new IllegalArgumentException(
          "QuantizedLinear: bits must be one of {2, 3, 4, 5, 6, 8}, got " + bits);
    }
    if (groupSize != 32 && groupSize != 64 && groupSize != 128) {
      throw new IllegalArgumentException(
          "QuantizedLinear: groupSize must be one of {32, 64, 128}, got " + groupSize);
    }
    // packedCols = in * bits / 32 only holds for power-of-2 bits -- of the now-enforced legal
    // set {2, 3, 4, 5, 6, 8} (confirmed against the shipped binary, Findings section), that's
    // exactly {2, 4, 8}: bits 3/5/6 pack unevenly (mlx-c's own get_pack_factor/
    // get_bytes_per_pack), so this check is skipped rather than wrong for them.
    if (32 % bits == 0) {
      long in = (long) scales.shape()[1] * groupSize;
      long expectedPackedCols = in * bits / 32;
      if (weight.shape()[1] != expectedPackedCols) {
        throw new IllegalArgumentException(
            "QuantizedLinear: weight's packed column count ("
                + weight.shape()[1]
                + ") is inconsistent with scales/groupSize/bits (expected "
                + expectedPackedCols
                + " for in="
                + in
                + ", bits="
                + bits
                + ")");
      }
    }
    param("weight", weight);
    param("scales", scales);
    param("biases", biases);
    hasBias = bias != null;
    if (hasBias) {
      param("bias", bias);
    }
    this.groupSize = groupSize;
    this.bits = bits;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    MLXArray y =
        MLXQuant.quantizedMatmul(
            x, param("weight"), param("scales"), param("biases"), true, groupSize, bits, MODE);
    return hasBias ? MLXOps.add(y, param("bias")) : y;
  }
}
