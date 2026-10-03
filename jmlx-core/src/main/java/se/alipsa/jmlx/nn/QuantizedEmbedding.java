package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Objects;
import java.util.SequencedMap;
import java.util.Set;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * An affine-quantized embedding table, in the {@code weight}/{@code scales}/{@code biases} layout
 * that MLX community checkpoints use. {@link #forward} gathers the packed rows, scales and biases
 * for the indices and dequantizes only those rows, so the full table is never expanded; {@link
 * #project} runs the fused quantized matmul against the transposed table, for a tied output head.
 *
 * <p>{@code weight} is {@code [numEmbeddings, packedDim]} {@code UINT32} with {@code packedDim =
 * dim * bits / 32}; {@code scales}/{@code biases} are each {@code [numEmbeddings, dim /
 * groupSize]}. As with {@link QuantizedLinear}, the shapes cannot prove that {@code groupSize} and
 * {@code bits} are the ones the table was quantized with; the caller pairs them.
 *
 * <p>{@code weight} itself is never trainable through {@link ModuleGrad} (no native gradient exists
 * for a quantized weight, as with {@link QuantizedLinear}), so a gradient step never produces a new
 * {@code weight} to write back. {@link #onParametersUpdated(Set)} therefore rejects a write through
 * {@link #update} that touches {@code weight} -- a shape check alone cannot verify a replacement
 * was quantized under this table's own {@code groupSize}/{@code bits} rather than a different pair
 * sharing the same product, and {@code groupSize}/{@code bits} are plain fields {@code update}
 * cannot see -- while a {@code scales}/{@code biases}-only write is validated against this table's
 * current, unchanged {@code weight}/{@code groupSize}/{@code bits} and accepted. Use {@link
 * #updateQuantization} when {@code weight} itself changes: it takes the replacement's {@code
 * groupSize}/{@code bits} explicitly and updates them together with the arrays atomically, so the
 * cached values never describe a different table than the installed one.
 */
public final class QuantizedEmbedding extends EmbeddingLayer {
  private static final String MODE = "affine";

  // Not final: updateQuantization reassigns both together with weight/scales/biases, atomically.
  // See that method's javadoc, and onParametersUpdated's, for why Module.update cannot be trusted
  // to keep these in sync with a caller-supplied weight/scales/biases replacement on its own.
  private int groupSize;
  private int bits;

  /** Creates a quantized embedding over the given packed table. */
  public QuantizedEmbedding(
      MLXScope scope, MLXArray weight, MLXArray scales, MLXArray biases, int groupSize, int bits) {
    super(scope);
    Objects.requireNonNull(weight, "weight");
    Objects.requireNonNull(scales, "scales");
    Objects.requireNonNull(biases, "biases");
    validate(weight, scales, biases, groupSize, bits);
    param("weight", weight);
    param("scales", scales);
    param("biases", biases);
    this.groupSize = groupSize;
    this.bits = bits;
  }

  /**
   * Rejects a {@link #update} write that touches {@code weight} -- a {@code scales}/{@code biases}
   * write is accepted instead, validated against this table's current, unchanged {@code
   * weight}/{@code groupSize}/{@code bits}: {@code weight} (and therefore the pairing {@code
   * groupSize}/{@code bits} describe) never changes in that case, so the constructor's
   * same-product-different-pair ambiguity does not arise. Use {@link #updateQuantization} when
   * {@code weight} itself changes.
   *
   * @throws IllegalStateException if {@code names} contains {@code "weight"}
   * @throws IllegalArgumentException if {@code names} contains {@code "scales"} or {@code "biases"}
   *     and the new values fail {@link #validateScalesAndBiasesAgainstWeight} against the current
   *     weight/groupSize/bits
   */
  @Override
  protected void onParametersUpdated(Set<String> names) {
    if (names.contains("weight")) {
      throw new IllegalStateException(
          "QuantizedEmbedding: weight must not be replaced via Module.update() -- a shape check"
              + " alone cannot verify a replacement was quantized under this table's own"
              + " groupSize/bits rather than a different pair sharing the same product; use"
              + " updateQuantization(...) instead, which takes groupSize/bits explicitly");
    }
    if (names.contains("scales") || names.contains("biases")) {
      validateScalesAndBiasesAgainstWeight(
          param("weight"), param("scales"), param("biases"), groupSize, bits);
    }
  }

  /**
   * Atomically replaces this table's {@code weight}/{@code scales}/{@code biases} together with the
   * {@code groupSize}/{@code bits} they were quantized under -- the only supported way to change
   * {@code weight} itself after construction, since {@link #update} rejects any write that touches
   * it (see {@link #onParametersUpdated(Set)}). Re-runs the constructor's own validation against
   * the new values as a single unit, exactly as if this table had been freshly constructed with
   * them; on rejection, no parameter or field is changed (validation runs before any write).
   *
   * <p>As with {@link QuantizedLinear#updateQuantization}, taking {@code groupSize}/{@code bits}
   * explicitly does not make this table able to verify that {@code weight}/{@code scales}/{@code
   * biases} were actually quantized at the values given -- no shape check can, ever, for the same
   * arithmetic reason. What it closes is the narrower hole where a plain {@link #update} would have
   * silently kept this table's <em>stale</em> {@code groupSize}/{@code bits} while the caller
   * changed the underlying data out from under them.
   *
   * @throws NullPointerException if {@code weight}, {@code scales}, or {@code biases} is {@code
   *     null}
   * @throws IllegalArgumentException if the new {@code weight}/{@code scales}/{@code biases}/{@code
   *     groupSize}/{@code bits} fail the same checks the constructor runs
   */
  public void updateQuantization(
      MLXArray weight, MLXArray scales, MLXArray biases, int groupSize, int bits) {
    Objects.requireNonNull(weight, "QuantizedEmbedding.updateQuantization: weight");
    Objects.requireNonNull(scales, "QuantizedEmbedding.updateQuantization: scales");
    Objects.requireNonNull(biases, "QuantizedEmbedding.updateQuantization: biases");
    validate(weight, scales, biases, groupSize, bits);
    SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
    values.put("weight", weight);
    values.put("scales", scales);
    values.put("biases", biases);
    rebind(values);
    this.groupSize = groupSize;
    this.bits = bits;
  }

  @Override
  public MLXArray forward(MLXArray indices) {
    MLXArray rows = MLXShape.takeAxis(param("weight"), indices, 0);
    MLXArray rowScales = MLXShape.takeAxis(param("scales"), indices, 0);
    MLXArray rowBiases = MLXShape.takeAxis(param("biases"), indices, 0);
    return MLXQuant.dequantize(rows, rowScales, rowBiases, groupSize, bits, MODE, null, null);
  }

  @Override
  public MLXArray project(MLXArray hiddenStates) {
    return MLXQuant.quantizedMatmul(
        hiddenStates,
        param("weight"),
        param("scales"),
        param("biases"),
        true,
        groupSize,
        bits,
        MODE,
        hiddenStates.scope());
  }

  private static void validate(
      MLXArray weight, MLXArray scales, MLXArray biases, int groupSize, int bits) {
    if (weight.dtype() != DType.UINT32 || weight.ndim() != 2) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: weight must be rank-2 UINT32, got "
              + weight.dtype()
              + " "
              + Arrays.toString(weight.shape()));
    }
    if (groupSize != 32 && groupSize != 64 && groupSize != 128) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: groupSize must be one of {32, 64, 128}, got " + groupSize);
    }
    if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 && bits != 8) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: bits must be one of {2, 3, 4, 5, 6, 8}, got " + bits);
    }
    validateScalesAndBiasesAgainstWeight(weight, scales, biases, groupSize, bits);
  }

  /**
   * Validates {@code scales}/{@code biases} against {@code weight}/{@code groupSize}/{@code bits}:
   * both must be floating, share the same rank-2 shape, have a first dimension matching {@code
   * weight}'s row count, and {@code weight}'s packed column count must match the {@code
   * groupSize}/{@code bits}-derived expectation. Shared by the constructor (via {@link #validate},
   * after {@code groupSize}/{@code bits} have already been checked against their legal sets there)
   * and {@link #onParametersUpdated(Set)}'s {@code scales}/{@code biases} branch (against the
   * just-written new values and the current, unchanged {@code weight}/{@code groupSize}/{@code
   * bits}).
   */
  private static void validateScalesAndBiasesAgainstWeight(
      MLXArray weight, MLXArray scales, MLXArray biases, int groupSize, int bits) {
    if (!scales.dtype().isInexact() || !biases.dtype().isInexact()) {
      throw new IllegalArgumentException("QuantizedEmbedding: scales and biases must be floating");
    }
    if (scales.ndim() != 2
        || !Arrays.equals(scales.shape(), biases.shape())
        || scales.shape()[0] != weight.shape()[0]) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: scales/biases must be [numEmbeddings, dim / groupSize], got "
              + Arrays.toString(scales.shape())
              + " and "
              + Arrays.toString(biases.shape()));
    }
    long packed = (long) scales.shape()[1] * groupSize * bits / 32;
    if (weight.shape()[1] != packed) {
      throw new IllegalArgumentException(
          "QuantizedEmbedding: weight has "
              + weight.shape()[1]
              + " packed columns but scales imply "
              + packed
              + " for groupSize "
              + groupSize
              + ", bits "
              + bits);
    }
  }
}
