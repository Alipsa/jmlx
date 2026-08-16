package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.SequencedMap;
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
 * checkpoint (Phase 5). {@code forward} passes {@code x.scope()} explicitly to {@link
 * MLXQuant#quantizedMatmul}'s target overload, the same reason {@link Linear#forward} targets
 * {@code x.scope()} for its own weight-bearing op (req/plans/phase4-m4-plan.md's Amendment): the
 * default (innermost-of-all-operands) overload leaks into the model scope once per call whenever
 * the model was built in a scope that is itself a descendant of {@code x}'s own.
 *
 * <p><strong>Incompatible with {@link ModuleGrad}</strong> -- not merely untested together: {@code
 * QuantizedMatmul}'s native backward pass has no gradient with respect to a quantized weight at all
 * (confirmed empirically: {@code "[QuantizedMatmul::vjp] no gradient wrt the quantized weights."}).
 * {@code ModuleGrad.of(quantizedLinear, loss).apply(...)} throws {@code MLXException} on its first
 * {@code apply} call whenever {@code loss} actually calls this layer's {@link #forward} -- the only
 * way this layer's registration in {@code tree} has any effect on the computed loss at all, and so
 * the normal case in practice -- not unconditionally for every possible {@code loss} (see {@link
 * ModuleGrad}'s own javadoc for the general rule this is an instance of: a {@code loss} that never
 * reaches a {@code QuantizedLinear}'s quantized weight does not trigger this failure, whether that
 * layer is the tree root passed to {@code ModuleGrad.of} directly or a descendant of it). This is
 * an inherent native limitation, not a gap this layer's own code could close: quantization-aware
 * training is out of scope for this layer (req/plans/phase4-m4-plan.md, "Deliberately not
 * covered").
 */
public final class QuantizedLinear extends Module implements UnaryModule {

  private static final String MODE = "affine";

  // Not final: updateQuantization reassigns both together with weight/scales/biases, atomically.
  // See that method's javadoc, and onParametersUpdated's, for why Module.update cannot be trusted
  // to keep these in sync with a caller-supplied weight/scales/biases replacement on its own.
  private int groupSize;
  private int bits;
  private final boolean hasBias;

  /**
   * {@code weight} is {@code [out, packedIn]} {@code UINT32} (packed columns = {@code in * bits /
   * 32} -- holds for every legal {@code bits} value, {@code {2, 3, 4, 5, 6, 8}}, not just the
   * power-of-2 subset {@code get_pack_factor}/{@code get_bytes_per_pack}'s byte-level packing might
   * suggest: {@code group_size}'s own legal set forces {@code in} to always be a multiple of 32, so
   * the byte-level unevenness never surfaces at this word granularity -- confirmed empirically, see
   * below); {@code scales}/{@code biases} are each {@code [out, in / groupSize]}; {@code bias}, if
   * non-null, is {@code [out]}. {@code groupSize}/{@code bits} are each restricted to the exact
   * legal sets this native version supports ({@code {32, 64, 128}} / {@code {2, 3, 4, 5, 6, 8}},
   * confirmed against the shipped binary, Findings section) rather than merely {@code > 0} -- an
   * out-of-set value would otherwise clear every other check unchanged and fail as an opaque native
   * error at first {@link #forward} call, the same failure mode {@code weight.dtype() != UINT32}
   * exists to prevent for a plain float weight. This hardcodes a native-version-specific set into
   * the layer, the same trade-off Global Constraint 4 already accepts for {@code mode} ("a
   * genuinely new upstream mode would need this validation set updated too") -- accepted here for
   * the same reason: legal range, not a default value (distinct from Global Constraint 5's "absent
   * means let native pick a default" argument). Native validates the packing/group-size
   * relationship on the first {@link #forward} call; this constructor validates only the shape
   * relationships, the legal-set membership, and (since {@code scales}/{@code biases} are not
   * dtype-checked by anything upstream of native itself, confirmed empirically: an {@code INT32}
   * pair passes construction and only fails at {@link #forward} with {@code "[quantized_matmul]
   * Only real floating types are supported"}) that both are one of the floating dtypes {@link
   * se.alipsa.jmlx.core.DType#isInexact()} recognizes -- everything it can check without unpacking
   * {@code weight} -- the same division of labor {@link Linear}'s own constructor draws. Non-null:
   * {@code weight}, {@code scales}, {@code biases}; {@code bias} is the only nullable parameter.
   * Unlike every {@code MLXQuant} method (Global Constraint 5), a {@code null} {@code
   * scales}/{@code biases} here fails as a bare {@code NullPointerException} out of {@code .ndim()}
   * with no named message -- the same gap {@link Linear}'s own constructor already has for its
   * {@code weight} parameter, so this is precedent-consistent rather than a new one, not a guard
   * this constructor is expected to add.
   *
   * <p><strong>This constructor cannot verify that {@code weight}/{@code scales}/{@code biases}
   * were actually quantized using the {@code groupSize}/{@code bits} given here</strong> -- only
   * that their shapes are consistent with <em>some</em> quantization at that {@code groupSize}/
   * {@code bits}. The packed-column formula above depends only on the product {@code groupSize *
   * bits}, never on the two values independently, so a {@code weight}/{@code scales} pair actually
   * produced by {@link MLXQuant#quantize} at a <em>different</em> {@code (groupSize, bits)} sharing
   * that same product (e.g. {@code groupSize=64, bits=2} passed off as {@code groupSize=32,
   * bits=4}: {@code 64*2 == 32*4}) satisfies every check this constructor can run, no matter how
   * the checks are written -- this is a property of the arithmetic itself, not a gap a smarter
   * shape check could close (confirmed empirically: {@code forward} on such a mismatched instance
   * neither throws nor silently produces obviously-garbage shapes; it silently computes wrong
   * numbers, whenever the caller's activation width happens to match the wrongly-derived {@code
   * in}). Callers are responsible for pairing {@code weight}/{@code scales}/{@code biases} with the
   * exact {@code groupSize}/{@code bits} that produced them; this constructor's checks exist to
   * catch shape mistakes (wrong rank, wrong dimension, wrong dtype, an out-of-set {@code
   * groupSize}/{@code bits}), not to verify that pairing.
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
    validate(weight, scales, biases, bias, groupSize, bits);
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

  /**
   * Always rejects: {@link #update} lets a caller replace {@code weight}/{@code scales}/{@code
   * biases}/{@code bias} without touching {@code groupSize}/{@code bits} at all (they are plain
   * fields, not registered parameters {@code update} can see), so a shape check re-run here could
   * only ever validate the replacement against this layer's <em>stale, possibly-no-longer-true</em>
   * {@code groupSize}/{@code bits} -- exactly the check the constructor's own javadoc explains is
   * provably unable to distinguish a same-shape replacement from one quantized under a genuinely
   * different {@code (groupSize, bits)} pair sharing the same product. An earlier version of this
   * override ran that check anyway and documented the gap as a narrow "blind spot"; that
   * documentation was itself wrong -- confirmed empirically, the gap is not narrow (it is every
   * {@code (groupSize, bits)} pair sharing a product with this layer's cached one, not one specific
   * coincidence) and not merely a missed native error (the mismatch can silently compute wrong
   * numbers with no error at all, if the caller's activation width happens to agree with the value
   * {@code in} the stale {@code groupSize} wrongly derives). Re-running a check that cannot detect
   * the exact failure mode it exists to catch is worse than no check: it advertises a safety
   * guarantee this class cannot provide through {@link #update}.
   *
   * <p>Use {@link #updateQuantization} instead, which takes the replacement's {@code groupSize}/
   * {@code bits} explicitly and updates them together with the arrays as a single atomic operation
   * -- the only way this layer's cached {@code groupSize}/{@code bits} can ever be trusted to
   * actually describe the arrays they are paired with after construction. {@link Module#update}
   * rolls the write back to its pre-call value when this throws (req/plans/phase4-m4-plan.md's
   * Amendment covers that fix in {@code Module} itself), so calling {@link #update} on this layer's
   * own parameters is a no-op other than the thrown exception -- never called from {@link #rebind}
   * (see {@link Module#onParametersUpdated()}), so {@code ModuleGrad}'s rebind-around-a-loss-call
   * usage is unaffected, though {@link ModuleGrad} is independently incompatible with this class
   * for the unrelated reason this class's own javadoc explains.
   *
   * @throws IllegalStateException always
   */
  @Override
  protected void onParametersUpdated() {
    throw new IllegalStateException(
        "QuantizedLinear: weight/scales/biases/bias must not be replaced via Module.update() -- a"
            + " shape check alone cannot verify a replacement was quantized under this layer's own"
            + " groupSize/bits rather than a different pair sharing the same product; use"
            + " updateQuantization(...) instead, which takes groupSize/bits explicitly");
  }

  /**
   * Atomically replaces this layer's {@code weight}/{@code scales}/{@code biases}/{@code bias}
   * together with the {@code groupSize}/{@code bits} they were quantized under -- the only
   * supported way to change this layer's quantization configuration after construction, since
   * {@link #update} always rejects a {@code weight}/{@code scales}/{@code biases}/{@code bias}
   * replacement (see {@link #onParametersUpdated()}). Re-runs the constructor's own validation
   * against the new values as a single unit, exactly as if this layer had been freshly constructed
   * with them; on rejection, no parameter or field is changed (validation runs before any write).
   * {@code bias}'s nullness must match whether this layer was originally constructed with one:
   * {@link #rebind} (which this method uses to write the {@code MLXArray}-typed parameters, so
   * {@link #onParametersUpdated()} does not fire for this call) can replace an existing parameter's
   * value but cannot register or remove one.
   *
   * <p>Requiring {@code groupSize}/{@code bits} explicitly, every time, does not make this layer
   * able to verify that {@code weight}/{@code scales}/{@code biases} were actually quantized at the
   * values given -- see the constructor's own javadoc: no shape check can do that, ever, for the
   * same arithmetic reason. What it does close is the narrower, previously real hole where {@link
   * #update} silently kept this layer's <em>stale</em> {@code groupSize}/{@code bits} while the
   * caller changed the underlying data out from under them; that specific mismatch is now
   * impossible, because {@code groupSize}/{@code bits} and the arrays they describe can only ever
   * change together, through this one method.
   *
   * @throws NullPointerException if {@code weight}, {@code scales}, or {@code biases} is {@code
   *     null}
   * @throws IllegalArgumentException if the new {@code weight}/{@code scales}/{@code biases}/{@code
   *     bias}/{@code groupSize}/{@code bits} fail the same checks the constructor runs
   * @throws IllegalArgumentException if {@code bias}'s nullness does not match whether this layer
   *     was constructed with one
   */
  public void updateQuantization(
      MLXArray weight, MLXArray scales, MLXArray biases, MLXArray bias, int groupSize, int bits) {
    validate(weight, scales, biases, bias, groupSize, bits);
    if ((bias != null) != hasBias) {
      throw new IllegalArgumentException(
          "QuantizedLinear.updateQuantization: bias must be "
              + (hasBias ? "non-null (this layer was constructed with one)" : "null")
              + ", got "
              + (bias == null ? "null" : "non-null"));
    }
    SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
    values.put("weight", weight);
    values.put("scales", scales);
    values.put("biases", biases);
    if (hasBias) {
      values.put("bias", bias);
    }
    rebind(values);
    this.groupSize = groupSize;
    this.bits = bits;
  }

  private static void validate(
      MLXArray weight, MLXArray scales, MLXArray biases, MLXArray bias, int groupSize, int bits) {
    if (weight.dtype() != DType.UINT32) {
      throw new IllegalArgumentException(
          "QuantizedLinear: weight must be UINT32-packed, got " + weight.dtype());
    }
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException(
          "QuantizedLinear: weight must be rank 2 [out, packedIn], got shape "
              + Arrays.toString(weight.shape()));
    }
    if (!scales.dtype().isInexact()) {
      throw new IllegalArgumentException(
          "QuantizedLinear: scales must be a floating dtype (FLOAT32, FLOAT16, or BFLOAT16), got "
              + scales.dtype());
    }
    if (!biases.dtype().isInexact()) {
      throw new IllegalArgumentException(
          "QuantizedLinear: biases must be a floating dtype (FLOAT32, FLOAT16, or BFLOAT16), got "
              + biases.dtype());
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
    // packedCols = in * bits / 32 holds for every legal bits value {2, 3, 4, 5, 6, 8}, not just
    // the power-of-2 subset -- confirmed empirically (probed in in {64, 96, 128, 160, 224} x bits
    // in {3, 5, 6}: w_q's last dim matched in * bits / 32 exactly every time, all UINT32). The
    // formula is always integral here because group_size is restricted to {32, 64, 128} above, so
    // in (= scales.shape()[1] * groupSize) is always a multiple of 32 -- the byte-level pack
    // factor mlx-c's own get_pack_factor/get_bytes_per_pack describes never actually divides a
    // word unevenly at these sizes. An earlier version of this check ran only when 32 % bits == 0,
    // on the wrong belief that bits 3/5/6 pack unevenly enough to break the formula; that gated
    // check silently let a groupSize/bits mismatch through construction, deferring it to an opaque
    // native error at first forward() -- exactly the failure mode this check exists to prevent.
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

  /**
   * {@code y = quantizedMatmul(x, weight, scales, biases, transpose=true)}, targeting {@code
   * x.scope()} explicitly (see this class's own javadoc) {@code + bias} when {@link #hasBias}. That
   * final {@code add} is not scope-targeted: {@link MLXOps#add} has no explicit-target overload, so
   * under the inverted layout the class javadoc describes, the {@code hasBias} branch can still
   * leak one array into the model scope per call -- the exact same residual gap {@link
   * Linear#forward}'s own {@code hasBias} branch has today (confirmed empirically), not a new one
   * this class introduces. Closing it would mean adding a target overload to {@link MLXOps#add}
   * itself, which is shared by every op in this codebase, not something scoped to this class.
   */
  @Override
  public MLXArray forward(MLXArray x) {
    MLXArray y =
        MLXQuant.quantizedMatmul(
            x,
            param("weight"),
            param("scales"),
            param("biases"),
            true,
            groupSize,
            bits,
            MODE,
            x.scope());
    return hasBias ? MLXOps.add(y, param("bias")) : y;
  }
}
