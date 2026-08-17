package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Objects;
import java.util.SequencedMap;
import java.util.Set;
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
 * <p><strong>{@code weight} itself cannot be trained through {@link ModuleGrad}</strong> -- not
 * merely untested: {@code QuantizedMatmul}'s native backward pass has no gradient with respect to a
 * quantized weight at all (confirmed empirically: {@code "[QuantizedMatmul::vjp] no gradient wrt
 * the quantized weights."}). {@link ModuleGrad} excludes any non-floating-dtype parameter from
 * differentiation entirely, by dtype ({@link se.alipsa.jmlx.core.DType#isInexact()}) -- confirmed
 * empirically that excluding {@code weight}'s index from {@code argnums} avoids the native failure
 * outright, rather than merely deferring it, so {@code scales}/{@code biases}/{@code bias}
 * <em>are</em> trainable through {@code ModuleGrad.of(quantizedLinear, loss).apply(...)} exactly
 * like any other parameter, whether or not {@code loss} actually reaches this layer's {@link
 * #forward} (see {@link ModuleGrad}'s own javadoc for the general rule this is an instance of).
 * Only {@code weight} is excluded: this is an inherent native limitation, not a gap this layer's
 * own code could close -- full quantization-aware training (updating the quantized weight itself)
 * is out of scope for this layer (req/plans/phase4-m4-plan.md, "Deliberately not covered").
 *
 * <p>Since {@code weight} is never trainable, a gradient step never has a new {@code weight} to
 * write back -- only {@code scales}/{@code biases} (and, separately, {@code bias}, which has no
 * quantization relationship to {@code groupSize}/{@code bits} at all). {@link
 * #onParametersUpdated(Set)} only rejects a write through {@link #update} that touches {@code
 * weight} itself; a {@code scales}/{@code biases}-only write (with or without {@code bias}) is
 * validated against this layer's current, unchanged {@code weight}/{@code groupSize}/{@code bits}
 * and accepted -- so a generic training loop applying {@code ModuleGrad}'s gradients back onto a
 * tree containing this layer can do so through a single, ordinary {@code tree.update(grads)} call
 * spanning every sibling, without needing a typed reference to this layer at all (see {@link
 * #onParametersUpdated(Set)}'s own javadoc for why this is safe even though the identical
 * replacement of {@code weight} itself is not). {@link #updateScalesAndBiases} remains available as
 * a direct, single-layer alternative when a typed reference is already in hand.
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
   * Rejects a {@link #update} write that touches {@code weight} -- {@code scales}/{@code biases}
   * (with or without {@code bias}) are both accepted, subject to validation, and so is {@code bias}
   * alone: {@code weight} itself is what {@code groupSize}/{@code bits} describe, so a write that
   * replaces it without also replacing them (they are plain fields, not registered parameters
   * {@code update} can see) could pair a new {@code weight} with this layer's <em>stale,
   * possibly-no-longer-true</em> {@code groupSize}/{@code bits} -- exactly the check the
   * constructor's own javadoc explains is provably unable to distinguish a same-shape replacement
   * from one quantized under a genuinely different {@code (groupSize, bits)} pair sharing the same
   * product. That ambiguity is specifically about {@code weight} changing without {@code
   * groupSize}/{@code bits} changing with it -- it does NOT apply to a {@code scales}/{@code
   * biases}-only write, where {@code weight} (and therefore the pairing {@code groupSize}/{@code
   * bits} describe) never changes at all: {@link #validateScalesAndBiasesAgainstWeight} validates
   * the new {@code scales}/{@code biases} against the current, unchanged {@code weight}/{@code
   * groupSize}/{@code bits} exactly as {@link #updateScalesAndBiases} does, and either accepts the
   * write or throws (triggering {@link Module#update}'s own write-rollback). Nor does it apply to
   * {@code bias}, which has no quantization relationship to {@code groupSize}/{@code bits} at all
   * -- but {@code bias} still has its own, independent shape invariant relative to {@code weight}
   * (rank 1, length {@code weight.shape()[0]}), which the constructor enforces and this hook now
   * re-validates too, rather than letting {@link #update} install a {@code bias} the constructor
   * would have rejected. An earlier version of this override rejected {@code scales}/{@code biases}
   * unconditionally too, on the mistaken premise that the constructor's ambiguity applied to them
   * the same way it applies to {@code weight} -- it does not, since that ambiguity is inherently
   * about {@code weight}'s own identity, not about {@code scales}/{@code biases} in isolation. A
   * still earlier version rejected every write to any of this layer's own parameters
   * unconditionally, including a {@code bias}-only write that has no bearing on {@code
   * groupSize}/{@code bits} at all -- {@code names} (added together with {@link
   * Module#onParametersUpdated(Set)}) is what lets this override discriminate.
   *
   * <p>Use {@link #updateQuantization} instead of a plain {@link #update} only when {@code weight}
   * itself is changing: it takes the replacement's {@code groupSize}/{@code bits} explicitly and
   * updates them together with the arrays as a single atomic operation -- the only way this layer's
   * cached {@code groupSize}/{@code bits} can ever be trusted to actually describe a NEW {@code
   * weight} (see that method's own javadoc for why even it cannot verify the pairing, only keep the
   * three from silently drifting apart). {@link #update}/{@link #updateScalesAndBiases} are equally
   * valid, interchangeable ways to write a {@code scales}/{@code biases}-only change (e.g. a
   * gradient step -- {@code weight} itself is never trainable, see this class's own javadoc):
   * {@link #update} additionally notifies (this hook, a no-op after validation succeeds) and
   * participates in a multi-module {@link Module#update} call's rollback; {@link
   * #updateScalesAndBiases} is a direct, single-layer call that skips both.
   *
   * @throws IllegalStateException if {@code names} contains {@code "weight"}
   * @throws IllegalArgumentException if {@code names} contains {@code "scales"} or {@code "biases"}
   *     and the new value fails {@link #validateScalesAndBiasesAgainstWeight}
   * @throws IllegalArgumentException if {@code names} contains {@code "bias"} and the new value is
   *     not rank 1 with length {@code weight.shape()[0]}
   */
  @Override
  protected void onParametersUpdated(Set<String> names) {
    if (names.contains("weight")) {
      throw new IllegalStateException(
          "QuantizedLinear: weight must not be replaced via Module.update() -- a shape check alone"
              + " cannot verify a replacement was quantized under this layer's own groupSize/bits"
              + " rather than a different pair sharing the same product; use"
              + " updateQuantization(...) instead, which takes groupSize/bits explicitly");
    }
    if (names.contains("scales") || names.contains("biases")) {
      validateScalesAndBiasesAgainstWeight(
          param("weight"), param("scales"), param("biases"), groupSize, bits);
    }
    if (names.contains("bias")) {
      validateBiasAgainstWeight(param("weight"), param("bias"));
    }
  }

  /**
   * Atomically replaces this layer's {@code weight}/{@code scales}/{@code biases}/{@code bias}
   * together with the {@code groupSize}/{@code bits} they were quantized under -- the only
   * supported way to change {@code weight} itself after construction, since {@link #update} rejects
   * any write that touches it (see {@link #onParametersUpdated(Set)}). Re-runs the constructor's
   * own validation against the new values as a single unit, exactly as if this layer had been
   * freshly constructed with them; on rejection, no parameter or field is changed (validation runs
   * before any write). {@code bias}'s nullness must match whether this layer was originally
   * constructed with one: {@link #rebind} (which this method uses to write the {@code
   * MLXArray}-typed parameters, so {@link #onParametersUpdated(Set)} does not fire for this call)
   * can replace an existing parameter's value but cannot register or remove one.
   *
   * <p>Requiring {@code groupSize}/{@code bits} explicitly, every time, does not make this layer
   * able to verify that {@code weight}/{@code scales}/{@code biases} were actually quantized at the
   * values given -- see the constructor's own javadoc: no shape check can do that, ever, for the
   * same arithmetic reason. What it does close is the narrower, previously real hole where {@link
   * #update} silently kept this layer's <em>stale</em> {@code groupSize}/{@code bits} while the
   * caller changed the underlying data out from under them: {@link #update} itself can no longer do
   * that (it rejects any write touching {@code weight} outright, per {@link
   * #onParametersUpdated(Set)}).
   *
   * <p><strong>{@link #rebind} is a separate, un-narrowable hole this method cannot close.</strong>
   * {@code rebind} is {@code public} and {@code final} on {@link Module} -- by design, so {@code
   * ModuleGrad}'s traced-primal swap (which must never validate or notify) keeps working for every
   * subclass -- so this class has no way to intercept a direct {@code
   * quantizedLinear.rebind(Map.of("weight", someOtherWeight))} call at all: confirmed empirically,
   * such a call succeeds silently even when the replacement's shape is inconsistent with this
   * layer's cached {@code groupSize}/{@code bits} (every constructor invariant broken, no
   * exception), and a {@code FLOAT32} replacement succeeds too, only failing later, deep inside
   * native code, at the next {@link #forward} call. This method being the sole path through {@code
   * this class's own API} does not mean {@code weight}/{@code groupSize}/{@code bits} can only ever
   * change together in practice -- only that no method <em>this class defines</em> can change them
   * apart from each other. A caller that bypasses this class's API via the inherited {@code rebind}
   * is outside what any validation here can reach.
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

  /**
   * Replaces this layer's {@code scales}/{@code biases} alone, leaving {@code weight}/{@code
   * bias}/{@code groupSize}/{@code bits} untouched -- the escape hatch a generic training loop
   * needs to write a gradient step's {@code scales}/{@code biases} back onto this layer: {@code
   * weight} itself is never trainable through {@link ModuleGrad} (no native gradient exists for it
   * at all, see this class's own javadoc), so a gradient step never produces a new {@code weight}
   * to write back, and routing a {@code scales}/{@code biases}-only change through {@link
   * #updateQuantization} would force the caller to also re-supply {@code weight}/{@code bias}/
   * {@code groupSize}/{@code bits} unchanged, none of which this method's caller has any reason to
   * have on hand.
   *
   * <p>Unlike {@link #updateQuantization}, this method takes no {@code groupSize}/{@code bits}: the
   * replacement is validated against this layer's <em>current</em> {@code weight}/{@code
   * groupSize}/{@code bits} instead, via the same {@link #validateScalesAndBiasesAgainstWeight}
   * helper {@link #onParametersUpdated(Set)}'s own {@code scales}/{@code biases} branch now uses.
   * Neither {@code weight} nor {@code groupSize}/{@code bits} is actually immutable -- {@code
   * groupSize}/{@code bits} are plain, non-{@code final} fields reassigned by {@link
   * #updateQuantization}, and {@code weight} itself can be replaced via the inherited {@link
   * #rebind} (see that method's own javadoc for why this class cannot prevent that) -- but neither
   * changes as a side effect of THIS call, which is all the validation below actually needs: since
   * {@code weight} does not change here, the groupSize/bits-versus-arithmetic ambiguity the
   * constructor's own javadoc documents (a shape check alone cannot distinguish one {@code
   * (groupSize, bits)} pair from a different one sharing the same product) simply does not arise:
   * nothing about the quantized {@code weight} this layer's {@code groupSize}/{@code bits} describe
   * is changing, so there is nothing for that ambiguity to apply to.
   *
   * <p>Uses {@link #rebind}, not {@link #update}, so {@link #onParametersUpdated(Set)} does not
   * fire for this call -- the same choice {@link #updateQuantization} already makes, for the same
   * reason ({@link #rebind}'s own contract: replace an existing parameter's value without
   * notifying). As of this PR's round-12 review finding 1, a plain {@link Module#update} call also
   * accepts a {@code scales}/{@code biases}-only write (validated identically), so this method is
   * now a convenience for a caller that already holds a typed reference to this layer directly, not
   * the only route -- see {@link #onParametersUpdated(Set)}'s own javadoc.
   *
   * @throws NullPointerException if {@code scales} or {@code biases} is {@code null}
   * @throws IllegalArgumentException if {@code scales}/{@code biases} fails the same
   *     scales/biases-versus-weight checks {@link #onParametersUpdated(Set)}'s own {@code
   *     scales}/{@code biases} branch runs
   */
  public void updateScalesAndBiases(MLXArray scales, MLXArray biases) {
    Objects.requireNonNull(
        scales, "QuantizedLinear.updateScalesAndBiases: scales must not be null");
    Objects.requireNonNull(
        biases, "QuantizedLinear.updateScalesAndBiases: biases must not be null");
    validateScalesAndBiasesAgainstWeight(param("weight"), scales, biases, groupSize, bits);
    SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
    values.put("scales", scales);
    values.put("biases", biases);
    rebind(values);
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
    validateBiasAgainstWeight(weight, bias);
    if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 && bits != 8) {
      throw new IllegalArgumentException(
          "QuantizedLinear: bits must be one of {2, 3, 4, 5, 6, 8}, got " + bits);
    }
    if (groupSize != 32 && groupSize != 64 && groupSize != 128) {
      throw new IllegalArgumentException(
          "QuantizedLinear: groupSize must be one of {32, 64, 128}, got " + groupSize);
    }
    validateScalesAndBiasesAgainstWeight(weight, scales, biases, groupSize, bits);
  }

  /**
   * Validates {@code scales}/{@code biases} against {@code weight}/{@code groupSize}/{@code bits}:
   * both must be a floating dtype, share the same rank-2 shape, have a first dimension matching
   * {@code weight}'s out dimension, and {@code weight}'s packed column count must match the {@code
   * groupSize}/{@code bits}-derived expectation. Shared by the constructor (via {@link #validate},
   * after {@code bits}/{@code groupSize} have already been checked against their legal sets there),
   * {@link #updateScalesAndBiases} (against the current, unchanged {@code weight}/ {@code
   * groupSize}/{@code bits}), and {@link #onParametersUpdated(Set)}'s {@code scales}/{@code biases}
   * branch (against the just-written new values and the current, unchanged {@code weight}/{@code
   * groupSize}/{@code bits}) -- {@code weight} itself is never touched at either of the latter two
   * call sites, so the groupSize/bits-versus-arithmetic ambiguity that makes a shape check
   * untrustworthy when {@code weight} DOES change (the constructor's own javadoc) does not apply to
   * either of them: there is only one {@code weight} in play throughout, so there is nothing for a
   * same-product-different-pair substitution to hide behind.
   *
   * <p>packedCols = {@code in * bits / 32} holds for every legal {@code bits} value {@code {2, 3,
   * 4, 5, 6, 8}}, not just the power-of-2 subset -- confirmed empirically (probed {@code in} in
   * {@code {64, 96, 128, 160, 224}} x {@code bits} in {@code {3, 5, 6}}: {@code w_q}'s last dim
   * matched {@code in * bits / 32} exactly every time, all {@code UINT32}). The formula is always
   * integral here because {@code groupSize} is restricted to {@code {32, 64, 128}} by every caller
   * that can still reject an out-of-set value (the constructor, via {@link #validate}), so {@code
   * in} ({@code = scales.shape()[1] * groupSize}) is always a multiple of 32 -- the byte-level pack
   * factor mlx-c's own {@code get_pack_factor}/{@code get_bytes_per_pack} describes never actually
   * divides a word unevenly at these sizes. An earlier version of this check ran only when {@code
   * 32 % bits == 0}, on the wrong belief that {@code bits} in {@code {3, 5, 6}} pack unevenly
   * enough to break the formula; that gated check silently let a {@code groupSize}/{@code bits}
   * mismatch through construction, deferring it to an opaque native error at first {@link #forward}
   * call -- exactly the failure mode this check exists to prevent.
   */
  private static void validateScalesAndBiasesAgainstWeight(
      MLXArray weight, MLXArray scales, MLXArray biases, int groupSize, int bits) {
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
   * Validates {@code bias} against {@code weight}: {@code null} is always accepted (a layer may
   * legitimately have no {@code bias}); a non-null {@code bias} must be rank 1 with length {@code
   * weight.shape()[0]}. Shared by the constructor (via {@link #validate}) and {@link
   * #onParametersUpdated(Set)}'s own {@code bias} branch, so a {@code bias}-only {@link #update}
   * cannot install a value the constructor would have rejected -- {@code bias} has no quantization
   * relationship to {@code groupSize}/{@code bits} (unlike {@code weight}/{@code scales}/{@code
   * biases}), but it still has this independent shape invariant relative to {@code weight} -- an
   * earlier version of {@link #onParametersUpdated(Set)} let a {@code bias}-only write through
   * completely unvalidated, so this check was the one thing standing between a {@code bias} write
   * and the constructor's own invariant for it. Confirmed empirically before this fix existed: a
   * {@code [2,2]}-shaped {@code bias} replacement (rank 2, not rank 1) succeeded via {@link
   * #update} with no exception at all, and {@link #forward}'s {@code MLXOps.add(y, bias)} then
   * broadcast {@code y}'s {@code [1,2]} against it rather than failing, silently producing a {@code
   * [2,2]} result instead of the correct {@code [1,2]}.
   */
  private static void validateBiasAgainstWeight(MLXArray weight, MLXArray bias) {
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException(
          "QuantizedLinear: bias must be rank 1 with length weight.shape()[0]="
              + weight.shape()[0]
              + ", got shape "
              + Arrays.toString(bias.shape()));
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
