package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.Set;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * {@code quantize}, {@code dequantize} and {@code quantizedMatmul} (req/phase4-plan.md §8): affine
 * weight quantization and the fused matmul {@code QuantizedLinear}, in the neural-network package,
 * is built on (same "package named in prose, not a qualified {@code @link}" convention {@link
 * MLXGrad}'s own javadoc already uses for {@code ModuleGrad}, for the same reason: spec
 * Verification #6a's grep must never match a core class's source). See {@link MLX}'s javadoc for
 * the index of every sibling class in this package's split.
 */
public final class MLXQuant {

  private static final Set<String> MODES = Set.of("affine", "mxfp4", "mxfp8", "nvfp4");

  private MLXQuant() {}

  /**
   * Validates {@code mode} against the four upstream {@code QuantizationMode} values (Findings
   * section) before it ever reaches {@link NativeOps#cstr} -- {@code cstr}'s own javadoc documents
   * an intern-cache contract sized for a bounded, {@code private static final}-held set of
   * literals; a raw pass-through of this caller-supplied parameter would let a typo'd or
   * dynamically-built string grow {@code FACADE_ARENA} by one segment per distinct value ever seen,
   * and NPE out of {@code computeIfAbsent} with no named error on a {@code null} mode. {@code mode
   * == null} is checked explicitly, not left to {@code MODES.contains(null)} -- {@code
   * Set.of(...)}'s returned set throws {@code NullPointerException} out of {@code contains} on a
   * {@code null} argument (confirmed: {@code Set.of("a").contains(null)} throws), so a naive port
   * of this guard would just move the unnamed NPE from {@code cstr}'s cache to here instead of
   * fixing it.
   */
  private static void checkMode(String opName, String mode) {
    if (mode == null || !MODES.contains(mode)) {
      throw new IllegalArgumentException(opName + ": unsupported mode \"" + mode + "\"");
    }
  }

  /**
   * Affine quantization ({@code mlx_quantize}): packs {@code w} into a {@code UINT32}-encoded
   * weight plus per-group {@code scales}/{@code biases} for reconstruction (see {@link
   * #dequantize}). Under {@code mode="affine"} (the only mode {@code QuantizedLinear} uses), the
   * result is native's own three-array order: {@code [w_q, scales, biases]} -- but the result
   * length is {@code mode}-dependent, not a fixed 3: {@code mode="mxfp4"} returns only {@code [w_q,
   * scales]}, no {@code biases} (confirmed empirically). A caller using a non-affine {@code mode}
   * must check {@code result.length} rather than assume index 2 exists. {@code groupSize}/{@code
   * bits} are Java {@code null} for "let native pick its own default" (see this class's Findings
   * reference in {@link NativeOps#optInt} -- {@code QuantizedLinear} never uses this path, since a
   * persisted layer must remember its own exact values). {@code globalScale} is a Java {@code null}
   * for "none" -- a legitimate call, not an error, exactly like {@link MLXFast#rmsNorm}'s {@code
   * weight}. Non-null: {@code w}, {@code mode} -- {@code checkMode} already gives a named error for
   * a null {@code mode}; {@code w} is checked explicitly too, rather than left to fail as a bare,
   * unnamed {@code NullPointerException} out of {@code w.handle()} the way it did before this
   * class's own null-check enforcement was made consistent across every documented non-null array
   * operand (this method's, {@link #dequantize}'s, and {@link #quantizedMatmul}'s).
   *
   * <p>The result's scope is resolved via {@link NativeOps#scopeOf} over <em>both</em> array
   * operands, {@code w} and {@code globalScale} -- not just {@code w.scope()} the way {@link
   * MLXShape#split}'s {@code vectorOutOp} call does, because {@code split} has no nullable operand
   * and this op does (req/plans/phase4-m4-plan.md Global Constraint 7). A non-null {@code
   * globalScale} is real API surface here for correctness, but this codebase's Metal backend
   * rejects it unconditionally (Findings section) -- no test in this codebase can construct a legal
   * non-null {@code globalScale}, so none does; {@link #dequantize}'s cross-scope test uses {@code
   * biases} instead to exercise the identical rule.
   */
  public static MLXArray[] quantize(
      MLXArray w, Integer groupSize, Integer bits, String mode, MLXArray globalScale) {
    checkMode("quantize", mode);
    Objects.requireNonNull(w, "quantize: w must not be null");
    MLXScope target = NativeOps.scopeOf("quantize", w, globalScale);
    MemorySegment modeStr = NativeOps.cstr(mode);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment gs = NativeOps.optInt(tmp, groupSize);
      MemorySegment b = NativeOps.optInt(tmp, bits);
      MemorySegment gsHandle = NativeOps.nullableHandle(globalScale, tmp);
      return NativeOps.vectorOutOp(
          "quantize",
          target,
          (vec, stream) -> mlx_h.mlx_quantize(vec, w.handle(), gs, b, modeStr, gsHandle, stream));
    }
  }

  /**
   * Reconstructs a float array from {@code w}/{@code scales}/{@code biases} (the output of {@link
   * #quantize}, or an equivalent triple loaded from a checkpoint). {@code biases}/{@code
   * globalScale} are each a Java {@code null} for "none" (mlx-c's own nullable-array convention,
   * same as {@link #quantize}'s {@code globalScale}) -- but, like {@link #quantize}'s {@code
   * globalScale}, a non-null {@code globalScale} here is also unconditionally rejected on this
   * codebase's Metal backend (confirmed empirically, req/plans/phase4-m4-plan.md Findings): this
   * method's own {@code globalScale} is just as untestable in this codebase as {@code quantize}'s
   * is, not an ordinary nullable operand with no caveat. {@code dtype} is a Java {@code null} for
   * "let native pick its own default" -- confirmed empirically to follow {@code scales}' own dtype,
   * not unconditionally FLOAT32 (req/plans/phase4-m4-plan.md Findings): {@code scales} is FLOAT32
   * for a caller who quantized an unmodified FLOAT32 weight, but {@link #quantize} on a weight
   * already {@code astype}'d to FLOAT16 produces FLOAT16 {@code scales}, and an absent {@code
   * dtype} then dequantizes to FLOAT16, silently. Non-null: {@code w}, {@code scales}, {@code mode}
   * -- each checked explicitly (see {@link #quantize}'s own javadoc for why); {@code biases} is
   * deliberately NOT included in that check, since it stays legitimate, nullable API surface (see
   * below) rather than a documented-non-null operand this class's null-check enforcement applies
   * to.
   *
   * <p>Unlike {@code globalScale} (on either method), a non-null {@code biases} is real, legal
   * input on this codebase's Metal backend (Findings section) -- {@code MLXQuantTest}'s cross-scope
   * test for Global Constraint 7's {@code scopeOf}-over-a-nullable-operand rule uses this method's
   * {@code biases}, not either method's {@code globalScale}, for exactly that reason.
   *
   * <p><strong>{@code biases} stays nullable API surface, but a null value is not actually legal
   * under {@code mode="affine"}</strong> -- confirmed empirically (a finding beyond this plan's own
   * pre-work probes): native rejects it with {@code "[dequantize] Biases must be provided for
   * affine quantization"}, thrown from {@code mlx::core::dequantize} itself. The parameter is left
   * nullable regardless since a hypothetical non-affine mode may not require it and this codebase
   * cannot verify that either way (Global Constraint 5) -- but every caller using {@code
   * mode="affine"} (the only mode this codebase exercises) must pass a real {@code biases}.
   */
  public static MLXArray dequantize(
      MLXArray w,
      MLXArray scales,
      MLXArray biases,
      Integer groupSize,
      Integer bits,
      String mode,
      MLXArray globalScale,
      DType dtype) {
    checkMode("dequantize", mode);
    Objects.requireNonNull(w, "dequantize: w must not be null");
    Objects.requireNonNull(scales, "dequantize: scales must not be null");
    MLXScope target = NativeOps.scopeOf("dequantize", w, scales, biases, globalScale);
    MemorySegment modeStr = NativeOps.cstr(mode);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment gs = NativeOps.optInt(tmp, groupSize);
      MemorySegment b = NativeOps.optInt(tmp, bits);
      MemorySegment dt = NativeOps.optDtype(tmp, dtype);
      MemorySegment biasesHandle = NativeOps.nullableHandle(biases, tmp);
      MemorySegment globalScaleHandle = NativeOps.nullableHandle(globalScale, tmp);
      MemorySegment res = mlx_h.mlx_array_new(target);
      NativeOps.checked(
          "dequantize",
          () ->
              mlx_h.mlx_dequantize(
                  res,
                  w.handle(),
                  scales.handle(),
                  biasesHandle,
                  gs,
                  b,
                  modeStr,
                  globalScaleHandle,
                  dt,
                  NativeOps.DEFAULT_STREAM));
      return new MLXArray(target, res);
    }
  }

  /**
   * Fused quantized matrix multiply ({@code mlx_quantized_matmul}): {@code x @ w^T} when {@code
   * transpose=true}, without ever materializing a dequantized {@code w} -- the operation {@code
   * QuantizedLinear}, in the neural-network package, is built on (same "package named in prose, not
   * a qualified {@code @link}" convention {@link MLXGrad}'s own javadoc already uses for {@code
   * ModuleGrad}, for the same reason: spec Verification #6a's grep must never match a core class's
   * source). {@code transpose=true} is the convention that matches {@code Linear}'s own checkpoint
   * layout: {@code w}/{@code scales}/{@code biases} describe the packed form of an {@code [out,
   * in]} weight (confirmed empirically, req/plans/phase4-m4-plan.md Findings). {@code biases} is a
   * Java {@code null} for "none", same nullable-array convention as {@link #dequantize} -- and,
   * like {@link #dequantize}'s {@code biases}, a null value is not actually legal under {@code
   * mode="affine"}: native rejects it with {@code "[quantized_matmul] Biases must be provided for
   * affine quantization"} (confirmed empirically against this method directly, not just inferred
   * from {@code dequantize}'s shared affine path). Non-null in practice under {@code "affine"}:
   * {@code x}, {@code w}, {@code scales}, {@code biases}, {@code mode} -- but only {@code x},
   * {@code w}, {@code scales}, {@code mode} are checked explicitly (the target overload below is
   * where the check runs, since this overload immediately delegates to it): {@code biases} is
   * deliberately excluded, since it stays legitimate, nullable API surface (a hypothetical
   * non-affine mode may not need it) rather than a documented-non-null operand, the same
   * distinction {@link #dequantize}'s own javadoc draws.
   *
   * <p>Allocates into {@link NativeOps#scopeOf}'s innermost-of-the-four-operands rule, which only
   * lands in the step scope for the layout {@code Global Constraint 6} assumes: {@code x} in the
   * step scope, {@code w}/{@code scales}/{@code biases} in an ancestor (model) scope. On the
   * inverted layout -- a model built inside a scope that is itself a <em>descendant</em> of {@code
   * x}'s own, e.g. {@code x} from the root and the model from a child scope -- the innermost rule
   * instead picks the model scope, leaking one array into it per call (confirmed empirically,
   * req/plans/phase4-m4-plan.md's Amendment). Use the {@link #quantizedMatmul(MLXArray, MLXArray,
   * MLXArray, MLXArray, boolean, Integer, Integer, String, MLXScope)} overload with an explicit
   * {@code target} (typically {@code x.scope()}, mirroring {@code Linear.forward}'s {@code
   * transpose(W, x.scope())}) to avoid that leak under the inverted layout.
   */
  public static MLXArray quantizedMatmul(
      MLXArray x,
      MLXArray w,
      MLXArray scales,
      MLXArray biases,
      boolean transpose,
      Integer groupSize,
      Integer bits,
      String mode) {
    return quantizedMatmul(
        x,
        w,
        scales,
        biases,
        transpose,
        groupSize,
        bits,
        mode,
        NativeOps.scopeOf("quantizedMatmul", x, w, scales, biases));
  }

  /**
   * Same fused quantized matmul as {@link #quantizedMatmul(MLXArray, MLXArray, MLXArray, MLXArray,
   * boolean, Integer, Integer, String)}, but allocates the result into {@code target} instead of
   * the innermost scope among the four operands -- e.g. {@code QuantizedLinear.forward} passing
   * {@code x.scope()} so the result lands in the step scope even under the inverted layout the
   * no-target overload's javadoc documents. {@code target} must be related to every operand --
   * checked via {@link MLXScope#innermost} against {@link NativeOps#scopeOf}'s own result, the same
   * invariant {@link NativeOps#unaryOp(String, MLXArray, MLXScope, NativeOps.UnaryOp)}'s target
   * overload documents. {@code x}/{@code w}/{@code scales} are checked explicitly here (this is the
   * one call site both overloads funnel through); {@code biases} is not -- see the no-target
   * overload's own javadoc for why.
   */
  public static MLXArray quantizedMatmul(
      MLXArray x,
      MLXArray w,
      MLXArray scales,
      MLXArray biases,
      boolean transpose,
      Integer groupSize,
      Integer bits,
      String mode,
      MLXScope target) {
    checkMode("quantizedMatmul", mode);
    Objects.requireNonNull(x, "quantizedMatmul: x must not be null");
    Objects.requireNonNull(w, "quantizedMatmul: w must not be null");
    Objects.requireNonNull(scales, "quantizedMatmul: scales must not be null");
    MLXScope.innermost(NativeOps.scopeOf("quantizedMatmul", x, w, scales, biases), target);
    MemorySegment modeStr = NativeOps.cstr(mode);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment gs = NativeOps.optInt(tmp, groupSize);
      MemorySegment b = NativeOps.optInt(tmp, bits);
      MemorySegment biasesHandle = NativeOps.nullableHandle(biases, tmp);
      MemorySegment res = mlx_h.mlx_array_new(target);
      NativeOps.checked(
          "quantizedMatmul",
          () ->
              mlx_h.mlx_quantized_matmul(
                  res,
                  x.handle(),
                  w.handle(),
                  scales.handle(),
                  biasesHandle,
                  transpose,
                  gs,
                  b,
                  modeStr,
                  NativeOps.DEFAULT_STREAM));
      return new MLXArray(target, res);
    }
  }
}
