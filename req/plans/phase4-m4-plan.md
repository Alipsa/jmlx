# jmlx Phase 4 — M4 implementation plan (`QuantizedLinear`)

**Spec:** `req/phase4-plan.md` §8 (the three quantization entry points and their parameter-shape
table), §3 (the two remaining generic op-body helpers this plan builds: `optInt`, `optDtype` --
`vectorOutOp` already exists, built in M3, and needs no change), §2 (`scopeOf`/`innermost`, and the
rule that every multi-operand op -- including nullable operands -- resolves its target through it),
§5 (`Module`/layer conventions: `param`/`child`, the checkpoint-layout contract `Linear` already
established), Decision 8 (test goldens; composition-identity fallback when a direct golden would
encode mlx's internal packing layout), §9 (Documentation -- folded into this plan; see the note
below), Testing approach (`dequantize(quantize(w)) ≈ w` as "the only tractable `QuantizedLinear`
assertion").

Branch: `worktree-phase4-m4`, off `main` at `59c9afe` (PR #10 merged, M3 done). Native bootstrap
staged in this worktree (`native/install/lib/mlx.metallib` exists) -- `@EnabledIfNativeAvailable`
tests run for real, against real Apple Silicon GPU hardware.

**§9 (Documentation) is folded into this plan, not deferred again.** `req/phase4-plan.md`'s own
Status table note on §9 reads "stays `NOT STARTED` until M4 also lands" -- M1/M2/M3 each deferred it
for that reason; M4 is the merge point the note is waiting for, so this plan's Task 3 does it rather
than opening a fifth "not yet" entry. §10 (CI, self-hosted runner) is a separate, already-tracked
follow-up with no dependency on M4 and stays out of scope here, same as every prior plan.

## Findings from this plan's pre-work (Verification item 0d, resolved)

Per `req/phase4-plan.md`'s Verification item 0d, `mlx_quantize`'s exact output shapes/dtypes and
`mlx_quantized_matmul`'s `transpose` semantics are not fully pinned down by the header alone. Probed
with a scratch test class (`ScratchQuantizeProbe`, in `se.alipsa.jmlx.core` for package-private access
to `NativeOps`/`nullableHandle`/`cstr`), compiled and run against this worktree's real bindings, then
deleted -- same precedent as M2/M3's probes. All findings below are measured output, not derived.

* **`mlx_quantize`'s three outputs, `[1,64]` FLOAT32 input, `group_size=32`, `bits=4`,
  `mode="affine"`, `global_scale` absent.** Status `0`, exactly 3 outputs (`vectorOutOp`'s existing
  shape, unchanged from `split`'s):
  * `w_q`: shape `[1, 8]`, dtype `3` == `mlx_h.MLX_UINT32()` -- confirms the packing formula
    `packedCols = cols * bits / 32` (`64 * 4 / 32 = 8`), and confirms `DType.UINT32`'s existing native
    value needs no change to read this output correctly.
  * `scales`, `biases`: each shape `[1, 2]`, dtype `10` == `mlx_h.MLX_FLOAT32()` -- confirms
    `numGroups = cols / groupSize` (`64 / 32 = 2`), one scale and one bias term per group.
* **`packedCols = cols * bits / 32` only holds for power-of-2 `bits` (2, 4, 8).** Confirmed by
  reading `mlx/backend/common/quantized.h`'s `get_pack_factor`/`get_bytes_per_pack` (the wheel ships
  no `quantized.cpp`, so this is the packing contract, not an inference): `bits` of 3 or 5 pack 8
  elements into a non-power-of-2-sized unit (`get_bytes_per_pack` returns `5` for `bits=5`, not
  `5*8/32`), and `bits=6` uses a 4-element pack factor -- neither fits the simple formula this plan
  otherwise relies on. Only `bits` values with `32 % bits == 0` (2, 4, 8 -- not 3, 5, 6) are safe for
  any assertion built on `packedCols`; `QuantizedLinear`'s constructor (Task 2) scopes its
  groupSize/bits consistency check to that condition rather than assuming the formula universally.

  **Amendment (post-merge review): this bullet's conclusion is empirically false.** Derived from
  reading `get_pack_factor`/`get_bytes_per_pack`'s *byte-level* pack unit, not from measuring
  `mlx_quantize`'s actual output shape -- the byte-level unevenness those functions describe never
  surfaces at the *word* granularity `packedCols` is computed in, because `group_size`'s own legal
  set (`{32, 64, 128}`) forces `cols` (`= scales.shape()[1] * group_size`) to always be a multiple of
  32. Probed directly: `cols` in `{64, 96, 128, 160, 224}` x `bits` in `{3, 5, 6}` all produced
  `w_q.shape()[-1] == cols * bits / 32` exactly, every time. `QuantizedLinear`'s `32 % bits == 0`
  gate was itself wrong, not merely conservative -- it silently skipped the packed-column
  consistency check for half the legal `bits` set, reintroducing the deferred-opaque-native-error
  failure mode the check exists to prevent (a mismatched `groupSize` at `bits=3` cleared
  construction and only failed at first `forward()`, with `[quantized_matmul] The shapes of the
  weight and scales are incompatible...`). Fixed: the check now runs unconditionally for all six
  legal `bits` values.
* **`global_scale` is unconditionally rejected on the Metal backend, for both `quantize` and
  `dequantize`, in both modes tried.** Probed directly (`ScratchQuantizeProbe2`, same precedent): a
  non-null `global_scale` with `mode="affine"` on the same `[1,64]` input returns status `1` with
  native error `"[quantize] Global scale is not supported on the Metal backend."`; the identical
  message (`"[dequantize] Global scale is not supported on the Metal backend."`) is returned by
  `mlx_dequantize` under the same conditions -- **`dequantize`'s `globalScale` is equally dead on this
  codebase's backend, not just `quantize`'s.** The message strings themselves were confirmed against
  the shipped binary (`strings native/install/lib/libmlx.dylib | grep -i "global scale"`), which is
  also where the check actually lives -- the file:line mlx-c's own exception-catching wrapper reports
  alongside the message (e.g. `mlx-c/mlx/c/ops.cpp:2580`) is where mlx-c re-throws a C++ exception as a
  C status, not where the guard itself is; this repo has no `libmlx` source to cite instead. The binary
  also contains a *second*, mode-specific string (`"Global scale is only supported for 'nvfp4'"`);
  probed with `mode="nvfp4"` (which that string suggests should be allowed) and got the identical
  Metal-backend message regardless, so **this codebase's evidence does not establish which check runs
  first in general** -- only that on this Metal-only codebase, every mode tried is rejected the same
  way. Since this codebase only ever runs against the Metal backend (macOS/Apple Silicon only, per
  `req/initial-plan.md`), **no test here can ever exercise a non-null `globalScale` on either method.**
  Both `MLXQuant.quantize` and `MLXQuant.dequantize` still resolve their target via `scopeOf(..., 
  globalScale)` for API correctness (a future non-Metal backend, or a caller who passes one anyway,
  must still get the right scope before hitting the native error) -- but the cross-scope test proving
  that resolution empirically needs a different nullable operand. `dequantize`'s `biases` is the
  substitute: confirmed by the same probe run to accept a non-null value under `mode="affine"` with no
  restriction. Step 1f's test list uses `biases`, not `globalScale`, for this reason.
* **`mlx_dequantize` with `dtype` absent (`has_value=false`) returns FLOAT32**, matching `scales`'
  own dtype -- output shape `[1, 64]`, round-trips the original data with **max absolute error
  ≈0.1067** against the probe's `w[i] = i * 0.1f - 3.2f` (`i` in `0..63`, range `[-3.2 .. 3.1]`)
  input (16 representable levels per 32-element group at `bits=4`, consistent with a ≈0.4/16 ≈ 0.107
  step size for that group's local range). **`MLXQuantTest`'s
  `dequantizeRoundTripsWithinQuantizationError` uses this exact fixture** (not a different,
  unstated one) so `EPS_ROUNDTRIP = 0.15f` is a verified bound against the row it is actually
  checked against, not just "loosely above" a number measured on a different input. (A second
  fixture, `QuantizedLinearTest`'s own `(i % 7 - 3) * 0.3f`, was also probed for comparison: max abs
  error ≈0.1125, still safely under `EPS_ROUNDTRIP` -- either fixture would have worked, but
  `MLXQuantTest` commits to the one already named in this section rather than introducing a third.)
* **`mlx_quantized_matmul(x, w_q, scales, biases, transpose=true, ...)` treats `w_q` as the packed
  form of a checkpoint-layout `[out, in]` weight and computes `x @ w^T`** -- exactly `Linear`'s own
  weight convention, with the transpose fused into the native call rather than a separate op.
  Confirmed: `x` = the same `[1,64]` row used to build `w_q` (so the expected unquantized result is
  `dot(x,x) = 218.56`); actual quantized result `219.67467`, a ≈0.5% relative difference consistent
  with 4-bit quantization noise on `w_q` alone (`x` itself is unquantized FLOAT32 in this call).
  Output shape `[1, 1]`, matching `x`'s batch dim and `w_q`'s logical `out=1`.
* **The composition-identity comparisons this plan's tests rely on (`quantizedMatmul` vs.
  `matmul(x, transpose(dequantize(...)))`, computed from the *same* quantized components) agree to
  within float32 rounding noise, not quantization noise.** Probed directly, two fixtures:
  * `[1,64]` `w`/`x` (the `(i % 7 - 3) * 0.3f` fixture, `group_size=32`/`bits=4`/`mode="affine"`):
    `quantizedMatmul` result `22.275002` vs. the composed `matmul(x, transpose(dequantize(...)))`
    result `22.275003` -- max abs diff `1.9e-6`. `MLXQuantTest`'s
    `quantizedMatmulMatchesDequantizedFloatMatmul` uses `EPS_EXACT = 1e-5f` for this row (not a
    fresh, looser constant): the diff is five orders of magnitude below the quantization error this
    row is deliberately isolating itself from, so the facade's usual exact-op tolerance already
    covers it.
  * `[2,64]`/`[1,64]` (`QuantizedLinearTest`'s actual fixture: `w` = `(i % 7 - 3) * 0.3f`, `x` =
    `(i % 5 - 2) * 0.2f`, `bias = [0.5, -0.5]`, same `group_size`/`bits`/`mode`): `y1 =
    quantizedMatmul(...) + bias` = `[0.11749995, -0.16250014]`; `y2 = matmul(x,
    transpose(dequantize(...))) + bias` = `[0.11750007, -0.16249979]` -- max abs diff `3.6e-7`. The
    Findings section's earlier `dot(x,x) = 218.56` vs. `219.67467` comparison (a *different* row,
    quantized-vs-original-unquantized) is not this test's shape and its ≈0.5%-relative magnitude does
    not transfer here -- `QuantizedLinearTest`'s `EPS = 1e-3f` is confirmed generous, not tight, for
    the composition identity it actually checks.
  * `[64,64]` `w`/`x` for `transpose=false`, fixture `w[i] = (i % 7 - 3) * 0.3f` for `i` in
    `0..(64*64-1)`, `x[i] = (i % 5 - 2) * 0.2f` for `i` in `0..63` (`w` laid out as `[in, out]`, both
    `=64`, so `transpose=false`'s `x @ w` is legal directly and `out=64` is itself divisible by
    `group_size=32`): `quantizedMatmul(x, quantize(w)..., transpose=false)` vs. `matmul(x,
    dequantize(quantize(w)))` (no `transpose(...)` on the right-hand side) -- max abs diff `≈2.4e-7`,
    same order of magnitude as the `transpose=true` rows above, so `EPS_EXACT` applies here too.
    **`w`'s own transpose does not work as this fixture** (an earlier draft of this plan proposed it):
    probed directly and confirmed `mlx_quantize` rejects a `[64,1]` last axis outright (`mlx_quantize`
    groups along the *last* axis, and `1` is not divisible by `group_size=32`) -- the actual native
    error, paraphrased from the runtime-interpolated message (not quoted verbatim; the source string
    is assembled from fragments plus interpolated values): "the last dimension of the matrix needs to
    be divisible by the quantization group size 32, however the provided matrix has shape (64, 1)".
* **`mlx_dequantize`'s `dtype` parameter accepts `DType.FLOAT16` (`has_value=true`, a non-default
  value), producing a genuinely different result than the absent-`dtype` FLOAT32 default.** Probed
  directly on the `[1,64]` `(i % 7 - 3) * 0.3f` fixture -- **deliberately not** Step 1f's declared
  default fixture (`w[i] = i * 0.1f - 3.2f`); the values below only reproduce under this one, so
  `MLXQuantTest`'s `optDtype`-`has_value=true` row states this fixture explicitly rather than
  inheriting the file's default, using the "unless a row is specifically testing a different value"
  escape hatch Step 1f's preamble already allows: `dequantize(..., dtype=FLOAT16)` returns status `0`,
  `result.dtype()` native value `9` (== `mlx_h.MLX_FLOAT16()`), first few values `-0.78759766,
  -0.5625, -0.33740234` -- distinct from the FLOAT32 default this same fixture produces elsewhere in
  this section. (The row's surviving assertion is `dtype()` alone, which is fixture-independent --
  but the fixture is still named exactly, per this document's own anti-drift rule for every other
  probed fixture.)
* **`groupSize`/`bits`'s exact legal sets, confirmed against the shipped binary**
  (`strings native/install/lib/libmlx.dylib`, not inferred): `bits` in `{2, 3, 4, 5, 6, 8}`
  ("`... is not supported. The supported bits are 2, 3, 4, 5, 6 and 8.`"); `group_size` in
  `{32, 64, 128}` ("`... is not supported. The supported group sizes are 32, 64, and 128.`" and,
  separately, "`Quantization group size must be 32, 64 or 128.`"). `QuantizedLinear`'s constructor
  (Task 2) enforces both sets directly, not merely `> 0`: an out-of-set value (e.g. `bits=1`/`16`/`32`,
  `groupSize=16`/`48`/`100`) would otherwise clear every other constructor check and fail as an opaque
  native error at first `forward` call, the same failure mode `weight.dtype() != UINT32` exists to
  prevent for a plain float weight. This also makes the `32 % bits == 0` power-of-2 gate's story exact
  rather than approximate: of this now-enforced set `{2, 3, 4, 5, 6, 8}`, the power-of-2 subset is
  exactly `{2, 4, 8}` -- before this set was enforced, the gate silently also admitted `1`/`16`/`32`,
  values this section never otherwise discusses.
* **`group_size`/`bits` both absent (`has_value=false`) does not error -- native applies defaults.**
  Same `[1,64]` input: `w_q` shape is still `[1, 8]` (bits default `4`, unchanged packing), but
  `scales` shape is `[1, 1]` (**one** group covering the whole row) where the explicit-`32` run
  produced `[1, 2]` -- so the native default `group_size` is `64` (the full row), not `32`. This path
  has no `QuantizedLinear` caller (the layer always supplies explicit values, per Global Constraint 6
  below) but is exercised directly by `MLXQuantTest` for `optInt`'s absent-value branch.
* **`mlx_gather_qmm`, `mlx_qqmm` and `mlx_gather_mm` also exist in this `ops.h`** (a gathered/MoE
  quantized-matmul family) but are not named anywhere in spec §8, which lists exactly three entry
  points matching `project-outline.md`'s single `QuantizedLinear` deliverable. Out of scope --
  no consumer in this codebase needs gathered/indexed quantized matmul.
* **Four `QuantizationMode` values exist upstream** (`wheel/mlx/include/mlx/primitives.h:155`:
  `Affine, Mxfp4, Mxfp8, Nvfp4`), all reachable through the same `const char* mode` parameter this
  plan's ops expose. Only `"affine"` is exercised by any test in this plan or used by
  `QuantizedLinear` -- consistent with `req/phase4-plan.md`'s own Research findings ("`mlx_quantize`
  with `mode="affine"`" is the only mode that document's Decisions ever reference). The other three
  are unvalidated by this codebase; see Deliberately not covered.

## Global Constraints

1. **Package boundaries are unchanged from M1-M3.** `optInt`/`optDtype` (`NativeOps`) and `quantize`/
   `dequantize`/`quantizedMatmul` (`MLXQuant`) live in `se.alipsa.jmlx.core`; `QuantizedLinear` lives
   in `se.alipsa.jmlx.nn` and depends only on the public `core` surface. `grep -rn
   'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` must stay empty (spec
   Verification #6a).
2. **`NativeOps` keeps its zero-sibling-dependency invariant.** `optInt`/`optDtype` are built directly
   from the generated `mlx_optional_int_`/`mlx_optional_dtype_` struct accessors, exactly mirroring
   `optFloat`'s existing shape -- no call into `MLX`, `MLXQuant`, or any other sibling.
3. **Only the two remaining generic helpers are added: `optInt`, `optDtype`.** `vectorOutOp` (§3's
   sixth helper) already exists, built in M3 for `split`, and needs zero changes: `mlx_quantize`'s
   `(mlx_vector_array* res, ...)` shape is identical, so `MLXQuant.quantize` calls the existing helper
   with a new lambda body, the same way `MLXShape.split` already does.
4. **`mode` is a generic `String` parameter on all three `MLXQuant` methods**, matching spec §8's
   named signatures (`quantize(w, gs, bits, mode)` etc.). Unlike `MLXFast.scaledDotProductAttention`,
   which *hides* its own native `mask_mode` string behind `causal`/`maskArr` booleans mapped onto
   three `private static final MASK_MODE_*` constants (`MLXFast.java:109-111`) -- exactly the
   opposite of what this constraint originally claimed as precedent -- `MLXQuant` exposes `mode`
   directly, because spec §8 names it as a real parameter with four legal upstream values (Findings
   section) that this facade has no natural boolean to collapse them into. Each of the three methods
   validates `mode` against the closed set `{"affine", "mxfp4", "mxfp8", "nvfp4"}` (`throw new
   IllegalArgumentException` naming the method and the bad value) before calling
   `NativeOps.cstr(mode)`: `cstr`'s own javadoc documents an intern-cache contract sized for a
   bounded set of `private static final`-held literals, and a raw pass-through of caller-supplied
   `mode` would violate that (a typo'd or dynamically-built string still allocates one
   `FACADE_ARENA` segment per distinct value it has ever seen, unbounded) and NPE out of
   `computeIfAbsent` on a null `mode` with no named error. **This second failure mode is not fixed by
   `MODES.contains(mode)` alone** -- `Set.of(...)`'s returned set throws its own unnamed `NullPointerException`
   out of `contains` on a `null` argument (confirmed: `Set.of("a").contains(null)` throws), so
   `checkMode`'s guard needs an explicit `mode == null` branch checked before `MODES.contains(mode)`,
   not just the membership check, to actually name the error this constraint describes.
   `QuantizedLinear` itself always calls with
   the literal `"affine"`: the only mode with any documented rationale in this repo (Research
   findings: "`mlx_fast_affine_quantize` does not exist ... filled by `mlx_quantize` with
   `mode="affine"`"). `Mxfp4`/`Mxfp8`/`Nvfp4` remain reachable through `MLXQuant` for a future caller
   (validated, still sent to native as a real string) but are not exercised by any test here -- a
   genuinely new upstream mode would need this validation set updated too, an accepted trade-off
   given `cstr`'s closed-set assumption.
5. **`groupSize`/`bits` are boxed `Integer`, nullable for "absent"** on all three `MLXQuant` methods --
   mirroring `MLXFast.rope`'s existing `Float base` parameter for the same by-value-optional-struct
   shape. `QuantizedLinear`'s own constructor takes plain non-null `int groupSize, int bits`: a
   persisted layer must remember the exact values it was built with (native's own defaults, found
   above, are a moving target this layer must not silently depend on), so the "absent" path is real
   API surface on `MLXQuant` (tested directly by `MLXQuantTest`) but has no `QuantizedLinear` caller.
   **Every method's javadoc states which of its parameters may be `null`**, since `scopeOf` silently
   tolerates a `null` array operand (it just skips it when resolving the target) and a caller passing
   a `null` for a genuinely required operand (e.g. `dequantize`'s `scales`) would otherwise NPE
   opaquely on `.handle()`/`.scope()` with no explanatory message: `quantize`'s nullable operand is
   `globalScale`; `dequantize`'s are `biases`, `globalScale`, `dtype` (plus `groupSize`/`bits`, all
   three methods); `quantizedMatmul`'s is `biases` (plus `groupSize`/`bits`). `w`/`scales`/`x` and
   `mode` are never `null` in any of the three methods. Both methods' `globalScale` is nullable API
   surface that this codebase can never exercise non-null on either one (Findings section: rejected
   unconditionally on the Metal backend) -- `dequantize`'s is not an exception to that just because
   its javadoc used to mention only `quantize`'s.
6. **`quantizedMatmul` needs no explicit-target overload the way `Linear.forward`'s `transpose(W)`
   did (§2's fifth sub-hazard).** That hazard is specifically about a *single-operand* op on a
   parameter -- `scopeOf(op, a)` degenerates to `a.scope()`, which is the model scope when `a` is a
   weight. `quantizedMatmul(x, w, scales, biases, ...)` always has `x` (an activation, never a
   parameter) as a real second operand, so `scopeOf` resolves to the innermost of `x`'s scope and the
   parameter scopes -- correctly the step scope -- with no special-casing. Stating this now so a later
   reader does not "fix" a leak that was never there by adding an overload with no consumer.
7. **Every multi-operand `MLXQuant` op resolves its target via `NativeOps.scopeOf(...)` over every
   array operand, including nullable ones** -- `quantize`'s `(w, globalScale)`, `dequantize`'s `(w,
   scales, biases, globalScale)`, `quantizedMatmul`'s `(x, w, scales, biases)`. `quantize`'s case is
   the one worth naming explicitly: `MLXShape.split` (its `vectorOutOp` precedent) passes `a.scope()`
   directly as the target because `split` has no nullable operand, but `quantize` does (`globalScale`)
   -- so its target must be `scopeOf("quantize", w, globalScale)`, not `w.scope()`, or a `globalScale`
   from a strictly inner scope than `w` would be allocated into the wrong (outer) scope without ever
   running its own `checkAccess()`. **This rule is written into `quantize` (and `dequantize`, for its
   own `globalScale`) for correctness, but neither method's own cross-scope test can be the one that
   proves it:** the Findings section confirms a non-null `globalScale` errors on this codebase's only
   target backend (Metal) in every mode actually tried (`"affine"` and `"nvfp4"`), so no test here can
   construct a legal non-null `globalScale` to put in a child scope, on either method. `MLXQuantTest`'s
   dedicated cross-scope row (Task 1, Step 1f) instead exercises this same `scopeOf`-over-a-nullable-
   operand rule through `dequantize`'s `biases` -- confirmed by the
   same probe to have no such restriction -- which discriminates the identical hazard (a `w.scope()`-
   only target silently passing where `scopeOf(w, biases)` would not) without requiring an
   operand this backend can never legally construct.
8. **Style:** run `./gradlew spotlessApply` before finishing, then `./gradlew build` (Spotless,
   Checkstyle, full test suite including native tests and the forked `loaderGuardTest`) must succeed.
   Report actual test counts from the build output.
9. **Testing style, matching every existing test file:** `@EnabledIfNativeAvailable` on the test
   class; `try (MLXScope scope = new MLXScope())`; `assertArrayEquals(expected, actual.toFloatArray(),
   EPS)`. Quantization error is not float rounding error, so **round-trip** rows (`dequantize(quantize(
   w)) ≈ w`) use `EPS_ROUNDTRIP = 0.15f`, derived from the Findings section's measured error, not the
   facade's usual `1e-5f`/`1e-3f`. **Composition-identity** rows (`quantizedMatmul` vs. `matmul(...,
   transpose(dequantize(...)))`, or `QuantizedLinear.forward` vs. `Linear.forward` on the dequantized
   weight) compare two computations over the *same* quantized components rather than across the lossy
   quantize/dequantize transformation itself -- verified directly (Findings section: measured diffs
   `≈1.9e-6` and `≈3.6e-7`) to be within float32 rounding noise, so these rows use the facade's usual
   `EPS_EXACT = 1e-5f`/`1e-3f` rather than a freshly loosened constant. The two row shapes are easy to
   conflate; keeping them named separately here is why they end up with different EPS constants below.
10. **`MLXQuant` gets a new, dedicated test file (`MLXQuantTest`), not new methods on
    `MLXNumericTest`/`MLXFastTest`.** Matches how `MLXFastTest` was created new when `MLXFast` was
    born at M1, not folded into `MLXNumericTest` -- a new op-family class gets a new test file; only
    *additions* to an existing class's op family go into that class's existing test file (M3's Global
    Constraint 7).

## Native surface this plan uses (confirmed against the generated bindings and the probe above)

| Java call | C signature | Notes |
|---|---|---|
| `mlx_h.mlx_quantize(res, w, group_size, bits, mode, global_scale, s)` | `int mlx_quantize(mlx_vector_array* res, const mlx_array w, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_array global_scale, const mlx_stream s)` (`ops.h`) | `res` is the existing `vectorOutOp` shape (built in M3 for `mlx_split`, unchanged here). `group_size`/`bits` are by-value `optInt` structs (new helper). `global_scale` is a by-value nullable struct via the existing `nullableHandle`. Output shapes/dtypes for a `[1,64]`/`32`/`4` call: see Findings. |
| `mlx_h.mlx_dequantize(res, w, scales, biases, group_size, bits, mode, global_scale, dtype, s)` | `int mlx_dequantize(mlx_array* res, const mlx_array w, const mlx_array scales, const mlx_array biases /* may be null */, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_array global_scale /* may be null */, mlx_optional_dtype dtype, const mlx_stream s)` (`ops.h`) | The densest single call in this plan: two by-value optional structs (`group_size`/`bits` via `optInt`, `dtype` via the new `optDtype`) plus two nullable-array structs (`biases`, `global_scale`) in one call. `dtype` absent -> FLOAT32 output (Findings). |
| `mlx_h.mlx_quantized_matmul(res, x, w, scales, biases, transpose, group_size, bits, mode, s)` | `int mlx_quantized_matmul(mlx_array* res, const mlx_array x, const mlx_array w, const mlx_array scales, const mlx_array biases /* may be null */, bool transpose, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_stream s)` (`ops.h`) | `transpose=true` is the `Linear`-compatible convention: `w`/`scales`/`biases` describe a `[out, in]`-layout original weight, result is `x @ w^T` (Findings). No array output beyond the single `res` -- a plain `scopeOf`-resolved hand-rolled body, same shape as `MLXFast.rmsNorm`. |
| `mlx_optional_dtype_` | `struct { mlx_dtype value; bool has_value; }`, 8 bytes -- confirmed by reading the generated binding directly; byte-identical layout to `mlx_optional_int_`/`mlx_optional_float_`. No native constructor function -- allocate and set both fields, `value` via the package-private `DType.nativeValue()`. |

## Task 1: `NativeOps.optInt`/`optDtype` + `MLXQuant` ops (`se.alipsa.jmlx.core`)

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/NativeOps.java`
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXQuant.java`
- New: `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXQuantTest.java`

### Step 1a: `NativeOps` — `optInt`

Add next to `optFloat` (`NativeOps.java:250-255`), same shape:

```java
/**
 * The native encoding of an {@code mlx_optional_int} by-value struct (req/phase4-plan.md §8's
 * Native surface table: 8 bytes, {@code {int value; bool has_value;}}, no native constructor
 * function -- same shape as {@link #optFloat}, for {@code mlx_quantize}/{@code mlx_dequantize}/
 * {@code mlx_quantized_matmul}'s {@code group_size}/{@code bits} parameters). {@code value == null}
 * encodes "absent" ({@code has_value=false}); native applies its own default in that case (confirmed
 * empirically, req/plans/phase4-m4-plan.md's Findings section) -- this facade does not duplicate it.
 */
static MemorySegment optInt(Arena tmp, Integer value) {
  MemorySegment seg = mlx_optional_int_.allocate(tmp);
  mlx_optional_int_.value(seg, value != null ? value : 0);
  mlx_optional_int_.has_value(seg, value != null);
  return seg;
}
```

### Step 1b: `NativeOps` — `optDtype`

```java
/**
 * The native encoding of an {@code mlx_optional_dtype} by-value struct -- same 8-byte {@code
 * {int value; bool has_value;}} shape as {@link #optInt}/{@link #optFloat} (confirmed by reading the
 * generated {@code mlx_optional_dtype_} binding directly: byte-identical layout, only the field's C
 * type name differs). {@code value == null} encodes "absent"; {@code mlx_dequantize} then produces
 * its own default dtype -- FLOAT32 in every case this facade can construct (confirmed empirically,
 * req/plans/phase4-m4-plan.md's Findings section: absent {@code dtype} matches {@code scales}' own
 * dtype, which this facade only ever produces as FLOAT32 via {@link MLXQuant#quantize}).
 */
static MemorySegment optDtype(Arena tmp, DType value) {
  MemorySegment seg = mlx_optional_dtype_.allocate(tmp);
  mlx_optional_dtype_.value(seg, value != null ? value.nativeValue() : 0);
  mlx_optional_dtype_.has_value(seg, value != null);
  return seg;
}
```

### Step 1c: `MLXQuant` — `quantize`

Replace the placeholder body (currently just a private constructor) with:

```java
public final class MLXQuant {

  private static final Set<String> MODES = Set.of("affine", "mxfp4", "mxfp8", "nvfp4");

  private MLXQuant() {}

  /**
   * Validates {@code mode} against the four upstream {@code QuantizationMode} values (Findings
   * section) before it ever reaches {@link NativeOps#cstr} -- {@code cstr}'s own javadoc documents an
   * intern-cache contract sized for a bounded, {@code private static final}-held set of literals; a
   * raw pass-through of this caller-supplied parameter would let a typo'd or dynamically-built string
   * grow {@code FACADE_ARENA} by one segment per distinct value ever seen, and NPE out of {@code
   * computeIfAbsent} with no named error on a {@code null} mode. {@code mode == null} is checked
   * explicitly, not left to {@code MODES.contains(null)} -- {@code Set.of(...)}'s returned set throws
   * {@code NullPointerException} out of {@code contains} on a {@code null} argument (confirmed:
   * {@code Set.of("a").contains(null)} throws), so a naive port of this guard would just move the
   * unnamed NPE from {@code cstr}'s cache to here instead of fixing it.
   */
  private static void checkMode(String opName, String mode) {
    if (mode == null || !MODES.contains(mode)) {
      throw new IllegalArgumentException(opName + ": unsupported mode \"" + mode + "\"");
    }
  }

  /**
   * Affine quantization ({@code mlx_quantize}): packs {@code w} into a {@code UINT32}-encoded weight
   * plus per-group {@code scales}/{@code biases} for reconstruction (see {@link #dequantize}). The
   * three-array result order is native's own: {@code [w_q, scales, biases]}. {@code groupSize}/{@code
   * bits} are Java {@code null} for "let native pick its own default" (see this class's Findings
   * reference in {@link NativeOps#optInt} -- {@code QuantizedLinear} never uses this path, since a
   * persisted layer must remember its own exact values). {@code globalScale} is a Java {@code null}
   * for "none" -- a legitimate call, not an error, exactly like {@link MLXFast#rmsNorm}'s {@code
   * weight}. Non-null: {@code w}, {@code mode}.
   *
   * <p>The result's scope is resolved via {@link NativeOps#scopeOf} over <em>both</em> array
   * operands, {@code w} and {@code globalScale} -- not just {@code w.scope()} the way {@link
   * MLXShape#split}'s {@code vectorOutOp} call does, because {@code split} has no nullable operand
   * and this op does (req/plans/phase4-m4-plan.md Global Constraint 7). A non-null {@code
   * globalScale} is real API surface here for correctness, but this codebase's Metal backend rejects
   * it unconditionally (Findings section) -- no test in this codebase can construct a legal non-null
   * {@code globalScale}, so none does; {@link #dequantize}'s cross-scope test uses {@code biases}
   * instead to exercise the identical rule.
   */
  public static MLXArray[] quantize(
      MLXArray w, Integer groupSize, Integer bits, String mode, MLXArray globalScale) {
    checkMode("quantize", mode);
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

  // dequantize, quantizedMatmul follow in 1d/1e
}
```

Add `import java.util.Set;` alongside the class's other imports.

**`NativeOps.cstr` is called with a caller-supplied `mode` here, not a `private static final` field
the way `MLXFast`'s `MASK_MODE_*` constants are** -- but, unlike an earlier draft of this plan
claimed, that is not what `scaledDotProductAttention` itself does: SDPA hides its own `mask_mode`
string behind `causal`/`maskArr` booleans mapped onto exactly those three constants
(`MLXFast.java:109-111`), the opposite pattern. `cstr`'s own javadoc documents an intern-cache
contract ("even an accidental per-call use never grows `FACADE_ARENA` beyond one segment per distinct
literal ever seen") sized for a bounded, `private static final`-held set of literals -- `MLXQuant`'s
three methods take `mode` as a parameter instead (Global Constraint 4) because spec §8 names it as
real API surface with four legal values, so `checkMode` reproduces that bound explicitly rather than
relying on `cstr`'s cache alone to keep it closed. `QuantizedLinear` (Task 2) passes the same Java
string literal `"affine"` on every call, so `cstr`'s cache still allocates that one segment exactly
once.

### Step 1d: `MLXQuant` — `dequantize`

```java
/**
 * Reconstructs a float array from {@code w}/{@code scales}/{@code biases} (the output of {@link
 * #quantize}, or an equivalent triple loaded from a checkpoint). {@code biases}/{@code globalScale}
 * are each a Java {@code null} for "none" (mlx-c's own nullable-array convention, same as {@link
 * #quantize}'s {@code globalScale}) -- but, like {@link #quantize}'s {@code globalScale}, a non-null
 * {@code globalScale} here is also unconditionally rejected on this codebase's Metal backend
 * (confirmed empirically, req/plans/phase4-m4-plan.md Findings): this method's own {@code
 * globalScale} is just as untestable in this codebase as {@code quantize}'s is, not an ordinary
 * nullable operand with no caveat. {@code dtype} is a Java {@code null} for "let native pick its
 * own default" -- confirmed empirically to be FLOAT32 for every input this facade can construct
 * (req/plans/phase4-m4-plan.md Findings). Non-null: {@code w}, {@code scales}, {@code mode}.
 *
 * <p>Unlike {@code globalScale} (on either method), a non-null {@code biases} is real, legal input
 * on this codebase's Metal backend (Findings section) -- {@code MLXQuantTest}'s cross-scope test for
 * Global Constraint 7's {@code scopeOf}-over-a-nullable-operand rule uses this method's {@code
 * biases}, not either method's {@code globalScale}, for exactly that reason.
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
```

### Step 1e: `MLXQuant` — `quantizedMatmul`

```java
/**
 * Fused quantized matrix multiply ({@code mlx_quantized_matmul}): {@code x @ w^T} when {@code
 * transpose=true}, without ever materializing a dequantized {@code w} -- the operation {@code
 * QuantizedLinear}, in the neural-network package, is built on (same "package named in prose, not a
 * qualified {@code @link}" convention {@link MLXGrad}'s own javadoc already uses for {@code
 * ModuleGrad}, for the same reason: spec Verification #6a's grep must never match a core class's
 * source). {@code transpose=true} is the convention that matches {@code Linear}'s own checkpoint
 * layout: {@code w}/{@code scales}/{@code biases} describe the packed form of an {@code [out, in]}
 * weight (confirmed empirically,
 * req/plans/phase4-m4-plan.md Findings). {@code biases} is a Java {@code null} for "none", same
 * nullable-array convention as {@link #dequantize}. Non-null: {@code x}, {@code w}, {@code scales},
 * {@code mode}.
 *
 * <p>Needs no explicit-target overload the way {@code Linear.forward}'s {@code transpose(W, x.scope())}
 * does (req/phase4-plan.md §2's fifth sub-hazard): {@code x} is always a genuine second array
 * operand here, never absent, so {@link NativeOps#scopeOf} alone already resolves to the step scope
 * whenever {@code x} is an activation and {@code w}/{@code scales}/{@code biases} are model-scope
 * parameters (Global Constraint 6).
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
  checkMode("quantizedMatmul", mode);
  MLXScope target = NativeOps.scopeOf("quantizedMatmul", x, w, scales, biases);
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
```

Note: `checkMode` (Step 1c) is a private helper on `MLXQuant` itself, called at the top of all three
methods above -- this note exists once here since the code block for `quantize` already shows its
definition; `dequantize`/`quantizedMatmul` above just call it.

### Step 1f: `MLXQuantTest.java` (new file)

`@EnabledIfNativeAvailable`; fixture `w[i] = i * 0.1f - 3.2f` for `i` in `0..63`, shape `[1,64]`
(pinned to the exact fixture the Findings section's round-trip probe measured, not a restated
formula that may drift from the measured number); `group_size=32`, `bits=4`, `mode="affine"`
throughout unless a row is specifically testing a different value. `EPS_ROUNDTRIP = 0.15f` (above the
Findings section's measured `≈0.1067` max absolute error for this exact fixture); `EPS_EXACT = 1e-5f`
for shape/dtype/status-level assertions and for both composition-identity rows, `transpose=true` and
`transpose=false` (Findings section: measured diffs `≈1.9e-6` and `≈2.4e-7` respectively, both orders
of magnitude inside this bound).

| Test | Proves |
|---|---|
| `quantizeProducesUint32PackedWeightAndFloat32ScalesAndBiases` | The exact shapes/dtypes from Findings, asserted as a regression pin: `[1,64]`/`32`/`4` -> `w_q` `[1,8]` UINT32, `scales`/`biases` each `[1,2]` FLOAT32 |
| `dequantizeRoundTripsWithinQuantizationError` | `dequantize(quantize(w)) ≈ w` within `EPS_ROUNDTRIP` -- the Testing-table row named in the top-level spec |
| `quantizedMatmulMatchesDequantizedFloatMatmul` | `quantizedMatmul(x, quantize(w)..., transpose=true)` vs. `matmul(x, transpose(dequantize(quantize(w))))` computed from the **same** quantized components -- a composition identity (Decision 8) tighter than comparing against the original unquantized `w`, since both sides now do identical math fused vs. unfused and should agree near-exactly (Findings section: measured `≈1.9e-6`), isolating "is `quantizedMatmul` correct" from "how lossy is 4-bit quantization" |
| `quantizedMatmulWithTransposeFalseMatchesDequantizedFloatMatmulWithoutTranspose` | `transpose`'s `false` branch, the one path no other row exercises. Uses the dedicated `[64,64]` fixture pinned in the Findings section (**not** `w`'s own transpose, which the Findings section explains fails at `mlx_quantize` itself before the matmul is ever reached): `quantizedMatmul(x, quantize(w)..., transpose=false)` vs. `matmul(x, dequantize(quantize(w)))` with no `transpose(...)` on the right-hand side -- Findings section: measured `≈2.4e-7`, so this row uses `EPS_EXACT` like the `transpose=true` row above it |
| `dequantizeWithExplicitFloat16DtypeProducesAFloat16Result` | `optDtype`'s `has_value=true` branch. Uses the `(i % 7 - 3) * 0.3f` fixture pinned in the Findings section for this row specifically -- **not** this file's declared default fixture (`w[i] = i * 0.1f - 3.2f`), which was never probed with `dtype=FLOAT16` and would not reproduce the recorded values; and **not** `dtype=DType.FLOAT32`, which is numerically identical to the absent-`dtype` default and so cannot discriminate `has_value=true` from an `optDtype` that always wrote `has_value=false`. Assert `dequantize(..., dtype=DType.FLOAT16).dtype() == DType.FLOAT16` -- the name says only what the row actually checks: the `dtype()` assertion alone already discriminates the two branches, no value comparison needed (there is no `assertArrayNotEquals` in JUnit, and an `assertFalse(Arrays.equals(...))` negation would only restate what `dtype()` already proves) |
| `quantizeWithNullGlobalScaleAndAbsentGroupSizeBitsSucceeds` | The nullable/absent-optional paths this facade supports but `QuantizedLinear` never exercises (Global Constraint 5) -- mirrors `MLXFastTest`'s null-`weight` `rmsNorm` coverage |
| **`dequantizeWithBiasesInAChildScopeOfWAllocatesIntoTheChild`, and the unrelated-scope negative** | Global Constraint 7's `scopeOf(w, ..., biases, ...)` rule, discriminating the case a `w.scope()`-only target would silently pass: build `w`/`scales`/`biases` from `quantize(w)` in a parent scope (`quantize`'s `vectorOutOp` allocates all three outputs into the *same* target scope, so `biases` cannot come out of that call already living in a different scope than `scales`) -- then build a **second** `MLXArray` from `biases.toFloatArray()` via `MLX.array(child, ...)`, targeting a child scope, and use that copy as the `biases` argument. Assert the result lands in the child; then repeat with the copy built in an unrelated root scope instead, assert `IllegalArgumentException`. Same two-row shape as M3's SDPA `q`/`k`/`v` scope test. Uses `dequantize`'s `biases`, not `quantize`'s `globalScale` (Findings section: a non-null `globalScale` is unconditionally rejected on this codebase's Metal backend, so `quantize` itself cannot be the test vehicle for this rule) |
| `dequantizeWithNullBiasesAndNullGlobalScaleSucceeds` | The two nullable-array parameters unique to `dequantize` are independently optional, not a package deal |
| `quantizeRejectsAnUnsupportedMode`, `dequantizeRejectsAnUnsupportedMode`, `quantizedMatmulRejectsAnUnsupportedMode` | `checkMode`'s guard on each of the three methods, e.g. `mode="int4"` -> `IllegalArgumentException` naming the method and the bad value |
| `quantizeRejectsANullMode`, `dequantizeRejectsANullMode`, `quantizedMatmulRejectsANullMode` | `checkMode`'s explicit `mode == null` branch, on all three methods -- the one this guard exists to fix (`MODES.contains(null)` alone throws an unnamed `NullPointerException`; asserting `IllegalArgumentException` here is the regression pin for that) |

## Task 2: `QuantizedLinear` (`se.alipsa.jmlx.nn`)

**Files:**
- New: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/QuantizedLinear.java`
- New: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/QuantizedLinearTest.java`

**Design decision: the constructor takes already-quantized components, not a float weight to quantize
on construction.** This mirrors `Linear`'s own constructor (`MLXArray weight`, not "an initializer
spec") and is the only shape compatible with Phase 5's eventual checkpoint loader: a real quantized
checkpoint (the outline's own stated GGUF/MLX-quantization use case) stores `weight`/`scales`/`biases`
directly on disk -- there is no plaintext weight at load time to quantize. `MLXQuant.quantize` is the
tool a caller uses *before* construction (as the test in this task does, to build a fixture from a
known float weight) or never, if loading a real checkpoint. A `QuantizedLinear.from(Linear, groupSize,
bits)` convenience factory is a natural follow-on but has no caller yet -- see Deliberately not
covered.

```java
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
 * checkpoint (Phase 5). Unlike {@link Linear}, {@code forward} needs no explicit-target overload for
 * its weight-bearing op: {@link MLXQuant#quantizedMatmul}'s javadoc explains why.
 */
public final class QuantizedLinear extends Module implements UnaryModule {

  private static final String MODE = "affine";

  private final int groupSize;
  private final int bits;
  private final boolean hasBias;

  /**
   * {@code weight} is {@code [out, packedIn]} `UINT32` (packed columns = {@code in * bits / 32} --
   * only for `bits` in `{2, 4, 8}`; `bits` in `{3, 5, 6}` pack unevenly and this constructor cannot
   * check them, see below); {@code scales}/{@code biases} are each {@code [out, in / groupSize]`};
   * {@code bias}, if non-null, is {@code [out]}. {@code groupSize`/`bits} are each restricted to the
   * exact legal sets this native version supports (`{32, 64, 128}` / `{2, 3, 4, 5, 6, 8}`, confirmed
   * against the shipped binary, Findings section) rather than merely `> 0` -- an out-of-set value
   * would otherwise clear every other check unchanged and fail as an opaque native error at first
   * {@link #forward} call, the same failure mode {@code weight.dtype() != UINT32} exists to prevent
   * for a plain float weight. This hardcodes a native-version-specific set into the layer, the same
   * trade-off Global Constraint 4 already accepts for `mode` ("a genuinely new upstream mode would
   * need this validation set updated too") -- accepted here for the same reason: legal range, not a
   * default value (distinct from Global Constraint 5's "absent means let native pick a default"
   * argument). Native validates the packing/group-size relationship on the first {@link #forward}
   * call; this constructor validates only the shape relationships (and now the legal-set membership)
   * it can check without unpacking {@code weight} -- the same division of labor {@link Linear}'s own
   * constructor draws. Non-null: {@code weight}, {@code scales}, {@code biases}; {@code bias} is the
   * only nullable parameter. Unlike every {@code MLXQuant} method (Global Constraint 5), a {@code
   * null} {@code scales}/{@code biases} here fails as a bare {@code NullPointerException} out of
   * {@code .ndim()} with no named message -- the same gap {@link Linear}'s own constructor already
   * has for its {@code weight} parameter, so this is precedent-consistent rather than a new one, not
   * a guard this constructor is expected to add.
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
```

**Amendment (post-merge review): both the constructor javadoc above (packed-columns sentence,
`only for \`bits\` in \`{2, 4, 8}\`...`) and the `if (32 % bits == 0)` gate around the packed-column
consistency check are wrong, not just conservative -- see this document's own Findings-section
amendment (above, under "`packedCols = cols * bits / 32` only holds for power-of-2 `bits`") for the
full empirical correction.** `packedCols = in * bits / 32` holds for every legal `bits` value
(`{2, 3, 4, 5, 6, 8}`), because `group_size`'s own legal set forces `in` to always be a multiple of
32 -- the byte-level pack-factor unevenness `get_pack_factor`/`get_bytes_per_pack` describe never
surfaces at this word granularity. The gated check silently skipped the packed-column consistency
guard for `bits` in `{3, 5, 6}`, reintroducing the exact deferred-opaque-native-error failure mode
the check exists to prevent. The shipped `QuantizedLinear.java` no longer has this gate: the javadoc
sentence was corrected and the check now runs unconditionally for all six legal `bits` values,
pinned by `constructorRejectsInconsistentPackedColumnCountForNonPowerOfTwoBits`. This code listing is
left as originally written, per this repo's convention of amending rather than rewriting a merged
plan's code -- do not copy the `only for \`bits\` in \`{2, 4, 8}\`` sentence or the
`if (32 % bits == 0)` gate from it into new code.

**`groupSize`/`bits` are plain `int` fields, not `MLXArray` parameters** -- same reasoning as
`Linear`'s `hasBias` boolean (`Module`'s own javadoc: "a cached scalar field ... has no such hazard,
since it is never an `MLXArray` and is never rebound"). They are structural configuration of the
layer, not a value `update`/`rebind` ever touches.

**No memory-leak-loop test names a specific hazard here, unlike `Linear`'s.** `Linear.forward`'s own
`MLXMemoryLeakTest`-shaped test exists specifically because `transpose(W)` is a single-operand op on a
parameter that *could* allocate into the model scope if written wrong (§2's fifth sub-hazard).
`QuantizedLinear.forward` has no such op: `quantizedMatmul` is multi-operand with `x` always present
(Global Constraint 6), and `add(y, bias)` resolves through the existing `binaryOp`/`scopeOf` path from
`y` (already in the step scope) and `bias` (the model scope) exactly like `Linear`'s own `add` call.
A leak-loop test is still added, as a general regression guard consistent with every other layer
having one, but its javadoc says explicitly that it targets no named hazard, rather than inventing one
to match the pattern.

### `QuantizedLinearTest.java` (new file)

`@EnabledIfNativeAvailable`; fixture: `[2, 64]` float weight (`out=2`, `in=64` -- the smallest shape
divisible by the `groupSize=32` used throughout Task 1, doubled to `out=2` so shape-mismatch tests on
`bias`/`scales` have a non-trivial dimension to get wrong), built with deterministic non-uniform
values `(i % 7 - 3) * 0.3f`; `x` fixture `(i % 5 - 2) * 0.2f`, shape `[1,64]`; `bias = [0.5, -0.5]`.
`EPS = 1e-3f` -- verified against this exact fixture (probed directly, same precedent as Task 1's
Findings): `forwardMatchesLinearOnTheDequantizedWeight`'s two sides measured `[0.11749995,
-0.16250014]` vs. `[0.11750007, -0.16249979]`, max abs diff `≈3.6e-7`. This is a composition identity
on identical quantized components (not a comparison against the true unquantized weight, whose own
≈0.5%-relative quantization noise is a different, larger-magnitude question this test does not ask),
so `1e-3f` is confirmed generous here, not tight.

| Test | Proves |
|---|---|
| `forwardMatchesLinearOnTheDequantizedWeight` | Composition identity, tight `EPS = 1e-3f` (this is fused-vs-unfused arithmetic on identical quantized components, not a lossy comparison -- see `MLXQuantTest`'s equivalent row, and the verified fixture/tolerance note above): `new QuantizedLinear(wQ, scales, biases, bias, ...).forward(x)` vs. `new Linear(dequantize(wQ, scales, biases, ...), bias).forward(x)` |
| `forwardWithoutBiasComputesTheQuantizedTransformOnly` | The `hasBias ? add : y` branch's `false` side, mirroring `LinearTest`'s equivalent row |
| `parametersExposesWeightScalesBiasesAndOptionalBiasInTheCheckpointLayout` | `parameters()` returns `weight`/`scales`/`biases`/`bias` at the exact dotted paths a Phase 5 loader will write into -- the same "pins the layout contract" reasoning as `Linear`'s own `parametersReturnsWeightInTheCheckpointsOutInShapeNotItsTranspose` test |
| `constructorRejectsScalesAndBiasesWithMismatchedShapes`, `constructorRejectsScalesWithWrongOutDimension`, `constructorRejectsABiasOfTheWrongLength`, `constructorRejectsAnUnsupportedGroupSizeOrBits` | Each shape/value guard in the constructor, mirroring `LinearTest`'s `constructorRejects*` precedent. The last one asserts against the legal-set guard (e.g. `groupSize=100` or `bits=7`), not mere positivity -- `groupSize`/`bits` are restricted to `{32, 64, 128}`/`{2, 3, 4, 5, 6, 8}` (Findings section), so `bits=1`/`16`/`32` and `groupSize=16`/`48`/`100` must all be rejected here even though they're positive |
| `constructorRejectsAFloat32Weight` | The `weight.dtype() != UINT32` guard -- the cheapest, most likely real-world misuse (passing a plain `Linear`-style float weight, which otherwise clears every shape check unchanged) |
| `constructorRejectsAWeightWithAPackedColumnCountInconsistentWithGroupSizeAndBits` | The `packedCols`/`groupSize`/`bits` consistency guard, for `bits=4` (power-of-2, so the check actually runs). **Not** "a `weight` shaped for `groupSize=16` passed alongside `scales`/`biases` shaped for `groupSize=32`" as an earlier draft of this plan proposed -- `packedCols = in * bits / 32` has no `groupSize` term at all, so a `weight`/`scales`/`biases` triple that is *internally* consistent (both actually built at the same `groupSize`) never trips this guard regardless of which `groupSize` value is passed to the constructor; that example's own numbers confirm it (`in = 2*32 = 64`, `expected = 64*4/32 = 8` either way, matching the fixture's actual `weight.shape()[1] = 8`). The guard instead fires on a **declared** `groupSize`/`bits` that is inconsistent with the *actual* `weight`/`scales` shapes already built: quantize the `[2,64]` fixture at `groupSize=32`/`bits=4` (giving `weight` `[2,8]`, `scales` `[2,2]`, per Findings), then construct `QuantizedLinear` passing `groupSize=64` instead of `32` -- **not** `groupSize=16`: the constructor's own legal-set guard (this task's other new check) now rejects `16` before the packed-column check is ever reached, so the mismatched value must itself be a legal one. `groupSize=64`: `in = scales.shape()[1] * 64 = 128`, `expected = 128*4/32 = 16`, actual `weight.shape()[1] = 8` -> throws for the intended reason |
| `activeMemoryDoesNotGrowWithPerIterationChildScopeUnderALongLivedParent` | General regression guard (see the class-level note above on why no specific hazard is named); same warmup/measured-iteration/threshold shape as `LinearTest`'s equivalent |

## Task 3: Documentation (§9)

No production op or layer code in this task -- only the four items spec §9 names, each checked against
the current tree rather than assumed still-needed.

* **`req/project-outline.md`'s Phase 4 section (`:83-91`): mark delivered**, with the same explicit
  reconciliation Phase 3 used for "thread-safe" rather than a bare checkmark -- state that `nn` is
  built on a public op surface living in `core` (not inside `nn` itself, contra a literal reading of
  the outline's own package name), that autograd (`MLXGrad`) shipped in M2 even though the outline
  lists no autograd deliverable anywhere under Phase 4, and that `ROPE` shipped as `MLXFast.rope` (a
  `core` op, M3) rather than as an `nn` module the way the outline's own Phase 4 deliverables list
  groups it alongside `SiLU`/`GELU` -- this codebase has no `nn.Rope` type at all.
* **The architecture diagram (`req/project-outline.md:13`) still lists `Stream`, `Device` and
  `Autograd` as `core` types.** Amend the `core` row: `Autograd` is real, as `MLXGrad`; `Stream` and
  `Device` still do not exist (this slice caches a single process-lifetime default of each -- see
  `MLX.defaultDevice()`/`defaultStream()` -- and exposes no switching). Amend the line rather than
  leaving it aspirational, per §9's own instruction.
* **`MLX.java`'s class javadoc needs two fixes, not zero.** Verified against the current tree while
  writing this plan (an earlier draft of this plan claimed no change was needed here -- that was
  wrong on both counts):
  * Its own split-index list names `MLXOps`, `MLXShape`, `MLXFast`, `MLXQuant` and `MLXRandom` but
    **omits `MLXGrad`** (added in M2, after the index was written in M0a and never revisited). Add it.
  * Its javadoc still states "every op's result is allocated in the same scope as its first
    `MLXArray` operand" -- superseded by §2's `scopeOf`/`innermost` rule (the innermost scope among
    *every* operand, not just the first) as of M0b/M1. Correct the sentence to describe `scopeOf`
    rather than the pre-§2 rule it names.
* **Every split class still missing the "see `MLX`'s javadoc for the index of every sibling"
  back-pointer gets it, not just `MLXQuant`.** Verified against the current tree while writing this
  plan (an earlier draft asserted `MLXQuant` was "the one split class still missing it" -- false):
  `MLXOps` and `MLXShape` already carry it; `MLXFast`, `MLXRandom`, `MLXGrad` and `MLXQuant` do not.
  Task 1 replaces `MLXQuant.java`'s M0a placeholder ("Empty until M4 ... created now, during M0a's
  pure-motion facade split") with a real description of `quantize`/`dequantize`/`quantizedMatmul`
  plus the back-pointer. This item adds the same one-line back-pointer to `MLXFast`, `MLXRandom` and
  `MLXGrad`'s existing class javadocs (each already has a real description; only the back-pointer
  sentence is missing) -- four classes' javadoc touched in this task, not one.
* **`jmlx-examples/HelloMLX`: add a `Linear` forward pass inside a child scope.** Current `HelloMLX`
  (`jmlx-examples/src/main/java/se/alipsa/jmlx/examples/HelloMLX.java`) only exercises M0a/M0b-era ops
  (`add`/`matmul`, cross-shape broadcast) in a single flat scope -- neither of Phase 4's two
  behavioural changes (cross-scope ops resolving via `scopeOf`, child-scope lifetime) is visible in
  the demo. Add, after the existing `broadcastSum` block: build a `Linear` with a weight/bias in the
  outer (model) scope, open a child scope, build an activation `x` in the child, call
  `linear.forward(x)`, print the result, let the child close. This demonstrates exactly the shape
  Task 2's own `QuantizedLinear` forward relies on (weight in an ancestor scope, activation in a
  descendant), without requiring `HelloMLX` to depend on quantization specifically.

## Task 4: full verification pass

Mirrors `req/plans/phase4-m3-plan.md`'s own closing task. No new production code -- this task only
runs checks and fixes anything they surface.

- [ ] `./gradlew build` -- Spotless, Checkstyle, the forked `loaderGuardTest`, plus every module's test
  suite. Report actual test counts.
- [ ] `./gradlew :jmlx-core:test --tests '*MLXMemoryLeakTest*'` -- confirms this plan's new op surface
  introduced no regression in the existing leak suite.
- [ ] `./gradlew :jmlx-core:test --tests '*MLXGpuVerificationTest*'` -- kernels still dispatch to GPU.
- [ ] `./gradlew :jmlx-examples:run` -- `HelloMLX` prints correct values for the new `Linear` forward
  pass in a child scope (Task 3's last bullet).
- [ ] `scripts/regen-bindings.sh` then `git diff --exit-code jmlx-ffi/src/main/generated/java` -- must
  stay clean. This plan adds zero new mlx-c symbols (`mlx_quantize`/`mlx_dequantize`/
  `mlx_quantized_matmul` and `mlx_optional_dtype_` are all already bound, confirmed while researching
  this plan) -- the last of `req/phase4-plan.md`'s four merge points to prove Phase 4 needed no
  binding regeneration at all.
- [ ] `grep -rn 'mlx_array_new(' jmlx-core/src/main` -- every call site's allocator is `scopeOf(...)`'s
  result, an explicit `MLXScope` parameter, or a confined `Arena` freed by a local `finally` (spec
  Verification #6). `dequantize`/`quantizedMatmul`'s hand-rolled bodies from Task 1 must fall into the
  first case.
- [ ] `grep -rn 'mlx_vector_array_new' jmlx-core/src/main` -- every call site's allocator confirmed
  **by eye** to be a confined `Arena`, never an `MLXScope` (spec Verification #7). `quantize`
  introduces no new call site here at all -- it reuses `vectorOutOp`, whose own `mlx_vector_array_new`
  call was already verified in M3.
- [ ] `grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` -- zero
  matches (spec Verification #6a).
- [ ] Update the Status table at the top of `req/phase4-plan.md`: mark M4 `Done` and §9 `Done`, each
  with a commit hash and a short findings note (matching M1-M3's own entries) summarizing this plan's
  empirically-confirmed quantization facts, so a later reader does not have to re-derive them.
- [ ] Mark Verification item **0d** `DONE, confirmed` inline in `req/phase4-plan.md`'s own Verification
  section (`:1120`), the same way **0b** and **0f** already are there -- not just via the Status table
  update above. The Status table alone left 0b/0f's own Verification-section entries as the only place
  a reader would otherwise have to cross-reference to confirm they were actually resolved; 0d gets the
  same treatment this task's own findings just earned it.

## Deliberately not covered by this plan

- **§10 (CI, self-hosted runner)** -- an unrelated, already-tracked follow-up; not touched here, same
  as every prior plan.
- **`Mxfp4`/`Mxfp8`/`Nvfp4` quantization modes** -- reachable through `MLXQuant`'s generic `mode`
  string parameter but exercised by no test; their packing layout is not `affine`'s and is not
  verified against this mlx-c version at all (Findings section).
- **`mlx_gather_qmm`/`mlx_qqmm`/`mlx_gather_mm`** (gathered/MoE quantized matmul) -- not named in spec
  §8; no consumer in this codebase.
- **`QuantizedLinear.from(Linear, groupSize, bits)`, or any other quantize-on-construction
  convenience** -- no caller needs it yet (see Task 2's design-decision note); the two-step
  `MLXQuant.quantize` then `new QuantizedLinear(...)` path this plan's own test uses is not hidden
  behind a factory until a second caller makes the duplication worth naming.
- **Quantization-aware training** (gradients through `quantizedMatmul`, or a `ModuleGrad` that treats
  quantized weights specially) -- `QuantizedLinear` is an inference-shaped layer; nothing in Phase 4's
  scope trains one.
- **Safetensors/GGUF loading of real quantized checkpoints** -- Phase 5, per `req/phase4-plan.md`'s
  own Out of scope section. `QuantizedLinear`'s components-based constructor (Task 2) is the hook Phase
  5 will load into, exactly as `Linear`'s own constructor already is.
