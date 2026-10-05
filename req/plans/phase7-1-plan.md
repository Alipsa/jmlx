# Phase 7.1 — Core inference modules

- **Status:** implemented locally, 2026-10-05; milestone acceptance pending oracle/CI gates.
- **Parent:** `req/plans/phase7-plan.md` §7.1 and public-contract decisions 5–6.
- **Prerequisite:** the dated Phase 7 amendment in `req/full-roadmap.md` is present.

## 1. Objective and scope

Deliver the conventional inference layers needed by the Phase 7 encoders and vision towers, with
safe core facades and differential evidence against the pinned Python MLX runtime. 7.0 is not a
dependency. 7.2 depends on this milestone; 7.3a may start after the convolution, normalization and
activation work below passes its gates.

Include every operation, layer and evaluation loss listed in the parent §7.1. Keep model loaders,
checkpoint layout conversion, image preprocessing, cross-attention, T5 relative-position bias,
training dropout, optimizer/loss-gradient guarantees and broad Phase 8 parity out of scope. No
native pin change or generated-binding regeneration is planned. An upstream limitation must be
documented explicitly rather than worked around by silently changing the promised semantics.

## 2. Baseline and design constraints

- Extend `MLXOps` and `MLXShape`; create `MLXConv` for the stable convolution family.
- New unary layers extend the existing `UnaryLayer`. `Module` itself has no universal forward
  method: `Sequential` accepts unary layers, while `ModuleList` stores arbitrary modules.
- Register weights through `param` and children through `child`; forward reads parameters fresh,
  preserving `update`, `rebind` and `ModuleGrad` behavior. Do not cache weight-derived arrays.
- Activations are channels-last: NLC, NHWC and NDHWC. Ordinary convolution weights are `[out,
  spatial kernel dimensions..., in/groups]`. Verify transpose-convolution layout separately.
- Results use `NativeOps.scopeOf` across every array operand, including bounds and padding values.
  Parameter-only preprocessing and creation operations explicitly target the activation scope.
- All downcalls use `NativeOps.checked`, the result scope's stream and existing allocation helpers.
  Temporary native storage closes on success and failure. Native errors remain `MLXException`.
- No production `ffi` access outside core. Native tests use `@EnabledIfNativeAvailable`.

## 3. Probe gate before implementation

Add small, native-gated probe cases and `StridedViewSafetyTest`, run with the current staged
runtime. Write actual observations to `req/plans/phase7-1-probe-findings.md`: native pins, Java,
OS/device, precision mode, commands, inputs, shapes/dtypes, status/error messages and conclusions.
Inspect the installed pinned Python MLX layer sources as well as runtime results. Do not substitute
current upstream documentation for the pinned behavior.

| Surface | Required experiment and decision |
| --- | --- |
| `mlx_conv1d/2d/3d`, `mlx_conv_general` | Distinct channels/kernel values prove layout; grouped/depthwise cases, asymmetric padding, kernel/input dilation, flip and output dimensions; invalid shape errors, including failures delayed until eval. |
| `mlx_conv_transpose1d/2d/3d` | Weight orientation and channel/group relationships; stride, dilation, padding and output-padding combinations; record supported combinations separately for each rank. See findings: 3-D groups are unsupported; grouped transpose2d with non-unit stride is rejected due to a GPU defect. |
| `mlx_pad`, `mlx_pad_symmetric` | Test `constant`, `edge`, `reflect`, `symmetric`; only expose modes accepted by the runtime. Check scalar pad value, promotion, selected axes, empty dimensions and negative widths. Symmetric means equal widths, not reflection. |
| `mlx_max_axes`, `mlx_max_axis`, `mlx_max`, `mlx_var_axes`, `mlx_softmax_axes` | Empty/negative/duplicate axes, keepdims, empty reductions, ddof and precise softmax semantics; preserve valid native behavior. |
| `mlx_as_strided` | Element-stride and offset units; contiguous/transposed/sliced/broadcast input semantics, including a row-contiguous sliced view with a nonzero backing-buffer origin; safe negative/zero strides, zero extents, scalar views, alias lifetime and lazy evaluation. |
| `mlx_tile`, `mlx_repeat_axis`, `mlx_clip` | Zero/negative repetitions, rank extension and negative/out-of-range repeat axes; both array bounds, inverted bounds, and whether empty bounds encode one-sided clipping. |
| Python `nn` | Pool window rounding/padding, upsample coordinates and rounding, normalization axes/statistics and GroupNorm ordering, activation/loss defaults and positional-encoding formulas, including the sinusoidal `dims == 2` edge case. |

Separate safe regression tests from potentially process-fatal probes. Add `StridedViewSafetyTest` to
ordinary `:jmlx-core:test`, hence `check`, and to CI's required native suite list. It checks
offset-zero and row-contiguous sliced-view origins using only in-bounds views. Pin changes therefore
exercise these safety invariants in regular native CI.

Use `./scripts/run-phase71-native-probes.sh` as the single supported entry point for fatal probe
evidence. The launcher invokes the internal `:jmlx-core:phase71NativeProbe` Test task once per case
with `--tests <case-class>` and the task-specific `--rerun` option. Each invocation runs exactly one
class in its own JVM; no `forkEvery` setting is needed. Exclude fatal-case classes from ordinary
`test` and `float32GoldenTest`; the internal task is evidence only, never part of `check`. The
launcher continues after nonzero exit statuses, preserves each case's reports/logs in a distinct
output location before the next invocation, and emits an aggregate outcome summary and exit status.
Record all outcomes, distinguishing executed successes, expected failures and infrastructure errors.

For each invocation, inspect the fresh JUnit XML for the selected class, not only Gradle's exit
status. Any skipped test or zero executed tests is an infrastructure error, even when Gradle exits
zero. Missing, stale or malformed XML also fails evidence validation. Clear or isolate each case's
result directory before invocation so an old report cannot count as current evidence. A fatal
process termination without complete XML can count as an expected failure only when that outcome was
explicitly declared for the case and fresh process logs identify that case's execution and expected
termination; otherwise it is an infrastructure error. Include XML execution/skip counts and exit
status in the aggregate summary, and return nonzero for any infrastructure error or unexpected
failure. Never evaluate a view whose reachable addresses are outside a known allocation: unsafe
offset probes stop at Java validation, or remain unexecuted and documented. Probing must not
introduce an out-of-bounds read.

Any missing runtime capability gets a dated finding and an explicit plan/parent-plan amendment
before scope is reduced. The probe gate fixes constructors/defaults before public APIs land.

## 4. Facade contracts and exact bindings

| Facade | Java API | mlx-c declaration |
| --- | --- | --- |
| `MLXOps` | `abs`, `minimum`, `clip`, `floor`, `log1p` | `mlx_abs`, `mlx_minimum`, `mlx_clip`, `mlx_floor`, `mlx_log1p` |
| `MLXOps` | `maxAxes(a, axes, keepdims)`, `varAxes(a, axes, keepdims, ddof)` | `mlx_max_axes`, `mlx_var_axes` |
| `MLXOps` | `softmaxAxes(a, axes, precise)` | `mlx_softmax_axes` |
| `MLXOps` | `logSoftmax(a, axis)` | composition using existing `mlx_logsumexp_axis` and subtraction |
| `MLXShape` | `pad(a, axes, low, high, value, mode)`, `padSymmetric(a, width, value, mode)` | `mlx_pad`, `mlx_pad_symmetric` |
| `MLXShape` | `asStrided(a, shape, strides, offset)` | `mlx_as_strided` |
| `MLXShape` | `tile(a, repetitions)`, `repeatAxis(a, repeats, axis)` | `mlx_tile`, `mlx_repeat_axis` |
| `MLXConv` | `conv1d/2d/3d`, `convTranspose1d/2d/3d`, `convGeneral` | `mlx_conv1d/2d/3d`, `mlx_conv_transpose1d/2d/3d`, `mlx_conv_general` |

Convolution methods expose every native argument: strides, padding, dilation, groups, transpose
output padding; general convolution exposes low/high padding instead of symmetric padding, plus
input dilation and flip. Use scalar spatial arguments for 1-D and defensively copied spatial arrays
for 2-D/3-D and general convolution. Validate array lengths, channel relationships and checked
output-size arithmetic without rejecting valid pinned-runtime inputs. Do not add string `same`
padding unless its semantics are separately derived and tested. Bias is a layer composition.

`clip` accepts array bounds and broadcasting. Probe empty-bound one-sided clipping; expose it only
if supported, documenting the exact representation. Otherwise require both bounds and use
`minimum`/`maximum` for one-sided clipping. `tile` allows zero repetitions and rank extension where
supported, rejects negative repetitions before native work, and documents leading-axis alignment.
`repeatAxis` allows zero repeats, rejects negative repeats, normalizes valid negative axes and
rejects out-of-range axes. Confirm these rules against the pinned runtime. Pad values are scalar
tensors; convenience scalar creation uses the operand's scope. Pad modes use a typed enum containing
only confirmed modes. Axes follow existing reduction conventions; tests establish negative-axis
normalization and empty axis behavior. `varAxes` exposes native `ddof`; `softmaxAxes` exposes
`precise`.

### Safe strided views

Use `int[] shape`, `long[] strides` in elements and a nonnegative `long offset`, checked before
conversion to native `size_t`. Shape and stride ranks must agree; extents must be nonnegative. For
nonempty views compute, with `Math.multiplyExact`/`Math.addExact`:

- `low = offset + sum(min(0, (extent - 1) * stride))`;
- `high = offset + sum(max(0, (extent - 1) * stride))`.

Require `0 <= low <= high < input.size()`. Overflow becomes `IllegalArgumentException`. Zero-extent
views reach no elements; their offset still must be representable. Rank-zero views reach one element
at offset. Negative/zero strides are supported only if the safe probes confirm support; otherwise
reject them with a named message.

Always call `mlx_contiguous(a, false)` into the result scope before striding, including already
contiguous inputs. Bounds refer to this normalized view's logical element count. The probe must
confirm that native offset zero addresses the start of that view, including a row-contiguous slice
sharing a larger buffer. If that invariant fails, materialize a proven independent copy or reject
the input before striding; block public exposure until this is resolved. The offset-origin invariant
is enforced by ordinary `StridedViewSafetyTest`, on CI's required native-suite list, on every native
build including pin changes. Record whether normalization aliases or copies; keep temporaries in the
result scope. Pooling starts with a package-private checked helper. Expose the public method only
after its complete boundary and lifetime tests pass. Never expose an unchecked escape hatch.

## 5. Module contracts

### Containers and inference mode

Add recursive `Module.train(boolean)` and `isTraining()`, defaulting to eval. Frozen parameters and
mode are independent. Newly registered children inherit the parent's current mode; switching mode
traverses existing children. Cycle/shared-parent trees retain `Module.child`'s existing unsupported,
undefined behavior; there is no runtime prohibition today. Document that recursive `train` has the
same limitation, including possible stack overflow for cycles; this milestone does not add
tree-ownership checks.

`Sequential` extends `UnaryLayer`, applies its fixed ordered children, and is identity when empty.
`ModuleList` extends `Module`, offers size/get/iteration over fixed children and no invented forward
operation. Both register decimal child names (`0`, `1`, ...), reject null children and preserve
parameter-path ordering. Fixed membership avoids changing frozen parameter trees.

`Dropout` validates its probability against the pinned MLX contract. Eval returns the input;
training throws `UnsupportedOperationException` naming Phase 11, including probability zero.
`BatchNorm` also rejects training before allocating temporaries. Document jmlx's deliberate eval
default instead of MLX's training default.

### Activations

Add `ReLU`, `LeakyReLU`, `ELU`, `SELU`, `Tanh`, `Sigmoid`, `Softplus`, `Mish`, `HardSwish` and
`QuickGELU` as unary modules. Match pinned constants and configurable slopes/defaults; use stable
Softplus (`max(x, 0) + log1p(exp(-abs(x)))`) rather than an overflow-prone exponential. Mish
composes Softplus and tanh. QuickGELU means Hugging Face's `x * sigmoid(1.702 * x)`; use the
constant 1.702 explicitly and a formula fixture with HF reference provenance. Do not substitute
MLX's `gelu_fast_approx` or assume a pinned MLX QuickGELU class exists.

Extend `Activation` additively for applicable model FFNs and centralize application so new enum
values do not leave existing switch sites incomplete. In `ArchitectureMappings`, extract one
string-to-`Activation` resolver accepting the config key name for error messages. Keep config
parsing, per-family allowlists, Gemma defaults and conflict rules separate from resolution.
Non-Gemma decoders still accept only silu/swish and reject `hidden_activation`; negative tests
include Llama `hidden_act: relu` and `hidden_activation: quick_gelu`. T5 in 7.2 obtains its
activation from `feed_forward_proj`/derived `dense_act_fn` (gated-gelu maps to gelu_new), not
`hidden_act`; it can call the resolver with that key name. Test `relu`, `silu`/`swish`, `gelu`,
`gelu_new`, `gelu_pytorch_tanh` and `quick_gelu`; unknown strings fail by name. BERT/T5/SigLIP/CLIP
support is not claimed merely by resolving a name.

### Spatial layers

- `Conv1d/2d/3d` and `ConvTranspose1d/2d/3d`: checkpoint-weight constructors, optional bias,
  immutable spatial configuration, shape/group validation and native-backed forward. Read weights
  fresh; transpose layout is fixed by the probe. No random weight initialization is required.
- `MaxPool1d/2d`, `AvgPool1d/2d`: checked strided windows plus max/mean reduction, following the
  pinned Python layer's exposed kernel/stride/padding and edge behavior. Explain padded-average
  denominators and max padding sentinels where applicable. Compute each output extent with
  `Math.floorDiv(paddedSize - window, stride) + 1` using checked arithmetic, never Java `/`. A
  larger-than-input window can produce a zero extent (for example size 2, window 3, stride 2);
  preserve pinned behavior: max reduction rejects a zero-size output, while average reduction
  follows native mean semantics. Reject a negative computed extent with
  `IllegalArgumentException` naming input/padded size, window and stride before calling the striding
  helper. Probe negative-extent behavior only through the Python pooling layer, never by
  constructing a native negative-shape view. Do not import PyTorch-only options.
- Padding layers: add unary `Pad` and `PadSymmetric` wrappers around the facades, with copied
  configuration and confirmed modes, covering the roadmap's padding-layer requirement.
- `Upsample`: nearest and linear for the spatial ranks supported by pinned MLX; immutable scales and
  mode, channels-last. Define output rounding, coordinate mapping and alignment options from the
  reference, then compose floor/clip/take and interpolation. Cover integer/fractional scales,
  size-one dimensions and boundaries. Clip every computed gather index to the valid spatial interval
  before `take`; handle empty source dimensions before any gather. Unsupported modes fail
  explicitly.

### Normalization and position

`GroupNorm` and `InstanceNorm` define exact spatial/channel reduction axes, population variance,
epsilon and optional affine parameters against pinned MLX. GroupNorm exposes any pinned ordering
option and requires channels divisible by groups. Probe the pinned 0.31.2 default: if it uses MLX's
interleaved grouping (`pytorch_compatible=false`), preserve and document that default. Contiguous
PyTorch grouping is `pytorch_compatible=true`; 7.3a checkpoint loaders must explicitly select it for
PyTorch provenance, with fixtures distinguishing the two layouts. Handle noncontiguous inputs
through core ops.

`BatchNorm` supports `trackRunningStats=true` (default) with loaded running mean/variance, and
`trackRunningStats=false` with batch mean/variance computed fresh in eval. Both support optional
affine weight/bias. Accept input ranks 2–4 only, reject other ranks before native work, and define
the channels-last reduction axes explicitly. Verify this contract against pinned 0.31.2; any
mismatch requires a plan amendment rather than silently choosing 0.32.2 semantics. Register running
statistics only when tracking is enabled; reject supplied statistics when tracking is disabled. Test
both eval paths, ranks 2/3/4, invalid ranks and affine on/off, with explicit names and shape checks.
Register running statistics consistently with the existing parameter mechanism and reject training
before mutation. Document the resulting limitation explicitly: `ModuleGrad.of(batchNorm, ...)`
includes registered floating running_mean/running_var in its differentiated paths because Module has
no buffer or per-key freeze facility. Add a test proving those paths and gradients are included, and
retain update/rebind behavior. Eval mode and whole-module `freeze()` do not exclude statistics from
differentiation. Non-trainable buffers/per-key freezing are deferred to Phase 11; no momentum update
or training BatchNorm claim is made here.

`SinusoidalPositionalEncoding` follows the pinned MLX implementation's dimensions, frequency, scale
and sine/cosine ordering; distinguish returned position features from adding them to input. `ALiBi`
follows the pinned MLX helper's symmetric `-abs(q-k)` distance, slope ordering and score-addition
behavior; expose `forward(scores, offset)` rather than silently adopting BLOOM/MPT key-position-only
bias. Test non-power-of-two head counts, query/key lengths and cached offsets. Creation accepts a
target scope; never retain activation-sized position tensors on a model scope. Use the pinned ALiBi
helper as the oracle. Probe sinusoidal dimensions 2 and minimum/odd sizes; reject a dimension that
makes the denominator undefined rather than generate accidental NaNs. Document any intentional
validation difference from the pinned helper.

Existing embedding classes serve learned positions and token types. Bag/sparse variants and
model-specific relative-position bias remain deferred as specified in the parent.

## 6. Evaluation losses

Add a static `Losses` utility under `nn`; return lazy `MLXArray` values. Define a shared reduction
with NONE, MEAN and SUM. Match the pinned `mlx.nn.losses` defaults, argument order and shape rules
in javadoc and fixtures rather than assuming every loss averages the same axes.

| API | Contract to fix and test |
| --- | --- |
| `crossEntropy` | Logits, class axis, integer or probability targets where supported, weights, label smoothing, stable log-softmax; smoothing endpoints and Java rejection of invalid labels before gather. |
| `binaryCrossEntropy` | Probability/logit input option, weights, stable logit formula, boundary probabilities and broadcasting. |
| `nll` | Log-probability input, target gather, class axis and reductions. |
| `mse`, `l1`, `smoothL1` | Pinned equal-shape validation, reduction and smooth-L1 beta/transition behavior. |
| `klDiv` | Pinned log-input/log-target convention, zero-target terms and reductions; no log-target option is planned. |
| `cosineSimilarity` | Explicit axis and the pinned default (verify axis=1 in 0.31.2), epsilon and zero-norm behavior; similarity is not silently negated into a loss. |

Compositions use existing gather/take/reduction operations and new facade slices; no direct FFI. Add
only a proven missing helper if fixtures reveal a dependency, recording its exact declaration and
inventory status. Weights are in scope for cross-entropy and BCE, with pinned broadcasting/reduction
rules. Ignored-target extensions and gradient guarantees remain deferred. Validate integer labels on
Java before any native gather (including target shape, class range and conversion overflow); if
tensor targets require a host read, document that synchronization. Invalid-label tests assert Java
rejection and never execute an out-of-bounds native gather. NLL follows the same rule.

## 7. Oracle and test evidence

Extend `tools/mlx-oracle/runner.py` with a Phase 7.1 fixture dispatcher and canonical input/expected
pairs. Use explicit fixed weights, biases and running statistics, not random initialization. Set
`MLX_ENABLE_TF32=0` explicitly in the shared Gradle oracle Exec configuration and declare it as a
task input; shell settings cannot change oracle precision. Add per-fixture-family provenance: retain
the existing GPU device entry for Phase 6, and add a Phase 7.1 profile with CPU device and
full-float32 precision. Update both `runner.py:run()` and `verify_environment.py` to select and
validate per-family profiles rather than relying solely on the global device. The environment
verifier must evaluate a small operation on every declared profile's device, including the 7.1 CPU
runtime, and validate precision configuration. Preserve legacy Phase 6 GPU validation; reject
missing or mismatched profiles for 7.1. CPU references remove GPU-generation differences from
byte-exact regeneration. Verify the canonical 7.1 output on both the development host and CI host
before accepting seven-decimal serialization; if CPU results differ, block the gate and adopt a
documented 7.1-only rounding policy with matching Java tolerances, leaving Phase 6 serialization
unchanged. Confirm that every planned reference operation supports CPU before adopting its fixture;
no silent GPU fallback. Existing Phase 6 serialized output and hashes must remain unchanged; verify
them under the now explicit mode. Any discovered existing precision drift is a separately reviewed
infrastructure issue, not permission to rewrite old goldens. Inspect reference implementation
signatures in the pinned environment before finalizing options. Document generator commands, CPU
selection and provenance-schema changes in `tools/mlx-oracle/README.md`. Extend `rounded()` with the
string sentinel `NaN` alongside existing `Infinity`/`-Infinity`, preserving `allow_nan=False`. Add
one shared `OracleFixtureReader` in `jmlx-core/src/testFixtures/java`, enabling core's
`java-test-fixtures` plugin and models' `testImplementation(testFixtures(project(':jmlx-core')))`.
Both core's 7.1 tests and models' `SamplingOracleTest` use it; replace the latter's private
`floatsWithInfinity` helper rather than maintaining two decoders. Decode all three sentinels and
reject unknown strings. Add `testFixturesImplementation libs.jackson.databind` in core; keep
assertion dependencies in test configurations unless the reader actually requires them.

In `jmlx-core/build.gradle`, skip both `testFixturesApiElements` and `testFixturesRuntimeElements`
with `components.java.withVariantsFromConfiguration(...) { skip() }`, following ffi's publication
guards. Add `verifyPublishedDependencies`, wired into core `check`, depending on
`generatePomFileForMavenPublication` and `generateMetadataFileForMavenPublication`. Assert the POM's
exact existing production dependency set (including jmlx-ffi) and that neither POM nor module
metadata contains fixture dependencies, fixture variants/capabilities or a fixture jar artifact.
Assert production metadata dependencies are unchanged too. This prevents Jackson/JUnit and the
internal reader variant leaking into Central publication without disabling in-build
`testFixtures(project(':jmlx-core'))` consumption.

Configure `tasks.withType(Test).configureEach` in both core and models with `jmlx.repository.root =
rootProject.projectDir.absolutePath` and task file inputs covering
`tools/mlx-oracle/fixtures/*.json` (input and expected JSON). Preserve models' existing root
property setting while adding the missing file inputs. Both ordinary and float32 tasks must rerun
after fixture generation. Verify with an up-to-date/input-change check that changing a fixture
invalidates the consuming test tasks. Add serialization/decoding cases for finite values, NaN and
both infinities. Compare NaN classification and infinity sign explicitly, never through a numeric
tolerance. Mathematical boundary NaNs are reference outputs; invalid labels are validation cases,
not oracle gather inputs.

Each new layer, loss and facade gets hand-computed cases plus differential fixtures. Fixtures cover
nondefault options, unequal dimensions and asymmetric values so layout errors cannot pass.
Convolution coverage includes all ranks, groups/depthwise, strides/dilations and transpose output
padding. Pooling covers overlapping windows and transposed/sliced inputs. Norms cover affine on/
off, epsilon, batches and spatial ranks. Every activation covers negative/zero/positive and large
magnitudes; losses cover every reduction, weights, stable extremes and shape failures.

Separate tests cover closed/wrong-thread arrays, related and unrelated scopes, operand-derived
allocation, parameter update/rebind, and temporary cleanup after native errors. A repeated-forward
memory test evaluates outputs in child scopes, closes them and compares warmed-up active bytes
against a stated bounded margin; model-scope memory must not grow per call. Stride tests include
first/last bounds, one past each, arithmetic overflow, negative/zero strides, empty extents, scalar
last-valid/one-past offsets, and source-scope closure.

Tag strict numerical references `full-float32`, selecting `:jmlx-core:float32GoldenTest` with
`MLX_ENABLE_TF32=0`. Use combined float32 bounds `abs(actual-reference) <= atol + rtol *
abs(reference)`, initially `atol=1e-4`, `rtol=1e-5`, with tighter exact/hand-computed assertions
where warranted. Record per-operation measured errors and justify exceptions before merge. The
parent's `1e-4` absolute HF-logits gate is not a universal layer tolerance: large-magnitude float32
activations need a relative term. Default-mode coverage runs separately with measured, documented
tolerances; never rewrite references to hide TF32 error. Exact shapes, integer values and parameter
paths remain exact in both modes. Add every required correctness suite (excluding the evidence-only
probe task) to `.github/workflows/ci.yml` in its correct ordinary or float32 result directory;
passing skipped tests is not exit evidence. Extend model mapping tests and run existing
decoder/model checks to protect current activation behavior.

## 8. Delivery sequence and gates

1. **Probe and contracts:** probe test, findings document, pinned-source audit, finalized defaults
   and supported-mode table. Gate: native uncertainty resolved or explicitly amended.
2. **Ops and inventory:** `MLXOps`, `MLXShape`, checked striding helper, `MLXConv`, fixture
   dispatcher and facade tests. Gate: safe stride boundaries/lifetimes and native cleanup pass
   before public `asStrided` is exposed.
3. **Containers and activations:** mode propagation, containers, Dropout, activation modules and
   shared model config resolver. Gate: parameter/rebind tests and existing decoder checks pass.
4. **Convolution, padding, normalization:** all ranks and transpose variants plus normalization
   fixtures. Gate: all listed layers match the oracle; 7.3a may begin using these and activations.
5. **Pooling and upsampling:** window helper integration and coordinate/edge fixtures. Gate:
   noncontiguous-input, boundaries and repeated-forward memory checks pass.
6. **Positions and evaluation losses:** positional fixtures, loss contracts and numerical extremes.
   Gate: every declared option and reduction has differential evidence.
7. **Hardening and documentation:** CI suite assertions, API examples, inventory freshness and final
   native/default/strict checks. Gate: complete milestone evidence below.

Each step is a reviewable commit/PR unit; land fixtures with their implementation. No partial
milestone completion claim while a listed layer or loss lacks reference evidence.

## 9. Inventory and documentation

Update `req/mlx-api-inventory-overrides.json` alongside each downcall: initially planned where
appropriate, implemented only with its Java location and tests. Regenerate the readable inventory.
Move `mlx_softmax_axis` from the incorrect `MLXShape` grouping to `MLXOps`, beside
`mlx_softmax_axes`. The inventory schema requires real generated bindings and unique mappings.
Document composed `logSoftmax`, losses and layers only in javadocs and the core README, leaving
shared inventory records' `facadeOrReason` unchanged. Do not create binding-less records, duplicate
`mlx_logsumexp_axis`, add unsupported fields, or assign `derived-java-api` to an implemented
primitive. No inventory-schema/buildSrc change is planned. `mlx_repeat` remains unplanned.

Update core README/examples and javadocs for layout, options, inference mode, normalization state,
strided-view safety and loss reductions. Record any unsupported-by-runtime declaration with probe
proof. Update the Phase 7 master plan's milestone status only when the gate passes; no new model
compatibility rows are earned by this core-only milestone.

## 10. Final verification and acceptance

Run on the supported native host after targeted development checks:

```sh
./gradlew generateMlxApiInventory
./gradlew verifyMlxApiCallSites verifyMlxApiHeaderCoverage
./gradlew verifyMlxOracle verifyMlxOracleFixtures
./gradlew :jmlx-core:check :jmlx-models:check
./gradlew build
git diff --check
```

Fixture rewriting is explicit: `./gradlew generateMlxOracleFixtures`, followed by reviewed
expected/provenance diffs and verification. Do not invoke it merely to resolve an assertion. Run
`./gradlew -p buildSrc check` only if isolated build logic changes.

Acceptance requires all named facade APIs, containers, modes, layers, positions and losses;
committed findings from `./scripts/run-phase71-native-probes.sh`; hand-computed and oracle cases for
each; strict/default precision results; no native suite skipped on the supported host;
scope/thread/error/leak checks; complete inventory mappings and green call-site verification. An
unstaged checkout must still build with the existing native skip guard. No publication, new module
or release-version bump is required.
