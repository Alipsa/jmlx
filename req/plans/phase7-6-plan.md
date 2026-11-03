# Phase 7.6 — Hybrid Qwen text and image inference

- **Status:** implementation in progress, 2026-11-03. Reference probes and core DeltaNet
  prototypes have local evidence; model integration and acceptance remain pending.
  See the [worklog](phase7-6-worklog.md) for the current checkpoint and
  [WP1 benchmark report](phase7-6-wp1-benchmark.md) for the pending default review.
- **Parent:** [Phase 7 §7.6](phase7-plan.md#76--hybrid-linear-attention-qwen-decoders-qwen38).
- **Prerequisites:** 7.1 core operations, 7.3b vision/MRoPE and the delivered 7.5 paired-pin
  upgrade. PR #76 is merged; its timing investigation leaves measurement limits with Phase 8.2.
- **Review boundary:** review this plan before implementation, as the parent requires. WP0 produces
  executable reference findings before production math or loader code begins.
- **Research:** [reference findings](phase7-6-reference-findings.md) separates inspected sources
  and metadata from the executable checks still required.

## 1. Scope and completion

Implement dense `qwen3_5` hybrid decoders, with text and image+text generation through
`TextGenerationModels.load`. Support HF float, mlx-vlm float/affine and mlx-lm float/affine
layouts. An mlx-lm artifact without vision weights reports TEXT only and rejects images.

The required artifacts are `Qwen/Qwen3.5-0.8B` for automated real-artifact acceptance and
`mlx-community/Qwen3.8-27B-4bit` for a recorded manual text/image run. Exact revisions and
metadata hashes are in the findings. A model name or matching config is not verification.

Keep the current native pairing: MLX 0.32.4 / mlx-c
`a26f2d584a14ec619ae8db9a09be03becd47598a`. No generated-binding changes, publication or version
bump. The first implementation uses existing core facades. Exclude MoE/shared experts, video,
MTP execution, speculative decoding, training, non-affine quantization, per-layer quantization
overrides, sliding retention and scheduler admission, including text-only scheduler requests.

Completion requires all of the following:

1. Strict HF float32 Tier-A prefill and cached-decode logits for text and image inputs; cached
   execution also agrees with fresh full forwards, including convolution and recurrent state.
2. Actual converter-produced float mlx-vlm and mlx-lm fixtures match the HF-layout fixture;
   affine fixtures separately match their unmodified MLX reference with identical packed
   weights, with the Q/K normalization discrepancy isolated and reported as specified in WP0.
3. Tier-B 0.8B exact greedy IDs and decoded output, with per-step logit and stability checks.
4. The pinned 27B affine artifact loads and matches mlx-vlm greedy IDs for text and image+text,
   with precision, prompt, pixels, resource use, host state and the measured normalization
   contribution recorded. A normalization-aligned diagnostic is not a replacement reference.
5. Request cleanup, bounded state retention and cancellation/failure tests pass in native CI;
   the compatibility report describes the evidence for each supported layout.

## 2. Reference environments and WP0 stop conditions

Add a separate opt-in tool, `tools/qwen35-reference/`, with independent `hf/` and `mlx/` venvs,
input requirements, hash locks, installers and provenance. Do not upgrade the existing
`tools/hf-reference`, image/tokenizer or Qwen3-VL reference locks to obtain this family.

Proposed direct pins are CPython 3.12, Transformers 5.22.1, Pillow 12.3.0, NumPy 2.5.3,
tokenizers 0.23.2, Jinja2 3.1.6 and safetensors 0.8.0. HF adds Torch 2.6.0 and torchvision 0.21.0 on macOS
(the matching CPU Torch wheel on Linux, if that platform is later verified). MLX adds
mlx-vlm 0.6.8, mlx-lm 0.32.2, mlx 0.32.4 and mlx-metal 0.32.4. Resolve and hash every transitive
dependency; a successful resolution is not an executed reference. Retain platform-specific
wheel provenance and import-source hashes. No Python execution becomes an ordinary build
prerequisite, and no model code downloads or executes remote Python.

WP0 must prove that these pins import, parse the artifact configs and execute tiny text/image
models on the recording platform. Select eager HF attention and the Torch delta-rule fallback;
disable optional FLA/causal-conv/hub kernel substitutions, use CPU float32, eval mode and fixed
seeds. Record actual classes/functions used. In particular, exercise the config's legacy fields
under Transformers 5.22.1 rather than trusting `transformers_version` in a model JSON.

For every MLX packed-fixture and 27B reference run, record the device/stream, Metal availability,
eval/training mode, explicit `use_kernel` setting, sequence length, compute/state dtypes and
actual selected gated-delta callable with its source hash. mlx-vlm 0.6.8's default GPU route
uses its own `gated_delta_kernel`, constructed through `mx.fast.metal_kernel`; it does not
call MLX's built-in `mx.fast.gated_delta_update`. Off GPU or with `use_kernel=False`, its
dispatcher selects `gated_delta_chunked` for multi-token scalar gating, otherwise
`gated_delta_ops`. Record the selected path for prefill and decode separately; a CPU/ops run
cannot be reported as evidence for the default GPU kernel. Probe these dispatch choices in WP0.

Extend each prefill/decode record to its convolution and gated-norm callables, source hashes,
input/weight/accumulation dtypes and casts. In the inspected mlx-vlm code, eligible single-token
decode with BF16/FP16 convolution weights uses `_qwen3_5_decode_depthwise_conv` with float32
accumulation; prefill and other cases use `nn.Conv1d`. Record which condition actually applies,
including the FLOAT32-weight comparison path, rather than assuming every decode uses the helper.
The gated RMSNorm uses `_precise_swiglu`, evaluating SiLU and multiplication in float32 before
casting back; record its use in both phases. These records support attribution of per-step
differences even when the gated-delta kernel is the same. Batch-one greedy runs have no
left-padding or speculative verification: ragged left-padded attention and target-verify
kernels are excluded from this reference matrix.

Separately inspect and probe MLX 0.32.4's built-in `mx.fast.gated_delta_update` (the C binding
is `mlx_fast_gated_delta_update`). Record its own supported head widths/grouping, sequence
lengths, dtypes, initial-state orientation and mask/device requirements, with accepted/rejected
boundary cases. Its limits have not been checked; do not transfer mlx-vlm's custom-kernel
constraints to it. A shape it rejects is recorded as ineligible for that benchmark candidate,
not silently switched to another implementation under the native-op label.

WP0 must isolate the Q/K normalization convention difference before setting MLX comparison
tolerances. Java and HF retain `x / sqrt(sum(x*x) + 1e-6)` (plus query head scaling).
mlx-lm 0.32.2 rescales its RMSNorm epsilon to match that formula; mlx-vlm 0.6.8 instead passes
`1e-6` directly, giving effective sum-space epsilon `Dk * 1e-6`. That is `32e-6` in the tiny
fixture and `128e-6` in both real artifacts. Its numerical impact has not been measured.

For the tiny float/packed fixtures and both 27B prompt cases, run paired unmodified mlx-vlm and
an explicitly labeled normalization-only diagnostic with RMSNorm epsilon `1e-6 / Dk`, retaining
the same Q/K scale multipliers. Preserve the diagnostic diff/source hash and exact argument
settings separately from the unmodified source. This attribution experiment must never produce
goldens or acceptance results. Use identical stored weights, precision, processed patches and
explicit teacher-forced token histories, in fresh sessions with other execution paths held fixed.
Record changes in normalized Q/K, recurrent outputs/final state and per-step logits, plus
Java-vs-stock and Java-vs-aligned logit errors and each run's greedy choices/margins. Account
for that convention contribution separately from remaining implementation/kernel differences.

The unmodified reference still owns the packed-fixture and 27B acceptance gates. Do not
silently absorb this known mathematical difference into a generic epsilon, loosen HF bounds,
change Java normalization, or report the diagnostic's IDs as stock mlx-vlm parity. If the
recorded stock comparison cannot satisfy its declared bound or exact-ID gate, stop for review
with the paired evidence rather than replacing the reference or assuming BF16 hides the effect.

If a lock cannot resolve or the reference cannot execute, record the failure and revise the
proposed pin in this plan before producing fixtures. Do not silently upgrade MLX. Do not patch
the reference math, except for the separately labeled, source-hashed normalization diagnostic
above, which must never produce goldens or acceptance results. Do not replace converter output
with hand-written Java-layout files.

Use seed 20261102 and a four-layer hybrid model with three linear layers and one full-attention
layer: hidden width 128, MLP width 256, two full-attention heads / one KV head, head width 64,
partial rotary width 16, MRoPE sections `[3,3,2]`, two linear key heads / six value heads,
linear key width 32 and value width 64, and an untied 128-token output vocabulary. The key
width must be at least 32 and divisible by 32 for the pinned mlx-vlm custom GPU delta kernel;
the previous 16-wide proposal would leave its `Dk / 32` loops empty. Keep unequal key/value
widths so the fixture still tests that distinction. This is not a verified shape rule for
MLX's separate built-in op. Add an
equal-head/tied-output variant and an eight-layer variant to test state indexing. A tiny
vision config uses two layers, width 64, four heads, intermediate width 128, patch 16,
temporal size 2, merge 2, no DeepStack and 64 learned positions, projecting to text width 128.
Remap special IDs into this vocabulary: pad 0, BOS 1, EOS 2, vision-start 118, vision-end 119,
image 120 and video 121 (video input is still rejected). Pin these IDs, their token spellings
and the paired tokenizer in the fixture config and goldens; do not inherit the real artifacts'
248053–248057 vision IDs. Validate distinctness and vocabulary bounds before reference export
and Java loading. Round source weights to BF16 before
computing the HF-layout golden, then up-cast to float32 for both reference and Java execution.
For float converted-layout equality, separately export a float32 fixture through each converter
so norm-offset conversion does not introduce BF16 re-rounding into that comparison.

## 3. Model composition and integration boundaries

Add `Qwen35Model implements TextGenerationModel`, a strict `Qwen35Descriptor`, a pure tensor
layout/preflight component and a request-owned hybrid execution session in `jmlx-models`.
Dispatch `qwen3_5` before generic `ArchitectureMappings`. Do not pretend its recurrent state is
a `List<KVCache>` or make it a `DecoderModel` solely to pass scheduler admission.

Core components are `GatedDeltaNet`, `RecurrentCache`, `GatedDecoderAttention` and
`HybridDecoderBlock` under `se.alipsa.jmlx.nn`. They own MLX module math and state through core
facades. `jmlx-models` selects components and maps weights; it never calls generated FFI.
Existing `DecoderAttention`/`DecoderBlock` entry points retain their behavior. Reuse `Linear`,
`QuantizedLinear`, `RMSNorm`, `SwiGLU`, `KVCache`, `MultiAxisRope` and the vision tower where their
contracts agree with the reference; do not relax Qwen3-VL-specific loader guards globally.

Extract the sampling/event portion of `DecoderModel.runStepLoop` into a package-private
`GenerationLoop` accepting a request-state adapter. The existing decoder entry points delegate
through a KV adapter; the hybrid adapter owns recurrent and full-attention state. Keep scope
creation, evaluation, penalties, seeded sampling, cancellation, finish reasons, abort reports,
listener behavior and original prompt-ID reporting consistent. This extraction has its own
commit and old-path exact-bit regression checks in both precision modes before hybrid use.
Avoid exposing a new general public cache/scheduler API in this milestone.

## 4. Recurrent attention and full attention

### DeltaNet math

Project Q/K/V, the output gate `z`, and scalar-head projections `a` and `b`. Causal depthwise
conv1d followed by SiLU acts on Q/K/V only, before their head reshape. Preserve the latest
`kernelSize - 1` real projected input positions; convolution padding is not real history.
Repeat each key/query head contiguously to its value-head group. L2 normalization is
`x / sqrt(sum(x*x) + 1e-6)`, followed by query scaling `1/sqrt(keyHeadDim)`.
This HF convention remains authoritative for Java. The mlx-vlm sum-space epsilon is larger;
the WP0 paired diagnostic measures that discrepancy independently of kernel selection.

Use float32 state and compute `beta = sigmoid(b)` and
`g = -exp(A_log) * softplus(a + dt_bias)`, with stable softplus composed from existing ops.
For one token and state `S` shaped `[B,Hv,Dk,Dv]`:

```text
Sdecay = exp(g) * S
delta  = beta * (v - k^T Sdecay)
Snext  = Sdecay + k outer delta
y      = q^T Snext
```

Normalize `y` per value head with the ordinary learned norm weight, then multiply by SiLU(z)
and apply `out_proj`. This gate is distinct from the full-attention sigmoid gate. Never add
the zero-centered norm offset to `linear_attn.norm.weight`.

Use 64-token chunked prefill with cumulative log decay, causal masks and batched matmuls.
The 2026-11-03 WP1 review selects chunked ops as the prefill algorithm and defers custom Metal
for this milestone; its 115 ms reference result motivates future work, not integration.
The built-in delta op remains excluded under the unchanged numerical gate. Before WP2, the
structural optimization pass must requalify at `1e-4` and target about twice mlx-vlm's own
chunked-ops cost: approximately 600 ms for 48 calls at 1,280 tokens on the recording host.

Invert the 64×64 unit-lower-triangular system by stable forward substitution, then apply it
to the right-hand side with a batched matmul. Batch state-independent preprocessing across at
most 16 chunks (1,024 positions), retaining the 64-position state scan. Repeated-squaring
inversion is not selected: the correlated-key FLOAT32 stress experiment is numerically unstable.
No linalg binding, kernel facade or API-inventory relaxation is required. Mask forbidden
log-decay entries before exponentiating to avoid overflow. Tail padding must be recurrence-neutral and removed from outputs; retain convolution
history from real tokens only. Evaluate/hoist preprocessing results and at most eight chunks
of carried state/output before closing their temporary scopes, so a prompt does not retain every earlier chunk's graph.
Decode has a dedicated single-token recurrence without per-layer evaluation, final concatenation
or output transpose. Cache replacement retains provisional lazy state; the session evaluates
logits and every retained state root once at the token boundary, and poisons the whole session
on evaluation failure before committing its position.

Compare chunked outputs/final state to both HF and a simple token recurrence, for lengths
1, 3, 4, 5, 63, 64, 65, 127, 128 and 129, including nonzero initial state and split prefills.
WP1 must compare four separately named paths before selecting the production prefill default:
chunked ops, token recurrence, mlx-vlm 0.6.8's custom `gated_delta_kernel` (the default GPU
reference path), and MLX 0.32.4's built-in `mlx_fast_gated_delta_update` (initially measured
through `mx.fast.gated_delta_update` in the isolated reference tool, where WP0 verified shape
support). Label mlx-vlm's off-GPU/`use_kernel=False` chunked/ops fallback separately from
its custom kernel; neither is an alias for the built-in native op. Compare identical prepared
Q/K/V, decay, beta and initial state, accounting explicitly for native gate convention and
state orientation; all candidates must pass output/final-state checks first. Run tiny shapes
and the real 0.8B/27B head dimensions at lengths 64, 256, 1024 and 1280, including the 27B's
48-linear-layer execution cost. Record native-call/dispatch counts, evaluated wall time,
temporary memory, and first-use versus warmed results under the host-state methodology.

Per-chunk evaluation bounds graph retention, not dispatch overhead: the ops substitution can
multiply small slice/update operations across rows, chunks and layers. HF normally uses
`solve_triangular`; its export fallback has 63 row iterations, batched over chunks. Neither
that source inspection nor an operation-count estimate is a measured slowdown. Record the
comparison and proposed default for review at the WP1 gate. If native acceleration wins,
add a separately reviewed core facade and inventory mapping before using it from Java;
custom Metal kernels still require measured evidence and a plan amendment.

### Gated full attention and partial MRoPE

Reshape `q_proj` per query head before splitting query and gate; splitting the flattened output
into two global halves would pair the wrong values. Apply zero-centered Q/K norm, partial
MRoPE, causal attention, then sigmoid(gate), then `o_proj`. Use the configured head dimension,
not `hiddenSize / numHeads`. Layer selection follows validated `layer_types`; when an interval
is also supplied, require agreement rather than silently overriding one with the other.

Require `text_config.rope_parameters` with `rope_type: default`, `mrope_interleaved: true`
and an explicit numeric `partial_rotary_factor`; other types/layouts and missing flags are
rejected by name. The nested factor is canonical. An optional duplicate at the top of
`text_config` must be numeric and equal to it; disagreement is rejected. If the nested factor
is absent, reject even if the top-level alias exists. This deliberately supports the observed
explicit configs rather than guessing between HF model/config defaults of 1.0 and 0.25.
WP0 records nested-only, matching/conflicting duplicates and missing-factor cases against HF;
Java tests pin the strict acceptance/rejection policy separately from HF's defaulting behavior.

For this family require `rotaryDim = headDim * partial_rotary_factor` to be a positive even
integer and `2 * sum(mrope_section) == rotaryDim`. Both real artifacts have 256-wide heads
and a 64-wide rotary prefix. The inverse-frequency denominator is the rotary width; the tail
passes through unchanged. Add differential prefix/tail tests against the pinned HF reference
at unequal T/H/W positions, then update `MultiAxisRope`'s current extension-only Javadoc.
The Qwen3-VL equality to the entire head width remains unchanged.

The inspected HF model does not read `output_gate_type`; its full-attention path applies
sigmoid. Accept only the observed absent/`swish` metadata values initially. WP0 must compare
the same weights/config with that key present and absent and record which gate it describes;
do not replace sigmoid with SiLU just because the artifact carries the string `swish`.

## 5. Checkpoint classification and loading

Classify from the complete shard-header key/shape/dtype inventory and config before
allocating weights. Quantization and layout are independent decisions. Detect/reject mixed
prefixes, colliding remaps, partial vision inventories and ambiguous norm-offset conventions.
The loader does not require a model-card README, converter name/version or an external
reference-tool provenance manifest. The inspected config/index do not identify the converter;
shard `__metadata__` has not been inspected. WP0 records it if present, but it does not override
the structural rules below. Converter/version records are reproducibility evidence for the
reference fixtures and manual acceptance, not an assumed input available to every loader.

| Layout | Text mapping | Conv axes | Vision handling | Zero-centered text norm |
| --- | --- | --- | --- | --- |
| HF | `model.language_model.*`, optional untied `lm_head.*` | `[C,1,K]` to `[C,K,1]` | `model.visual.*`; patch `[O,C,T,H,W]` to channels-last | add 1 exactly once |
| mlx-vlm | `language_model.model.*`, `language_model.lm_head.*` | already `[C,K,1]` | `vision_tower.*`; detect patch axes by shape | already shifted for a sanitized artifact |
| mlx-lm | `language_model.model.*`, `language_model.lm_head.*` | already `[C,K,1]` | absent: TEXT capability only | already shifted for a sanitized artifact |

The inspected mlx-lm wrapper uses the same text namespace as mlx-vlm. Loader capability comes
from the vision inventory, not an inferred converter identity or a fabricated prefix split.
WP0 verifies the actual saved namespace from each converter before loader implementation.

Apply these rules over the complete inventory, **before** dropping any MTP tensors:

- HF text prefix `model.language_model.*` with every conv `[C,1,K]`: treat the selected text
  norms as unshifted and add 1 exactly once. Recognized HF `mtp.*` tensors may be inventoried
  and skipped before allocation; they never execute.
- Converted text prefix `language_model.model.*` with every conv `[C,K,1]` and no MTP tensors
  (including prefixed MTP subtrees): treat selected text norms as already shifted; add nothing.
- Reject mixed prefixes, mixed conv axes, converted prefixes with any MTP subtree, or any
  other unsupported combination. Never guess from a single tensor or sanitize twice.

These define supported storage conventions, not proof that arbitrary tensor values were
produced by a particular converter. Shape-detect patch axes independently for float and affine
inputs alike. Unknown extras and colliding mappings are rejected by name. Record the exact
pinned converter sanitize behavior in fixture provenance and check its saved outputs against
these rules; do not require that provenance to be packaged with third-party checkpoints.

Validate required/optional tensors, dimensions, ties and aliases through `TensorPlan`/preflight.
Accept the observed legacy `vision_config.model_type: qwen3_5` only as this family's known vision
alias. Validate typed generation metadata without treating it as a runtime policy override.
Accept `language_model_only` only when absent or boolean `false`; reject `true` and non-booleans
by name in this milestone. It is typed inert metadata, not the capability selector: `false`
does not guarantee that vision tensors exist. A complete tower means TEXT+IMAGE, no vision
tensors means TEXT, and a partial tower is rejected, regardless of quantization. Thus the
supported text-only converter layout does not rely on setting this optional flag to `true`.
Check context/layer schedules, gate flags, epsilon, head divisibility and grouped linear heads.
All integer products/size accounting use overflow checks. Unsupported MoE and quantization
forms fail before weight upload.

Float weights load as FLOAT32. Uniform affine 4-bit/group-64 packed projections retain U32
weights and FLOAT32 scale/bias sidecars, reusing existing `QuantizedLinear` checks. Do not
quantize convolution, norm or recurrent state merely because neighboring projections are
packed. The 27B acceptance compares the same stored weights and stated compute precision in
Java and mlx-vlm; record any difference from mlx-vlm's default precision separately.

## 6. Cache, ownership, generation and vision

A request session holds one recurrent cache per linear layer, one KV cache per full-attention
layer and a single real-token position count. State and positions advance together only on
successful execution. Failure poisons that session; no partially updated session is reused.
The convolution state is `[B,K-1,C]`, recurrence `[B,Hv,Dk,Dv]`, both FLOAT32. The latter
is the canonical Java orientation; record transposes against each reference explicitly.

FULL and capacity-bounded FULL are supported for text and image requests. Capacity limits the
expanded prompt plus possible decode history, even though recurrent state is fixed-size.
Reject sliding before pixel processing or native inference. A recurrent cache cannot be
truncated: future Phase 9.2 prefix reuse must snapshot the entire state at an exact prefix
boundary, together with all full-attention K/V, convolution history and position/MRoPE delta.
No tail-trimming or public prefix-cache API is introduced here.

The request scope owns evaluated retained state. Per-chunk, per-layer and per-step children
own temporaries; evaluate all new retained arrays, not just logits, before closing them.
Retire replaced state promptly. Model parameters never retain request graphs. Cancellation is
checked before work, between images/chunks and between decode steps. A zero-token request
still validates prompt geometry, capability and capacity without running tower/prefill/decode.

Reuse `Qwen3VlImageProcessor` with the artifact's patch-16 config and mean/std `[0.5,0.5,0.5]`,
subject to WP0 proving its contract under the new reference. Keep the existing 4096 raw-row
budget and effective cap; record those explicit settings in every reference run. The reference
image target remains the explicitly selected slow PIL processor, not an assumed equivalence
with the artifact's Fast declaration. Record a fast/slow comparison; Java, HF and the manual
MLX comparison consume the same digest-pinned processed patches and rendered prompt/IDs.
If the new slow processor differs, document/port the difference before claiming reuse.

Reuse `Qwen3VlVisionTower` with no DeepStack and shape-selected patch axes. During WP5 remove
the tower constructor's non-empty-index restriction and update its Javadoc to permit an empty
array with no DeepStack merger subtrees and an empty feature-level list. Keep non-null,
index-range/order and exact weight-inventory checks. Preserve the existing non-empty guard in
`Qwen3VlDescriptor.parseDeepstackIndexes` and explicitly enforce it in every `Qwen3VlModel.load`
path before constructing the shared tower. `Qwen35Descriptor` instead requires an empty array.
Move the empty-array rejection currently asserted by
`Qwen3VlVisionTowerValidationTest.constructorRejectsInvalidDeepstackIndexes` to Qwen3-VL
descriptor/loader rejection tests. Replace that constructor assertion with a positive shared-tower
empty-levels case using weights with no DeepStack merger subtrees; leave its non-empty invalid
index/range/order cases intact. Add tests for both family guards and zero-iteration feature handling;
existing Qwen3-VL captures, logits and generation must remain unchanged.

Both real artifacts have 2304 learned positions (48×48), the same grid size as Qwen3-VL-2B;
only the tiny fixture has a smaller 64-position grid. Reverify interpolation and merger at
both grid sizes against HF; shared class names are not
proof. Prompt expansion retains original IDs in results, charges expanded positions, replaces
image sentinel rows once and carries the per-request MRoPE delta into decode. Release per-image
tower intermediates sequentially and image features after evaluated prefill state no longer
depends on them. Text-only execution through a vision-capable artifact must match its bare
text path. Public video and remote-image fetching stay rejected.

## 7. Ordered work packages and evidence

| WP | Deliverable and commit boundary | Gate before the next package |
| --- | --- | --- |
| 0 | Isolated locks, artifact/header inventories, reference/config/converter, dispatch/shape and normalization-attribution probes | references execute; normalization contribution measured; convolution/gated-norm/delta paths, built-in-op limits, rotary/gates, norm offsets, state orientation and preprocessing recorded |
| 1 | Core recurrent cache, token recurrence and chunked DeltaNet prototypes; custom-kernel and built-in-op comparisons | outputs/state match independent references; graphs bounded; four-path prefill benchmark and proposed default reviewed before selection |
| 2 | Gated full attention, partial-MRoPE tests and hybrid blocks | strict component goldens; existing decoder paths unchanged |
| 3 | Pure descriptor, tensor-plan and layout classifier; actual converted tiny fixtures | float layouts agree; malformed layouts fail before allocation; packed fixture matches stock MLX with normalization-only diagnostic reported separately |
| 4 | Separate generation-loop extraction, hybrid text loader/generation | both-mode old-path exact-bit checks; text Tier-A prefill/cached/fresh-forward equality |
| 5 | Vision composition, prompt expansion and text-only converted capability | multi-image and text-only Tier-A; processor/patch goldens; capacity and metadata contracts |
| 6 | Ownership, cancellation/failure injection, memory and benchmark evidence | retained-state accounting, repeated-request cleanup, bounded temporary/chunk graphs |
| 7 | 0.8B Tier-B and manual 27B affine text/image acceptance | exact IDs/output, logit stability, precision/resource/host/path provenance; stock-vs-aligned normalization evidence kept distinct |
| 8 | Compatibility/docs and final native CI | required native suites really execute; all completion gates above met |

For WP0 commit complete Hub file digests/header shapes and downloaded-source hashes under the
new tool's provenance, not only the index JSON. This planning pass inspected metadata and
source; it did not download weights, run converters or generate logits. Converted fixtures
must come from each package's actual converter entry point, with source input hashes, converter
arguments and output inventory. Inspect the saved artifact and compare it after reloading;
sanitizing an in-memory dict is not a conversion/reload test.

## 8. Validation and reporting

Add `full-float32` component/model suites in the existing core/models golden tasks, initially
using absolute tolerance `1e-4` for strict HF logits. Preserve all existing bounds. Test default
TF32 separately with a named, measured bound; never regenerate goldens to accommodate Java.
Converted affine comparison and real-artifact tolerances are recorded independently before
their Java acceptance run, with reference margin eligibility (`gap > 2*epsilon + margin`).
Those records must identify the normalization convention, its measured contribution and
remaining Java/reference error separately; the diagnostic never silently replaces stock MLX.
An ineligible prompt is not a passing exact-ID test; preserve the failed measurement and select
a documented eligible case before acceptance.

Add pure-Java `verifyQwen35ReferenceGoldens` to root `check` for the new provenance and source,
fixture, converter and shared-patch digests. Python generators remain explicit manual commands.
Any changed `buildSrc` helper requires its separate `./gradlew -p buildSrc check` run.

New required native suites include DeltaNet math/state, gated attention/partial rotary, hybrid
model goldens in FLOAT32, default-precision inference, generation contracts and memory ownership.
Add each real test class/result directory to CI's required-execution assertion when introduced.
Use `@EnabledIfNativeAvailable` consistently. The 0.8B Tier-B manifest gets a task-specific
suite and workflow row; the 27B run remains a resource-qualified manual acceptance, not a hosted
runner claim. Reuse the existing Tier-B download/hash infrastructure.

Record logical memory separately from measured allocator/process peaks. With batch one and
FLOAT32, recurrent-plus-convolution storage derives from
`linearLayers * 4 * (Hv*Dk*Dv + (K-1)*(2*Hk*Dk + Hv*Dv))`. Full-attention KV grows by
`fullLayers * 2 * Hkv * headDim * 4` bytes per real position. These are accounting formulas,
not allocator bounds. Measure and justify overhead/headroom independently. Repeated generations,
longer prompts and cancellation/fault paths must not retain earlier request or chunk graphs.

Benchmark first inference separately from warmed prefill/decode, with evaluated outputs/state,
alternating comparisons and timestamped per-run host observations. State the limit when a host
sample is missing. PR #76's investigation is not proof about every contention mix or first use.

Update `req/phase6-compatibility.md`, inference/precision/Tier-A/Tier-B reports, the model and
vision READMEs, parent/roadmap status and AGENTS.md when their corresponding evidence lands.
Leave unrelated pending Phase 7 acceptance gates visible. Phase 7 is complete only when the
parent's full gate is satisfied; delivery of 7.5 or synthetic 7.6 tests alone cannot close it.
